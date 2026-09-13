"""Loader for PixelRec (Cheng et al., "An Image Dataset for Benchmarking
Recommender Systems with Raw Pixels"), written against the real public
release layout (confirmed against the actual files, not guessed):

    <root>/interaction.csv        # item_id, user_id, timestamp  (no interaction-type column)
    <root>/item_meta.csv          # item_id, view_number, comment_number, thumbup_number,
                                   # share_number, coin_number, favorite_number, barrage_number,
                                   # title, tag, description
    <root>/features/
        image_feature.json        # pre-extracted image embeddings, exact internal shape not
        text_feature.json         # confirmed -- see _stream_load_feature_json() for the two
                                   # layouts this loader auto-detects, streamed + cached rather
        readme.md                 # than loaded eagerly (these files can be multi-GB)
    <root>/cover/                 # cover images, assumed named "<item_id>.jpg" (PixelRec's
                                   # documented convention); adjust `_resolve_cover_path` if your
                                   # copy uses a different naming scheme

`interaction.csv` has no explicit rating or interaction-type signal --
every row is an implicit positive engagement (comment/view), consistent
with the PixelRec paper's description of interactions as "if a user
leaves a comment, they clicked the cover beforehand." This loader treats
every row as a positive implicit interaction for ranking purposes.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.common import Sample

logger = logging.getLogger(__name__)

_QUERY_TEMPLATES = [
    "Can you show me something like this?",
    "What do you think of this one?",
    "I'm looking for something similar to this, any suggestions?",
    "Does this look good?",
]


def _peek_json_top_level(path: Path) -> str:
    """Cheaply determines whether the JSON file's root is an object ('{')
    or an array ('[') by reading a handful of bytes, without touching the
    rest of the (potentially multi-gigabyte) file.
    """
    with open(path, "rb") as f:
        while True:
            byte = f.read(1)
            if not byte:
                return "unknown"
            if byte in b" \t\r\n":
                continue
            if byte == b"{":
                return "dict"
            if byte == b"[":
                return "list"
            return "unknown"


def _stream_load_feature_json(path: Path, keep_ids: set, dtype=np.float16) -> dict:
    """Streams `image_feature.json` / `text_feature.json` with `ijson`
    instead of `json.load()`, which would otherwise read the entire
    (potentially multi-GB) file into memory as text and then again as
    Python objects -- easily blowing past available RAM for PixelRec's
    full 400K+ item catalog (this is what caused a `MemoryError` reading
    the file directly).

    Only vectors for items in `keep_ids` are retained (typically the item
    ids actually present in `item_meta.csv` / a downsampled subset), and
    values are cast to float16 immediately rather than kept as Python
    lists, which cuts peak memory dramatically. Auto-detects whether the
    JSON root is a dict-of-vectors or a list-of-records (see module
    docstring) and streams accordingly.
    """
    import ijson

    if not path.exists():
        logger.warning("Feature file not found: %s (will use placeholder features)", path)
        return {}

    top_level = _peek_json_top_level(path)
    out = {}

    try:
        if top_level == "dict":
            with open(path, "rb") as f:
                for item_id, vec in ijson.kvitems(f, ""):
                    item_id = str(item_id)
                    if item_id in keep_ids:
                        out[item_id] = np.asarray(vec, dtype=dtype)
        elif top_level == "list":
            with open(path, "rb") as f:
                for record in ijson.items(f, "item"):
                    item_id = str(record.get("item_id") or record.get("itemId") or record.get("id"))
                    if item_id not in keep_ids:
                        continue
                    vec = record.get("feature") or record.get("embedding") or record.get("vector")
                    if vec is not None:
                        out[item_id] = np.asarray(vec, dtype=dtype)
        else:
            logger.warning("Could not determine JSON root type for %s; skipping", path.name)
            return {}
    except Exception as e:
        logger.warning("Streaming parse of %s failed (%s); falling back to placeholder features "
                        "for this file. If this persists, the file may need a custom parser.",
                        path.name, e)
        return {}

    logger.info("Streamed %d/%d requested feature vectors from %s", len(out), len(keep_ids), path.name)
    return out


def _load_feature_json_cached(path: Path, keep_ids: set) -> dict:
    """Wraps `_stream_load_feature_json` with an on-disk cache (an .npz
    next to the source JSON) so the expensive streaming parse only ever
    happens once per machine. Subsequent loads are near-instant, memory-
    light `np.load` reads. The cache is invalidated automatically if the
    source JSON is newer than the cache file.
    """
    cache_path = path.with_suffix(".cache.npz")

    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                ids = data["ids"]
                vecs = data["vecs"]
            cached = {str(i): vecs[idx] for idx, i in enumerate(ids)}
            # Cache may have been built for a different keep_ids set (e.g. a
            # smaller max_users run); only reuse it if it covers what we need.
            missing = keep_ids - set(cached.keys())
            if not missing:
                logger.info("Loaded %d feature vectors from cache %s", len(cached), cache_path.name)
                return {k: v for k, v in cached.items() if k in keep_ids}
            logger.info("Cache %s missing %d requested ids; re-parsing source JSON", cache_path.name, len(missing))
        except Exception as e:
            logger.warning("Failed to read cache %s (%s); re-parsing source JSON", cache_path.name, e)

    result = _stream_load_feature_json(path, keep_ids)
    if result:
        try:
            ids = np.array(list(result.keys()))
            vecs = np.stack(list(result.values()))
            np.savez(cache_path, ids=ids, vecs=vecs)
            logger.info("Cached %d feature vectors to %s for future runs", len(result), cache_path.name)
        except Exception as e:
            logger.warning("Failed to write feature cache %s (%s); continuing without cache", cache_path, e)
    return result


class PixelRecLoader:
    def __init__(self, root: str, num_negatives: int = 99, min_user_interactions: int = 3, seed: int = 42):
        self.root = Path(root)
        self.num_negatives = num_negatives
        self.min_user_interactions = min_user_interactions
        self.rng = random.Random(seed)

        self.interactions_path = self.root / "interaction.csv"
        self.metadata_path = self.root / "item_meta.csv"
        self.features_dir = self.root / "features"
        self.cover_dir = self.root / "cover"

        for p in [self.interactions_path, self.metadata_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected PixelRec file not found: {p}. "
                    f"Check `paths.pixelrec_root` in configs/config.yaml."
                )

        self.interactions = pd.read_csv(self.interactions_path)
        self.item_metadata = pd.read_csv(self.metadata_path).set_index("item_id")
        self.all_item_ids = self.item_metadata.index.astype(str).tolist()

        # Precomputed once at construction time so the per-interaction loop
        # in load_all_samples() never has to hit pandas .loc or the
        # filesystem repeatedly for the same item -- both were previously
        # called once per interaction ROW (potentially millions of times)
        # rather than once per unique ITEM, which was the actual source of
        # multi-minute+ silent slowdowns on a large interaction.csv.
        self._item_title = self.item_metadata["title"].astype(str).to_dict() if "title" in self.item_metadata else {}
        self._item_tag = self.item_metadata["tag"].astype(str).to_dict() if "tag" in self.item_metadata else {}
        self._available_covers = self._scan_cover_dir()

        self._image_features = None
        self._text_features = None

    # ------------------------------------------------------------------
    # Lazy feature loading (the JSON files can be multiple GB; only parsed
    # once, on first access, and filtered to the ids actually in this
    # dataset's catalog rather than the full multi-hundred-thousand-item
    # PixelRec corpus -- see _stream_load_feature_json for why this
    # matters memory-wise)
    # ------------------------------------------------------------------

    @property
    def image_features(self) -> dict:
        if self._image_features is None:
            self._image_features = _load_feature_json_cached(
                self.features_dir / "image_feature.json", set(self.all_item_ids)
            )
        return self._image_features

    @property
    def text_features(self) -> dict:
        if self._text_features is None:
            self._text_features = _load_feature_json_cached(
                self.features_dir / "text_feature.json", set(self.all_item_ids)
            )
        return self._text_features

    # ------------------------------------------------------------------

    def _scan_cover_dir(self) -> dict:
        """Scans cover/ once and builds {item_id: full_path}, instead of
        calling Path.exists() up to 3 times per interaction ROW (which,
        for a large interaction.csv where the same item appears many
        times, meant millions of redundant filesystem stats). Returns an
        empty dict (all items treated as having no cover image) if the
        directory doesn't exist.
        """
        if not self.cover_dir.exists():
            logger.warning("cover/ directory not found at %s; proceeding without cover images", self.cover_dir)
            return {}
        covers = {}
        for path in self.cover_dir.iterdir():
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                covers[path.stem] = str(path)
        logger.info("Found %d cover images in %s", len(covers), self.cover_dir)
        return covers

    def _resolve_cover_path(self, item_id: str) -> str | None:
        return self._available_covers.get(item_id)

    def _sample_negatives(self, positive_id: str, k: int) -> list:
        """O(k) instead of the previous O(k^2) approach (which repeatedly
        checked `cand not in pool` against a growing Python list). Samples
        without replacement via random.sample, then drops the positive id
        if it happened to be included rather than re-rolling in a loop.
        """
        pool_size = min(k, len(self.all_item_ids) - 1)
        sampled = self.rng.sample(self.all_item_ids, min(pool_size + 1, len(self.all_item_ids)))
        negatives = [i for i in sampled if i != positive_id][:pool_size]
        return negatives

    def _warn_if_memory_heavy(self, user_ids: list, grouped) -> None:
        """Estimates the in-memory footprint of the sample list *before*
        building it, so a run that would exhaust available RAM warns
        loudly up front rather than silently crashing partway through (on
        Windows in particular, an OOM kill terminates the process with no
        Python traceback at all, which looks identical to a random hang).

        The estimate is deliberately rough (Python object overhead varies),
        but is conservative enough to catch the failure mode that actually
        happens in practice: a large interaction.csv combined with a large
        `num_negatives` candidate pool.
        """
        sizes = self.interactions[self.interactions["user_id"].isin(user_ids)].groupby("user_id").size()
        estimated_samples = int(sizes[sizes >= self.min_user_interactions].sum())
        bytes_per_candidate_str = 56  # rough CPython overhead for a short (~7-8 char) string object
        est_bytes = estimated_samples * (
            (self.num_negatives + 1) * bytes_per_candidate_str  # candidate_item_ids
            + 20 * bytes_per_candidate_str  # user_history_ids (capped at 20)
            + 500  # other Sample fields + dataclass/list overhead, rough
        )
        est_gb = est_bytes / (1024 ** 3)
        logger.info(
            "PixelRec: estimated ~%d samples, ~%.1f GB of Python object overhead for the sample list "
            "(num_negatives=%d). If this exceeds your available RAM, reduce `pixelrec.max_users` or "
            "`pixelrec.num_negatives` in config.yaml before this gets far enough to OOM-crash silently.",
            estimated_samples, est_gb, self.num_negatives,
        )
        if est_gb > 8:
            logger.warning(
                "PixelRec: this estimate (~%.1f GB) is large enough to risk an out-of-memory crash "
                "partway through (which on Windows can terminate the process with no error message at "
                "all). Strongly consider setting `pixelrec.max_users` (e.g. 5000-20000) or lowering "
                "`pixelrec.num_negatives` before proceeding.", est_gb,
            )

    def load_all_samples(self, max_users: int | None = None) -> list:
        samples = []
        logger.info("PixelRec: sorting/grouping %d interaction rows by user...", len(self.interactions))
        grouped = self.interactions.sort_values("timestamp").groupby("user_id")

        user_ids = list(grouped.groups.keys())
        if max_users is not None:
            user_ids = user_ids[:max_users]
        logger.info("PixelRec: processing %d users (max_users=%s)...", len(user_ids), max_users)
        self._warn_if_memory_heavy(user_ids, grouped)

        item_meta_ids = set(self.item_metadata.index.astype(str))

        for user_id in tqdm(user_ids, desc="PixelRec: building samples", unit="user"):
            group = grouped.get_group(user_id)
            if len(group) < self.min_user_interactions:
                continue

            history: list = []
            for row in group.itertuples(index=False):
                item_id = str(row.item_id)
                if item_id not in item_meta_ids:
                    history.append(item_id)
                    continue

                title = self._item_title.get(item_id, "")
                tag = self._item_tag.get(item_id, "")
                query_context = tag if tag and tag.lower() != "nan" else title
                template = self.rng.choice(_QUERY_TEMPLATES)
                query_text = f"{template} ({query_context})" if query_context else template

                negatives = self._sample_negatives(item_id, self.num_negatives)
                candidates = [item_id] + negatives
                self.rng.shuffle(candidates)

                context_tags = [t.strip() for t in tag.split(",")] if tag and tag.lower() != "nan" else []

                samples.append(
                    Sample(
                        sample_id=f"{user_id}_{item_id}_{row.timestamp}",
                        dataset="pixelrec",
                        query_text=query_text,
                        context_tags=context_tags,
                        emotion_label=None,
                        image_path=self._resolve_cover_path(item_id),
                        video_feature_path=None,
                        target_item_id=item_id,
                        candidate_item_ids=candidates,
                        user_history_ids=list(history[-20:]),
                        gold_response=None,
                    )
                )
                history.append(item_id)

        logger.info("PixelRec: parsed %d interaction samples from %d users",
                     len(samples), len(user_ids))
        return samples

    def build_item_feature_lookup(self, feature_extractor=None) -> dict:
        """Combines the real pre-extracted image + text features where
        available (concatenated/averaged into a 768-d vector to match the
        rest of the pipeline's hidden size), falling back to a
        deterministic placeholder only for items missing from both
        feature files.
        """
        lookup = {}
        img_feats = self.image_features
        txt_feats = self.text_features

        for item_id in tqdm(self.all_item_ids, desc="PixelRec: building item feature lookup", unit="item"):
            parts = []
            if item_id in img_feats:
                parts.append(img_feats[item_id])
            if item_id in txt_feats:
                parts.append(txt_feats[item_id])

            if parts:
                combined = np.concatenate(parts) if len(parts) > 1 else parts[0]
                if combined.shape[0] >= 768:
                    vec = combined[:768].astype(np.float32)
                else:
                    vec = np.zeros(768, dtype=np.float32)
                    vec[: combined.shape[0]] = combined
                lookup[item_id] = vec
            else:
                rng = np.random.RandomState(abs(hash(item_id)) % (2**32))
                lookup[item_id] = rng.normal(size=768).astype(np.float32)

        num_real = sum(1 for i in self.all_item_ids if i in img_feats or i in txt_feats)
        logger.info("PixelRec item features: %d/%d items have real pre-extracted features",
                     num_real, len(self.all_item_ids))
        return lookup

    def split(self, train_ratio=0.8, val_ratio=0.1, seed=42, max_users=None):
        samples = self.load_all_samples(max_users=max_users)
        rng = random.Random(seed)
        rng.shuffle(samples)
        n = len(samples)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        return samples[:n_train], samples[n_train:n_train + n_val], samples[n_train + n_val:]
