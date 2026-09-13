"""Loader for the VOGUE human-human fashion conversational recommendation
dataset -- written against the *actual* on-disk schema (confirmed by
inspecting real transcript and ratings files), not a guessed one.

Confirmed on-disk layout:

    <root>/conversation_trials/transcripts/c{conversation_id}_{catalogue}_{scenario}.json
    <root>/conversation_trials/item_ratings/assistant_ratings.csv
    <root>/conversation_trials/item_ratings/seeker_ratings.csv
    <root>/fashion_profiles/...     # participant style/preference profiles (schema not yet confirmed; not used by this loader)
    <root>/metadata/...             # item catalog details -- schema not yet confirmed; see `_resolve_item_image` below
    <root>/supplements/...          # not used by this loader
    <root>/surveys/...              # not used by this loader

Transcript file schema (one file per (conversation_id, catalogue, scenario)):

    {
      "conversation_id": int,
      "session_code": str,
      "scenario": int,
      "mentioned_items": [int, ...],   # LOCAL item indices (1-12) within this transcript's catalogue
      "gt_items": [int, ...],          # LOCAL item index/indices the seeker ultimately accepted
      "conversation_content": [
        {
          "turn": int,
          "timestamp": "HH:MM:SS",
          "content": {
            "utterances": [str, ...],
            "role": "Assistant" | "Seeker",
            "tags": [[str, ...], ...]   # per-utterance dialogue-act tags, e.g. "IQ", "CQ", "RS", "ACT"
          }
        },
        ...
      ],
      "catalogue": "a" | "b" | "c"
    }

Item indexing: each catalogue ("a", "b", "c") has its own 12-item block.
The `item_ratings/*.csv` files expose all three catalogues' items as a
single flat 36-column row (item_1..item_36) with -1 for items outside the
row's catalogue; the offset per catalogue is:

    catalogue "a" -> columns item_1..item_12   (offset 0)
    catalogue "b" -> columns item_13..item_24  (offset 12)
    catalogue "c" -> columns item_25..item_36  (offset 24)

so a transcript's `gt_items`/`mentioned_items` (local indices 1-12, as
referenced by "ItemNN" in the dialogue text) map onto a specific item's
global column via `CATALOGUE_OFFSET[catalogue] + local_idx`. This loader
represents every item with a stable string id `"{catalogue}_item_{local_idx}"`
(e.g. "a_item_2") so it never depends on the global column numbering
outside of parsing the ratings CSVs.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data.common import Sample

logger = logging.getLogger(__name__)

CATALOGUES = ["a", "b", "c"]
ITEMS_PER_CATALOGUE = 12
CATALOGUE_OFFSET = {cat: i * ITEMS_PER_CATALOGUE for i, cat in enumerate(CATALOGUES)}

# VOGUE's per-utterance tags (RTI=request task info, IQ=initial query,
# CQ=clarifying question, ANS=answer, RS=recommend set, ACT=accept, etc.)
# are speech-act labels, not emotion labels -- there is no ground-truth
# emotion annotation in this dataset. `emotion_label` is therefore left as
# None for VOGUE samples and the EDM's affect component falls back to its
# self-supervised classifier rather than a supervised target. This is a
# data-availability fact, not a workaround; do not infer emotion labels
# from speech-act tags.


def _global_item_id(catalogue: str, local_idx: int) -> str:
    return f"{catalogue}_item_{local_idx}"


class VOGUELoader:
    def __init__(self, root: str, seed: int = 42):
        self.root = Path(root)
        self.rng = random.Random(seed)

        self.transcripts_dir = self.root / "conversation_trials" / "transcripts"
        self.assistant_ratings_path = self.root / "conversation_trials" / "item_ratings" / "assistant_ratings.csv"
        self.seeker_ratings_path = self.root / "conversation_trials" / "item_ratings" / "seeker_ratings.csv"
        self.metadata_dir = self.root / "metadata"
        self.fashion_profiles_dir = self.root / "fashion_profiles"

        if not self.transcripts_dir.exists():
            raise FileNotFoundError(
                f"Expected transcripts directory not found: {self.transcripts_dir}. "
                f"Check `paths.vogue_root` in configs/config.yaml."
            )
        if not self.seeker_ratings_path.exists():
            raise FileNotFoundError(
                f"Expected seeker_ratings.csv not found at {self.seeker_ratings_path}."
            )

        self.transcript_paths = sorted(self.transcripts_dir.glob("c*.json"))
        if not self.transcript_paths:
            raise FileNotFoundError(f"No transcript files (c*.json) found under {self.transcripts_dir}")

        self.seeker_ratings = pd.read_csv(self.seeker_ratings_path)
        self.assistant_ratings = (
            pd.read_csv(self.assistant_ratings_path) if self.assistant_ratings_path.exists() else None
        )

        # All items that exist per catalogue, used for negative sampling
        # and as the fixed candidate pool for every sample in that catalogue.
        self.all_item_ids = [
            _global_item_id(cat, i) for cat in CATALOGUES for i in range(1, ITEMS_PER_CATALOGUE + 1)
        ]

        # Per-participant interaction history across conversations, built
        # from the seeker ratings table (their own highest-rated item per
        # earlier conversation), used to populate `user_history_ids`.
        self._participant_history = self._build_participant_history()

    # ------------------------------------------------------------------
    # Ratings parsing
    # ------------------------------------------------------------------

    def _build_participant_history(self) -> dict:
        history = defaultdict(list)
        rows = self.seeker_ratings.sort_values("conversation_id")
        for _, row in rows.iterrows():
            catalogue = str(row["catalogue"])
            offset = CATALOGUE_OFFSET.get(catalogue, 0)
            best_local_idx, best_rating = None, -1
            for local_idx in range(1, ITEMS_PER_CATALOGUE + 1):
                col = f"item_{offset + local_idx}"
                if col not in row:
                    continue
                val = row[col]
                if val is not None and val > best_rating:
                    best_rating = val
                    best_local_idx = local_idx
            if best_local_idx is not None and best_rating > 0:
                history[str(row["participant_id"])].append(_global_item_id(catalogue, best_local_idx))
        return dict(history)

    def _rating_vector_for(self, catalogue: str) -> dict:
        """Returns {global_item_id: mean_rating} aggregated across all
        seeker rows for this catalogue, used as a real (non-synthetic)
        quality signal folded into the item feature vectors.
        """
        offset = CATALOGUE_OFFSET.get(catalogue, 0)
        out = {}
        sub = self.seeker_ratings[self.seeker_ratings["catalogue"] == catalogue]
        for local_idx in range(1, ITEMS_PER_CATALOGUE + 1):
            col = f"item_{offset + local_idx}"
            if col not in sub.columns:
                continue
            vals = sub[col]
            vals = vals[vals >= 0]
            out[_global_item_id(catalogue, local_idx)] = float(vals.mean()) if len(vals) else 0.0
        return out

    # ------------------------------------------------------------------
    # Transcript parsing
    # ------------------------------------------------------------------

    def _resolve_item_image(self, catalogue: str, local_idx: int) -> str | None:
        """Best-effort image lookup under metadata/. The exact metadata/
        directory schema has not yet been confirmed against the real
        dataset; this tries a few plausible layouts and falls back to None
        (no image, handled gracefully by data/common.py) rather than
        guessing a path that doesn't exist. Once metadata/'s real
        structure is confirmed, replace this with a direct lookup (e.g.
        against a metadata/items.csv or a metadata/<catalogue>/ directory).
        """
        candidates = [
            self.metadata_dir / catalogue / f"item_{local_idx}.jpg",
            self.metadata_dir / catalogue / f"Item{local_idx:02d}.jpg",
            self.metadata_dir / f"{catalogue}_item_{local_idx}.jpg",
            self.metadata_dir / "images" / catalogue / f"item_{local_idx}.jpg",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _parse_transcript(self, path: Path) -> list:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        catalogue = str(data["catalogue"])
        conversation_id = str(data["conversation_id"])
        gt_items = data.get("gt_items") or []
        if not gt_items:
            logger.warning("Transcript %s has no gt_items; skipping", path.name)
            return []
        target_item_id = _global_item_id(catalogue, gt_items[0])

        candidate_ids = [_global_item_id(catalogue, i) for i in range(1, ITEMS_PER_CATALOGUE + 1)]

        # Scenario context: a generic "scenario_<n>" tag plus, if available,
        # the free-text scenario description sourced from the assistant
        # ratings table's `assistant_interpretation` column for this
        # conversation_id (e.g. "Fall outdoor activity at a farm").
        scenario_tag = f"scenario_{data.get('scenario')}"
        context_tags_base = [scenario_tag]
        if self.assistant_ratings is not None:
            match = self.assistant_ratings[self.assistant_ratings["conversation_id"].astype(str) == conversation_id]
            if len(match) and "assistant_interpretation" in match.columns:
                desc = str(match.iloc[0]["assistant_interpretation"])
                if desc and desc.lower() != "nan":
                    context_tags_base.append(desc)

        turns = data.get("conversation_content", [])

        # VOGUE transcripts don't embed participant_id directly; join on
        # conversation_id against the seeker ratings table to recover it
        # for history lookup.
        participant_id = None
        match = self.seeker_ratings[self.seeker_ratings["conversation_id"].astype(str) == conversation_id]
        if len(match):
            participant_id = str(match.iloc[0]["participant_id"])
        user_history = list(self._participant_history.get(participant_id, [])) if participant_id else []

        samples = []
        accumulated_context = list(context_tags_base)

        for i, turn in enumerate(turns):
            content = turn.get("content", {})
            if content.get("role") != "Seeker":
                continue

            utterance_text = " ".join(content.get("utterances", []))
            tags_flat = [t for group in content.get("tags", []) for t in group]
            accumulated_context = list(dict.fromkeys(accumulated_context + tags_flat))

            gold_response = None
            if i + 1 < len(turns):
                next_content = turns[i + 1].get("content", {})
                if next_content.get("role") == "Assistant":
                    gold_response = " ".join(next_content.get("utterances", []))

            samples.append(
                Sample(
                    sample_id=f"{path.stem}_t{turn.get('turn', i)}",
                    dataset="vogue",
                    query_text=utterance_text,
                    context_tags=accumulated_context,
                    emotion_label=None,  # not annotated in VOGUE; see module docstring
                    image_path=self._resolve_item_image(catalogue, gt_items[0]),
                    video_feature_path=None,
                    target_item_id=target_item_id,
                    candidate_item_ids=candidate_ids,
                    user_history_ids=user_history,
                    gold_response=gold_response,
                )
            )

        return samples

    def load_all_samples(self) -> list:
        samples = []
        for path in self.transcript_paths:
            try:
                samples.extend(self._parse_transcript(path))
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning("Skipping malformed transcript %s (%s)", path.name, e)
        logger.info("VOGUE: parsed %d turn-level samples from %d transcripts",
                     len(samples), len(self.transcript_paths))
        return samples

    def build_item_feature_lookup(self, feature_extractor=None) -> dict:
        """Builds a per-item feature vector. In the absence of confirmed
        per-item text/visual metadata (see metadata/ note above), this
        uses the real mean seeker rating for each item as a leading
        feature dimension, with the remainder filled by a deterministic
        hashed vector so items remain distinguishable. Replace with real
        attribute/image embeddings once metadata/'s schema is confirmed --
        this fallback is clearly a placeholder, not a claim of real visual
        features.
        """
        lookup = {}
        for catalogue in CATALOGUES:
            ratings = self._rating_vector_for(catalogue)
            for item_id, mean_rating in ratings.items():
                rng = np.random.RandomState(abs(hash(item_id)) % (2**32))
                base = rng.normal(size=768).astype(np.float32)
                base[0] = mean_rating  # first dim carries the real rating signal
                lookup[item_id] = base
        return lookup

    def split(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        samples = self.load_all_samples()
        rng = random.Random(seed)
        rng.shuffle(samples)
        n = len(samples)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        return samples[:n_train], samples[n_train:n_train + n_val], samples[n_train + n_val:]
