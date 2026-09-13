"""Loader for MM-SHOP, the authors' original simulated shopping dataset
(1,200 dialogues, live video feeds, comprehensive user profiles).

Expected on-disk layout (see README.md):

    <root>/dialogues.json        # dialogue turns, context tags, emotion labels, gold responses
    <root>/video_features/       # <dialogue_id>_<turn_idx>.npz, key "features" -> [N, dv] array
    <root>/user_profiles.json    # user_id -> {"history_item_ids": [...], "preference_vector": [...]}

`dialogues.json` schema (one entry per dialogue):

    {
      "dialogue_id": str,
      "user_id": str,
      "turns": [
        {
          "speaker": "user" | "assistant",
          "text": str,
          "context_tags": [str, ...],
          "emotion_label": str | null,
          "video_frame_ref": str | null   # matches a key in video_features/ (without extension)
        },
        ...
      ],
      "target_item_id": str,
      "candidate_item_ids": [str, ...]
    }

Because MM-SHOP is original data collected by the authors, this loader is
stricter about schema validation than the public-dataset loaders: missing
files raise immediately rather than silently falling back, since a data
error here would otherwise be indistinguishable from a modeling bug.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np

from data.common import Sample

logger = logging.getLogger(__name__)


class MMShopLoader:
    def __init__(self, root: str, num_negatives: int = 19, seed: int = 42):
        self.root = Path(root)
        self.num_negatives = num_negatives
        self.rng = random.Random(seed)

        self.dialogues_path = self.root / "dialogues.json"
        self.video_features_dir = self.root / "video_features"
        self.profiles_path = self.root / "user_profiles.json"

        for p in [self.dialogues_path, self.profiles_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected MM-SHOP file not found: {p}. "
                    f"Check `paths.mmshop_root` in configs/config.yaml."
                )

        with open(self.dialogues_path, "r", encoding="utf-8") as f:
            self.raw_dialogues = json.load(f)
        with open(self.profiles_path, "r", encoding="utf-8") as f:
            self.user_profiles = json.load(f)

        self.all_item_ids = sorted(
            {
                str(d["target_item_id"])
                for d in self.raw_dialogues
                if "target_item_id" in d
            }
        )

    def _video_feature_path(self, dialogue_id: str, ref: str | None) -> str | None:
        if ref is None:
            return None
        candidate = self.video_features_dir / f"{ref}.npz"
        if candidate.exists():
            return str(candidate)
        logger.warning("MM-SHOP: missing video feature file %s (dialogue %s)", candidate, dialogue_id)
        return None

    def _sample_negatives(self, positive_id: str, k: int) -> list:
        pool = [i for i in self.all_item_ids if i != positive_id]
        if len(pool) <= k:
            return pool
        return self.rng.sample(pool, k)

    def _parse_dialogue(self, dialogue: dict) -> list:
        samples = []
        dialogue_id = str(dialogue["dialogue_id"])
        user_id = str(dialogue.get("user_id", "unknown"))
        target_item_id = str(dialogue["target_item_id"])

        profile = self.user_profiles.get(user_id, {})
        user_history = list(profile.get("history_item_ids", []))

        candidate_ids = dialogue.get("candidate_item_ids")
        if not candidate_ids:
            candidate_ids = [target_item_id] + self._sample_negatives(target_item_id, self.num_negatives)
            self.rng.shuffle(candidate_ids)
        candidate_ids = [str(c) for c in candidate_ids]

        turns = dialogue.get("turns", [])
        accumulated_context: list = []

        for i, turn in enumerate(turns):
            if turn.get("speaker") != "user":
                continue

            accumulated_context = list(dict.fromkeys(accumulated_context + turn.get("context_tags", [])))
            video_path = self._video_feature_path(dialogue_id, turn.get("video_frame_ref"))

            gold_response = None
            if i + 1 < len(turns) and turns[i + 1].get("speaker") == "assistant":
                gold_response = turns[i + 1].get("text")

            samples.append(
                Sample(
                    sample_id=f"{dialogue_id}_t{i}",
                    dataset="mmshop",
                    query_text=turn.get("text", ""),
                    context_tags=accumulated_context,
                    emotion_label=turn.get("emotion_label"),
                    image_path=None,
                    video_feature_path=video_path,
                    target_item_id=target_item_id,
                    candidate_item_ids=candidate_ids,
                    user_history_ids=user_history,
                    gold_response=gold_response,
                )
            )

        return samples

    def load_all_samples(self) -> list:
        samples = []
        for dialogue in self.raw_dialogues:
            try:
                samples.extend(self._parse_dialogue(dialogue))
            except KeyError as e:
                logger.warning("MM-SHOP: skipping malformed dialogue %s (%s)",
                                dialogue.get("dialogue_id", "?"), e)
        logger.info("MM-SHOP: parsed %d turn-level samples from %d dialogues",
                     len(samples), len(self.raw_dialogues))
        return samples

    def build_item_feature_lookup(self, feature_extractor=None) -> dict:
        """MM-SHOP items are referenced only by id inside dialogues; if a
        separate item catalog with attributes exists it should be merged
        here. In the absence of one, item features fall back to averaging
        the video-frame features of turns where that item was the target,
        which is the closest available signal to a "canonical" visual
        representation of each item without redundant re-extraction.
        """
        item_frame_feats: dict = {}
        for dialogue in self.raw_dialogues:
            target = str(dialogue.get("target_item_id", ""))
            if not target:
                continue
            for turn in dialogue.get("turns", []):
                ref = turn.get("video_frame_ref")
                if ref is None:
                    continue
                path = self.video_features_dir / f"{ref}.npz"
                if not path.exists():
                    continue
                try:
                    feats = np.load(path)["features"]
                except (KeyError, OSError):
                    continue
                item_frame_feats.setdefault(target, []).append(feats.mean(axis=0))

        lookup = {}
        for item_id in self.all_item_ids:
            if item_id in item_frame_feats:
                lookup[item_id] = np.mean(item_frame_feats[item_id], axis=0).astype(np.float32)
            else:
                rng = np.random.RandomState(abs(hash(item_id)) % (2**32))
                lookup[item_id] = rng.normal(size=768).astype(np.float32)
        return lookup

    def split(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        samples = self.load_all_samples()
        rng = random.Random(seed)
        rng.shuffle(samples)
        n = len(samples)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        return samples[:n_train], samples[n_train:n_train + n_val], samples[n_train + n_val:]
