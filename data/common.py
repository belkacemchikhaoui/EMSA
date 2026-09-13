"""Shared dataset utilities: tokenization, image transforms, vocab handling.

All dataset-specific quirks are isolated to the individual *_loader.py
modules so the modeling code never needs to know which dataset it is
training on -- every loader emits samples in the same canonical schema
defined by `Sample` below.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer

IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# Emotion categories used consistently across datasets (Sec. 3.2, E_affect).
EMOTION_CATEGORIES = [
    "neutral",
    "uncertain",
    "satisfied",
    "frustrated",
    "excited",
    "disappointed",
]


def default_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ]
    )


def load_image(path: str, transform: Optional[transforms.Compose] = None) -> torch.Tensor:
    """Load an image from disk; returns a zero tensor (with a warning flag)
    if the file is missing or corrupt, so a single bad file never crashes
    a training run.
    """
    transform = transform or default_image_transform()
    try:
        img = Image.open(path).convert("RGB")
        return transform(img), True
    except (FileNotFoundError, OSError):
        return torch.zeros(3, IMG_SIZE, IMG_SIZE), False


@dataclasses.dataclass
class Sample:
    """Canonical training sample shared across VOGUE / PixelRec / MM-SHOP.

    Not every dataset populates every field (e.g. PixelRec has no dialogue
    turns; VOGUE has no pre-extracted video features). Absent fields are
    left as None / empty and downstream code checks for that explicitly
    rather than silently substituting synthetic values.
    """

    sample_id: str
    dataset: str                      # "vogue" | "pixelrec" | "mmshop"
    query_text: str                   # user utterance / query
    context_tags: list                # e.g. ["office", "formal"]
    emotion_label: Optional[str]      # ground-truth emotion, if annotated
    image_path: Optional[str]         # path to a raw image, if available
    video_feature_path: Optional[str] # path to pre-extracted region features (.npz)
    target_item_id: str
    candidate_item_ids: list          # positives + sampled negatives
    user_history_ids: list            # historical item ids for this user
    gold_response: Optional[str]      # reference response text, if available


class EMSADataset(Dataset):
    """Wraps a list[Sample] and lazily materializes tensors."""

    def __init__(
        self,
        samples: list,
        tokenizer: BertTokenizer,
        item_feature_lookup: dict,
        max_text_length: int = 64,
        image_transform: Optional[transforms.Compose] = None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.item_feature_lookup = item_feature_lookup
        self.max_text_length = max_text_length
        self.image_transform = image_transform or default_image_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_text(self, text: str) -> dict:
        enc = self.tokenizer(
            text or "",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def _load_visual(self, sample: Sample) -> torch.Tensor:
        if sample.video_feature_path is not None:
            try:
                feats = np.load(sample.video_feature_path)["features"]
                return torch.from_numpy(feats).float()
            except (FileNotFoundError, OSError, KeyError):
                pass
        if sample.image_path is not None:
            img_tensor, ok = load_image(sample.image_path, self.image_transform)
            if ok:
                # A raw image is encoded on-the-fly by the visual backbone;
                # here we just return the pixel tensor with a leading
                # "one region" dimension so downstream code has a uniform
                # [num_regions, ...] contract before the CNN runs.
                return img_tensor.unsqueeze(0)
        # No usable visual input: caller is responsible for masking this
        # out in the fusion encoder rather than treating it as a real zero.
        return torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        text_enc = self._encode_text(s.query_text)
        visual = self._load_visual(s)

        candidate_feats = []
        for cid in s.candidate_item_ids:
            candidate_feats.append(self.item_feature_lookup.get(cid, np.zeros(768, dtype=np.float32)))
        candidate_feats = torch.tensor(np.stack(candidate_feats), dtype=torch.float32)

        target_idx = (
            s.candidate_item_ids.index(s.target_item_id)
            if s.target_item_id in s.candidate_item_ids
            else 0
        )

        user_hist_feats = [
            self.item_feature_lookup.get(hid, np.zeros(768, dtype=np.float32))
            for hid in s.user_history_ids
        ] or [np.zeros(768, dtype=np.float32)]
        user_hist_feats = torch.tensor(np.stack(user_hist_feats), dtype=torch.float32)

        emotion_idx = (
            EMOTION_CATEGORIES.index(s.emotion_label)
            if s.emotion_label in EMOTION_CATEGORIES
            else EMOTION_CATEGORIES.index("neutral")
        )

        return {
            "sample_id": s.sample_id,
            "input_ids": text_enc["input_ids"],
            "attention_mask": text_enc["attention_mask"],
            "visual_regions": visual,
            "candidate_features": candidate_feats,
            "target_index": torch.tensor(target_idx, dtype=torch.long),
            "user_history_features": user_hist_feats,
            "context_tags": s.context_tags,
            "emotion_label": torch.tensor(emotion_idx, dtype=torch.long),
            "gold_response": s.gold_response or "",
        }


def collate_variable_candidates(batch: list) -> dict:
    """Pads the candidate/user-history dimension (which varies per sample)
    to the max length in the batch instead of assuming a fixed size.
    """
    max_cand = max(b["candidate_features"].shape[0] for b in batch)
    max_hist = max(b["user_history_features"].shape[0] for b in batch)
    max_regions = max(b["visual_regions"].shape[0] for b in batch)

    def pad(t: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_len = target_len - t.shape[0]
        if pad_len <= 0:
            return t
        pad_shape = (pad_len,) + tuple(t.shape[1:])
        return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)

    out = {
        "sample_id": [b["sample_id"] for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "visual_regions": torch.stack([pad(b["visual_regions"], max_regions) for b in batch]),
        "candidate_features": torch.stack([pad(b["candidate_features"], max_cand) for b in batch]),
        "target_index": torch.stack([b["target_index"] for b in batch]),
        "user_history_features": torch.stack([pad(b["user_history_features"], max_hist) for b in batch]),
        "context_tags": [b["context_tags"] for b in batch],
        "emotion_label": torch.stack([b["emotion_label"] for b in batch]),
        "gold_response": [b["gold_response"] for b in batch],
    }
    return out
