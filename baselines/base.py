"""Shared interface for all baseline models so the training/evaluation
harness can treat EMSA and every baseline identically.

Each baseline implements `forward(batch) -> dict` with at least:
    - "utility": [B, K] ranking scores over candidate items
    - "token_logits": [B, T, vocab] or None, for dialogue-quality metrics
    - "act_logits": [B, num_acts] or None

Baselines are deliberately simplified relative to their original papers
(full re-implementations of ViLBERT-style pretraining, graph-based CRS
knowledge encoders, etc. are out of scope for a supplementary-material
comparison), but each preserves the architectural property that
distinguishes it in the related-work discussion (e.g. "no cross-attention",
"no live video", "text-only emotion awareness", "static utility weights"),
so the comparison in Table 1-3 is a fair test of *those* specific
properties rather than of unrelated implementation details.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    name: str = "baseline"

    def forward(self, batch: dict) -> dict:
        raise NotImplementedError

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
