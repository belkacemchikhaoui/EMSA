"""CosRec baseline.

Distinguishing property: non-conversational, visual-similarity-only
recommendation (in the spirit of VBPR/MMGCN-style batch-mode multimodal
recommenders, Sec. 2, "operate in batch mode without interactive
dialogue"). No dialogue policy, no empathy component; ranking is purely
cosine similarity between a learned visual embedding of the queried
product and candidate item embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.base import BaselineModel


class CosRec(BaselineModel):
    name = "CosRec"

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.visual_proj = nn.Sequential(nn.Linear(256, hidden_size), nn.Dropout(dropout))
        self.pretrained_proj = nn.Linear(hidden_size, hidden_size)  # for pre-extracted region features

    def _encode_query_visual(self, visual_regions: torch.Tensor) -> torch.Tensor:
        if visual_regions.dim() == 5:
            b, n, c, h, w = visual_regions.shape
            x = visual_regions.view(b * n, c, h, w)
            feats = self.visual_encoder(x).flatten(1)
            feats = self.visual_proj(feats).view(b, n, -1).mean(dim=1)
        else:
            feats = self.pretrained_proj(visual_regions).mean(dim=1)
        return feats

    def forward(self, batch: dict) -> dict:
        query_visual = self._encode_query_visual(batch["visual_regions"])
        candidate_visual = self.pretrained_proj(batch["candidate_features"])

        utility = F.cosine_similarity(
            query_visual.unsqueeze(1), candidate_visual, dim=-1
        )

        return {
            "joint_repr": query_visual,
            "utility": utility,
            "token_logits": None,
            "act_logits": None,
        }
