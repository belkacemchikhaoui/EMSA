"""VisualBERT-CRS baseline.

Distinguishing property (per Related Work, Sec. 2): a single-stream
vision-language transformer (image + text tokens concatenated and jointly
self-attended, as in VisualBERT/ViLBERT-style models) used as the backbone
for a conversational recommender, with a simple linear ranking head and no
empathy-aware objective and no dedicated cross-attention fusion module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from baselines.base import BaselineModel


class VisualBertCRS(BaselineModel):
    name = "VisualBERT-CRS"

    def __init__(self, hidden_size: int = 768, text_encoder_name: str = "bert-base-uncased",
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, vocab_size: int = 30522):
        super().__init__()
        self.text_encoder = TextEncoderWrapper(text_encoder_name, hidden_size)
        self.visual_proj = nn.Linear(2048, hidden_size)

        # Single-stream: text and visual tokens are concatenated and passed
        # through a shared self-attention encoder (no cross-attention Q/K/V
        # split between modalities).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True,
        )
        self.joint_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.rank_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.response_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, batch: dict) -> dict:
        text_feats = self.text_encoder(batch["input_ids"], batch["attention_mask"])  # [B, M, H]

        # visual_regions may be raw pixels [B, N, 3, H, W] pooled to a flat
        # descriptor per region, or pre-extracted [B, N, H] features.
        visual = batch["visual_regions"]
        if visual.dim() == 5:
            b, n = visual.shape[:2]
            visual_flat = visual.mean(dim=(-1, -2))  # crude channel pooling as a lightweight visual token
            visual_flat = visual_flat.mean(dim=-1, keepdim=True).expand(-1, -1, 2048)
            visual_tokens = self.visual_proj(visual_flat)
        else:
            visual_tokens = visual if visual.shape[-1] == text_feats.shape[-1] else self.visual_proj(
                torch.nn.functional.pad(visual, (0, max(0, 2048 - visual.shape[-1])))
            )

        joint = torch.cat([text_feats, visual_tokens], dim=1)
        encoded = self.joint_encoder(joint)
        pooled = encoded.mean(dim=1)  # [B, H]

        candidate_repr = self.rank_head(batch["candidate_features"])  # [B, K, H]
        utility = torch.einsum("bh,bkh->bk", pooled, candidate_repr)

        return {
            "joint_repr": pooled,
            "utility": utility,
            "token_logits": None,
            "act_logits": None,
        }


class TextEncoderWrapper(nn.Module):
    def __init__(self, model_name: str, hidden_size: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.proj = (
            nn.Linear(self.bert.config.hidden_size, hidden_size)
            if self.bert.config.hidden_size != hidden_size
            else nn.Identity()
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.proj(out)
