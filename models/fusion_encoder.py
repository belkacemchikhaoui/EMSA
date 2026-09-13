"""Multimodal Fusion Encoder (MFE) -- Section 3.1 of the paper.

Visual encoding: pretrained ResNet-50 -> grid of regional features F_v.
Textual encoding: pretrained BERT -> contextual word embeddings F_u.
Cross-modal attention: multi-head cross-attention where query words attend
to visual regions (Eq. 1), producing H_vu, which is then aggregated by a
self-attention layer into a fixed-length joint representation H_t.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel


class VisualRegionEncoder(nn.Module):
    """Extracts a grid of regional features from a ResNet-50 backbone.

    Input can be either:
      - raw pixels [B, num_regions, 3, H, W] (one "region" = one raw frame,
        which is then internally re-gridded via the conv feature map), or
      - pre-extracted region features [B, num_regions, dv] (e.g. produced
        offline for MM-SHOP's stored video features), in which case this
        module is bypassed by the caller.
    """

    def __init__(self, hidden_size: int = 768, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        # Keep everything up to (and including) the last conv block so we
        # retain a spatial grid of features instead of the pooled vector.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # -> [B, 2048, 7, 7]
        self.proj = nn.Linear(2048, hidden_size)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: [B, num_frames, 3, H, W] -> [B, num_frames*49, hidden]"""
        b, n, c, h, w = pixels.shape
        x = pixels.view(b * n, c, h, w)
        feat_map = self.backbone(x)                       # [B*n, 2048, 7, 7]
        feat_map = feat_map.flatten(2).transpose(1, 2)     # [B*n, 49, 2048]
        feat_map = self.proj(feat_map)                     # [B*n, 49, hidden]
        return feat_map.view(b, n * feat_map.shape[1], -1)  # [B, n*49, hidden]


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # [B, seq_len, hidden]


class CrossAttentionFusion(nn.Module):
    """Implements Eq. (1): Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) V
    with Q from text, K/V from visual regions, followed by a self-attention
    aggregation layer producing a fixed-length H_t.
    """

    def __init__(self, hidden_size: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_size)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden_size)

        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_feats: torch.Tensor,       # [B, M, H] (query words)
        visual_feats: torch.Tensor,     # [B, N, H] (visual regions)
        text_mask: torch.Tensor = None,  # [B, M] 1=keep, 0=pad
        return_attention: bool = False,
    ):
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = text_mask == 0  # True = ignore, applied to *query* side below only for pooling

        # Q = text, K = V = visual -> H_vu: which visual regions matter per word
        h_vu, attn_weights = self.cross_attn(
            query=text_feats, key=visual_feats, value=visual_feats, need_weights=return_attention
        )
        h_vu = self.cross_norm(h_vu + text_feats)

        b = h_vu.shape[0]
        pool_q = self.pool_query.expand(b, -1, -1)
        pooled, _ = self.self_attn(
            query=pool_q,
            key=h_vu,
            value=h_vu,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.self_norm(pooled + pool_q)
        pooled = pooled + self.ffn_norm(self.ffn(pooled))
        h_t = pooled.squeeze(1)  # [B, H]

        if return_attention:
            return h_t, h_vu, attn_weights
        return h_t, h_vu, None


class MultimodalFusionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        text_encoder_name: str = "bert-base-uncased",
        dropout: float = 0.1,
        pretrained_visual: bool = True,
    ):
        super().__init__()
        self.visual_encoder = VisualRegionEncoder(hidden_size, pretrained=pretrained_visual)
        self.text_encoder = TextEncoder(text_encoder_name)
        self.fusion = CrossAttentionFusion(hidden_size, num_heads, dropout)
        self.text_proj_needed = self.text_encoder.bert.config.hidden_size != hidden_size
        if self.text_proj_needed:
            self.text_proj = nn.Linear(self.text_encoder.bert.config.hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_regions: torch.Tensor,
        precomputed_visual: bool = None,
        return_attention: bool = False,
    ):
        text_feats = self.text_encoder(input_ids, attention_mask)
        if self.text_proj_needed:
            text_feats = self.text_proj(text_feats)

        if precomputed_visual is None:
            precomputed_visual = visual_regions.dim() == 3

        if precomputed_visual:
            # visual_regions already [B, N, hidden]
            visual_feats = visual_regions
        else:
            visual_feats = self.visual_encoder(visual_regions)

        h_t, h_vu, attn = self.fusion(
            text_feats, visual_feats, text_mask=attention_mask, return_attention=return_attention
        )
        return {
            "joint_repr": h_t,        # [B, H]  -- used by CRE and EDM
            "text_feats": text_feats,  # [B, M, H]
            "visual_feats": visual_feats,  # [B, N, H]
            "cross_attn_weights": attn,     # [B, M, N] if requested, else None
        }
