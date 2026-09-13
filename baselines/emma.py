"""EMMA baseline (Ghandeharioun et al., "An emotion-aware wellbeing
chatbot").

Distinguishing property: multimodal, but via early fusion (visual and
text features are concatenated up front and jointly encoded, rather than
cross-attended) and emotion-*aware* rather than empathy-*optimized* -- it
consumes a static, pre-extracted emotion label as an auxiliary feature
rather than learning an empathy score via a dedicated reward (Sec. 2:
"use pre-extracted emotion labels rather than inferring empathy from
multimodal signals in real-time"). No dynamic utility weighting, no PPO
training of the dialogue policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from baselines.base import BaselineModel


class EMMA(BaselineModel):
    name = "EMMA"

    def __init__(self, hidden_size: int = 768, text_encoder_name: str = "bert-base-uncased",
                 num_layers: int = 4, num_heads: int = 8, num_emotion_categories: int = 6,
                 dropout: float = 0.1, vocab_size: int = 30522):
        super().__init__()
        self.bert = BertModel.from_pretrained(text_encoder_name)
        self.text_proj = (
            nn.Linear(self.bert.config.hidden_size, hidden_size)
            if self.bert.config.hidden_size != hidden_size else nn.Identity()
        )
        self.visual_proj = nn.Linear(hidden_size, hidden_size)
        self.emotion_embedding = nn.Embedding(num_emotion_categories, hidden_size)

        # Early fusion: concatenate [text; visual; emotion] up front, then
        # a single MLP -- no attention between modalities.
        self.early_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        # Fixed (non-learned, non-conditioned) utility weighting: a single
        # global scalar mix rather than EMSA's per-turn dynamic weights.
        self.static_utility_weight = nn.Parameter(torch.tensor(0.5))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)
        self.rank_head = nn.Linear(hidden_size, hidden_size)

    def _visual_pool(self, visual_regions: torch.Tensor) -> torch.Tensor:
        if visual_regions.dim() == 5:
            pooled = visual_regions.mean(dim=(2, 3, 4)).unsqueeze(-1).expand(-1, -1, 768).mean(dim=1)
        else:
            pooled = visual_regions.mean(dim=1)
        return self.visual_proj(pooled)

    def forward(self, batch: dict, decoder_input_ids: torch.Tensor = None) -> dict:
        text_out = self.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        text_pooled = self.text_proj(text_out.pooler_output)
        visual_pooled = self._visual_pool(batch["visual_regions"])

        # Static emotion label consumed as an auxiliary feature (no
        # empathy-score computation, no reward shaping).
        emotion_vec = self.emotion_embedding(batch["emotion_label"])

        fused = self.early_fusion(torch.cat([text_pooled, visual_pooled, emotion_vec], dim=-1))

        candidate_repr = self.rank_head(batch["candidate_features"])
        w = torch.sigmoid(self.static_utility_weight)
        utility = w * torch.einsum("bh,bkh->bk", fused, candidate_repr) + (1 - w) * torch.einsum(
            "bh,bkh->bk", text_pooled, candidate_repr
        )

        token_logits = None
        if decoder_input_ids is not None:
            memory = fused.unsqueeze(1)
            tok_emb = self.token_embedding(decoder_input_ids)
            t = decoder_input_ids.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(t).to(decoder_input_ids.device)
            decoded = self.decoder(tgt=tok_emb, memory=memory, tgt_mask=causal_mask)
            token_logits = self.output_head(decoded)

        return {
            "joint_repr": fused,
            "utility": utility,
            "token_logits": token_logits,
            "act_logits": None,
        }
