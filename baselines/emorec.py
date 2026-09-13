"""EmoRec baseline.

Distinguishing property: emotion-aware but text-only. Uses a text emotion
classifier to bias a collaborative-filtering-style ranking score, with no
visual input at all -- representative of the "emotion-aware recommender
systems that embody personality traits" line of work (Sec. 2, Empathic AI)
that predates multimodal empathic grounding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from baselines.base import BaselineModel


class EmoRec(BaselineModel):
    name = "EmoRec"

    def __init__(self, hidden_size: int = 768, text_encoder_name: str = "bert-base-uncased",
                 num_emotion_categories: int = 6, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(text_encoder_name)
        self.text_proj = (
            nn.Linear(self.bert.config.hidden_size, hidden_size)
            if self.bert.config.hidden_size != hidden_size else nn.Identity()
        )
        self.emotion_classifier = nn.Linear(hidden_size, num_emotion_categories)
        self.emotion_bias = nn.Embedding(num_emotion_categories, hidden_size)
        self.rank_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout))

    def forward(self, batch: dict) -> dict:
        text_out = self.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        text_pooled = self.text_proj(text_out.pooler_output)

        emotion_logits = self.emotion_classifier(text_pooled)
        emotion_weights = torch.softmax(emotion_logits, dim=-1)
        emotion_bias_vec = emotion_weights @ self.emotion_bias.weight  # soft mixture of emotion embeddings

        user_repr = text_pooled + emotion_bias_vec  # emotion additively biases the CF-style query vector

        candidate_repr = self.rank_head(batch["candidate_features"])
        utility = torch.einsum("bh,bkh->bk", user_repr, candidate_repr)

        return {
            "joint_repr": user_repr,
            "utility": utility,
            "token_logits": None,
            "act_logits": None,
            "emotion_logits": emotion_logits,
        }
