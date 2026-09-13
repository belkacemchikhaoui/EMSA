"""ReDial-BERT baseline.

Distinguishing property: a strong text-only dialog-based CRS (BERT encoder
+ autoregressive response decoder + a ranking head over dialogue history),
in the spirit of ReDial/INSPIRED-style systems (Sec. 2, Conversational
Recommender Systems). No visual modality of any kind and no empathy
objective; conversation history alone drives both ranking and generation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from baselines.base import BaselineModel


class ReDialBert(BaselineModel):
    name = "ReDial-BERT"

    def __init__(self, hidden_size: int = 768, text_encoder_name: str = "bert-base-uncased",
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, vocab_size: int = 30522):
        super().__init__()
        self.bert = BertModel.from_pretrained(text_encoder_name)
        self.text_proj = (
            nn.Linear(self.bert.config.hidden_size, hidden_size)
            if self.bert.config.hidden_size != hidden_size else nn.Identity()
        )
        self.history_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.history_proj = nn.Linear(hidden_size * 2, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)
        self.rank_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, batch: dict, decoder_input_ids: torch.Tensor = None) -> dict:
        text_out = self.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        text_pooled = self.text_proj(text_out.pooler_output)

        history = batch["user_history_features"]  # [B, T, H] treated as a proxy dialogue/interaction history
        _, hidden = self.history_encoder(history)
        hist_repr = self.history_proj(torch.cat([hidden[0], hidden[1]], dim=-1))

        fused = text_pooled + hist_repr  # simple additive combination of current turn + history

        candidate_repr = self.rank_head(batch["candidate_features"])
        utility = torch.einsum("bh,bkh->bk", fused, candidate_repr)

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
