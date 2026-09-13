"""UniMIND baseline.

Distinguishing property: a single shared text transformer trained
multi-task for both dialogue generation and recommendation ranking (Sec.
2: "a unified multi-task learning framework for multi-goal conversational
recommender systems"), with task-specific heads branching off a shared
encoder. Still text-only -- no video/image input, no empathy-aware reward.
This is the strongest text-only baseline in the comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from baselines.base import BaselineModel

UNIMIND_TASKS = ["recommend", "generate"]


class UniMIND(BaselineModel):
    name = "UniMIND"

    def __init__(self, hidden_size: int = 768, text_encoder_name: str = "bert-base-uncased",
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, vocab_size: int = 30522):
        super().__init__()
        self.bert = BertModel.from_pretrained(text_encoder_name)
        self.shared_proj = (
            nn.Linear(self.bert.config.hidden_size, hidden_size)
            if self.bert.config.hidden_size != hidden_size else nn.Identity()
        )
        # Task-embedding prefix, in the spirit of prompt-based multi-task
        # formulations used by unified CRS frameworks.
        self.task_embedding = nn.Embedding(len(UNIMIND_TASKS), hidden_size)

        self.rank_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, batch: dict, decoder_input_ids: torch.Tensor = None) -> dict:
        text_out = self.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        shared_repr = self.shared_proj(text_out.pooler_output)

        rec_task = self.task_embedding(torch.zeros(shared_repr.shape[0], dtype=torch.long, device=shared_repr.device))
        rec_repr = shared_repr + rec_task
        candidate_repr = self.rank_head(batch["candidate_features"])
        utility = torch.einsum("bh,bkh->bk", rec_repr, candidate_repr)

        token_logits = None
        if decoder_input_ids is not None:
            gen_task = self.task_embedding(
                torch.ones(shared_repr.shape[0], dtype=torch.long, device=shared_repr.device)
            )
            gen_repr = shared_repr + gen_task
            memory = gen_repr.unsqueeze(1)
            tok_emb = self.token_embedding(decoder_input_ids)
            t = decoder_input_ids.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(t).to(decoder_input_ids.device)
            decoded = self.decoder(tgt=tok_emb, memory=memory, tgt_mask=causal_mask)
            token_logits = self.output_head(decoded)

        return {
            "joint_repr": shared_repr,
            "utility": utility,
            "token_logits": token_logits,
            "act_logits": None,
        }
