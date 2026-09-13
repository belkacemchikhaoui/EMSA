"""MM-Dialog baseline.

Distinguishing property: late fusion. Text and visual encoders run
independently, each producing its own pooled representation; the two are
combined only at the decision level (via a simple gated sum) rather than
through token-level cross-attention. Reflects the paper's "these systems
operate in batch mode" / "cannot handle dynamic situated context"
critique of pre-cross-attention multimodal dialogue systems.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from baselines.base import BaselineModel


class MMDialog(BaselineModel):
    name = "MM-Dialog"

    def __init__(self, hidden_size: int = 768, text_encoder_name: str = "bert-base-uncased",
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, vocab_size: int = 30522):
        super().__init__()
        self.bert = BertModel.from_pretrained(text_encoder_name)
        self.text_proj = (
            nn.Linear(self.bert.config.hidden_size, hidden_size)
            if self.bert.config.hidden_size != hidden_size else nn.Identity()
        )
        self.visual_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

        # Late-fusion gate: a scalar per sample controlling how much the
        # visual branch contributes, learned independently of any
        # token-level attention over visual regions.
        self.gate = nn.Sequential(nn.Linear(hidden_size * 2, 1), nn.Sigmoid())

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
            # crude pooling of raw pixel regions to a single descriptor
            pooled = visual_regions.mean(dim=(2, 3, 4))  # [B, N]
            pooled = pooled.unsqueeze(-1).expand(-1, -1, 768).mean(dim=1)
        else:
            pooled = visual_regions.mean(dim=1)
        return self.visual_proj(pooled)

    def forward(self, batch: dict, decoder_input_ids: torch.Tensor = None) -> dict:
        text_out = self.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        text_pooled = self.text_proj(text_out.pooler_output)

        visual_pooled = self._visual_pool(batch["visual_regions"])

        gate = self.gate(torch.cat([text_pooled, visual_pooled], dim=-1))
        fused = gate * visual_pooled + (1 - gate) * text_pooled  # late (decision-level) fusion

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
