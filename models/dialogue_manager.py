"""Empathic Dialogue Manager (EDM) -- Section 3.2 of the paper.

Implements:
  - the three-component empathy score E_t = 1/3 (E_context + E_affect + E_appropriateness)   (Eq. 2/3)
  - E_context via cosine similarity between response and context embeddings   (Eq. 4)
  - E_affect via a learned emotion classifier over the user utterance        (Eq. 5)
  - E_appropriateness via normalized perplexity + BERTScore                  (Eq. 6)
  - the empathy-aware PPO reward  r_t = r_task + lambda_emp * r_empathy + lambda_coh * r_coherence  (Eq. 7)
  - the policy network pi_theta (transformer decoder) producing a dialogue
    act distribution and the response token sequence.

Dialogue acts (a_d_t) follow standard CRS action spaces: ask_clarification,
provide_recommendation, express_empathy, confirm, chitchat.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DIALOGUE_ACTS = [
    "ask_clarification",
    "provide_recommendation",
    "express_empathy",
    "confirm",
    "chitchat",
]


class EmpathyScorer(nn.Module):
    """Computes E_context, E_affect, E_appropriateness and their mean E_t.

    This module is intentionally free of any dataset-specific shortcuts:
    all three components are computed from model-internal representations
    or externally supplied text/embeddings, never from a lookup table.
    """

    def __init__(self, hidden_size: int = 768, num_emotion_categories: int = 6, affect_top_k: int = 2):
        super().__init__()
        self.affect_top_k = affect_top_k
        self.num_emotion_categories = num_emotion_categories
        # W_a in Eq. (5): projects the user-utterance embedding to emotion logits.
        self.affect_classifier = nn.Linear(hidden_size, num_emotion_categories)

    def context_empathy(self, response_embedding: torch.Tensor, context_embedding: torch.Tensor) -> torch.Tensor:
        """Eq. (4): E_context_t = sigmoid(cos_sim(h_resp, h_context))"""
        cos_sim = F.cosine_similarity(response_embedding, context_embedding, dim=-1)
        return torch.sigmoid(cos_sim)

    def affect_empathy(self, user_utterance_embedding: torch.Tensor, target_emotion: torch.Tensor = None):
        """Eq. (5): fraction of the top-k predicted emotion categories that
        match the (soft) target distribution. When `target_emotion` (ground
        truth label indices) is supplied we use it directly as the
        indicator set; otherwise we return the classifier's confidence mass
        on its own top-k as a self-consistency proxy, and separately return
        the logits so a supervised loss can be applied during training.
        """
        logits = self.affect_classifier(user_utterance_embedding)  # [B, C]
        probs = F.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=min(self.affect_top_k, probs.shape[-1]), dim=-1).indices  # [B, k]

        if target_emotion is not None:
            hit = (topk == target_emotion.unsqueeze(-1)).any(dim=-1).float()
            score = hit  # indicator 1[a in top-k], Eq. (5) with |A| effectively 1 (the true label)
        else:
            score = torch.gather(probs, 1, topk).sum(dim=-1) / self.affect_top_k

        return score, logits

    def appropriateness_empathy(self, response_perplexity: torch.Tensor, bertscore: torch.Tensor) -> torch.Tensor:
        """Eq. (6): E_appropriateness = 1/2 (PPL^-1 + BERTScore)
        `response_perplexity` should already be normalized to (0, inf); we
        invert it so lower perplexity -> higher score, then clip to [0, 1].
        """
        inv_ppl = torch.clamp(1.0 / torch.clamp(response_perplexity, min=1e-6), max=1.0)
        bertscore = torch.clamp(bertscore, min=0.0, max=1.0)
        return 0.5 * (inv_ppl + bertscore)

    def forward(
        self,
        response_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
        user_utterance_embedding: torch.Tensor,
        response_perplexity: torch.Tensor,
        bertscore: torch.Tensor,
        target_emotion: torch.Tensor = None,
    ):
        e_context = self.context_empathy(response_embedding, context_embedding)
        e_affect, affect_logits = self.affect_empathy(user_utterance_embedding, target_emotion)
        e_appropriateness = self.appropriateness_empathy(response_perplexity, bertscore)
        e_t = (e_context + e_affect + e_appropriateness) / 3.0
        return {
            "empathy_score": e_t,
            "e_context": e_context,
            "e_affect": e_affect,
            "e_appropriateness": e_appropriateness,
            "affect_logits": affect_logits,
        }


class DialoguePolicyNetwork(nn.Module):
    """pi_theta: transformer-decoder policy over (dialogue_act, response_tokens),
    conditioned on the fused multimodal state s_t = H_t plus dialogue history.
    Also emits a scalar state-value estimate for PPO's advantage computation.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_response_length: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_response_length = max_response_length

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_response_length, hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

        self.act_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, len(DIALOGUE_ACTS)),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, state: torch.Tensor, decoder_input_ids: torch.Tensor = None):
        """state: [B, H] fused multimodal + dialogue-history representation."""
        memory = state.unsqueeze(1)  # [B, 1, H]

        act_logits = self.act_head(state)          # [B, num_acts]
        value = self.value_head(state).squeeze(-1)  # [B]

        if decoder_input_ids is None:
            return {"act_logits": act_logits, "value": value, "token_logits": None}

        b, t = decoder_input_ids.shape
        positions = torch.arange(t, device=decoder_input_ids.device).unsqueeze(0).expand(b, t)
        tok_emb = self.token_embedding(decoder_input_ids) + self.pos_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(t).to(decoder_input_ids.device)

        decoded = self.decoder(tgt=tok_emb, memory=memory, tgt_mask=causal_mask)
        token_logits = self.output_head(decoded)  # [B, T, vocab]

        return {"act_logits": act_logits, "value": value, "token_logits": token_logits}

    @torch.no_grad()
    def generate(self, state: torch.Tensor, bos_id: int, eos_id: int, max_len: int = None):
        max_len = max_len or self.max_response_length
        b = state.shape[0]
        device = state.device
        tokens = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            out = self.forward(state, tokens)
            next_token_logits = out["token_logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, eos_id), next_token)
            tokens = torch.cat([tokens, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break
        return tokens


@dataclass
class RewardWeights:
    lambda_empathy: float = 0.3
    lambda_coherence: float = 0.2


def compute_reward(
    task_success: torch.Tensor,       # [B] bool: did user accept recommendation
    turn_penalty: float,
    empathy_t: torch.Tensor,          # [B] current-turn empathy score
    empathy_prev: torch.Tensor,       # [B] previous-turn empathy score
    response_perplexity: torch.Tensor,  # [B] normalized perplexity of generated response
    weights: RewardWeights,
) -> dict:
    """Eq. (7): r_t = r_task + lambda_emp * r_empathy + lambda_coh * r_coherence"""
    r_task = task_success.float() - turn_penalty  # +1 if accept, -0.1 per turn otherwise handled by caller
    r_empathy = empathy_t - empathy_prev
    r_coherence = -torch.log(torch.clamp(response_perplexity, min=1e-6))

    total = r_task + weights.lambda_empathy * r_empathy + weights.lambda_coherence * r_coherence
    return {
        "reward": total,
        "r_task": r_task,
        "r_empathy": r_empathy,
        "r_coherence": r_coherence,
    }
