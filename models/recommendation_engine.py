"""Contextual Recommendation Engine (CRE) -- Section 3.3 of the paper.

Utility(i) = alpha * VisualSim(F_v, F_i) + beta * PrefSim(P, P_i) + gamma * ContextSim(H_ctx_t, C_i)   (Eq. 8)
[alpha, beta, gamma] = softmax(W_w H_t + b_w)                                                          (Eq. 9)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicUtilityWeighting(nn.Module):
    """Small NN conditioned on the fused representation H_t that predicts
    the [alpha, beta, gamma] mixture weights (Eq. 9).
    """

    def __init__(self, hidden_size: int = 768, mlp_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 3),
        )

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        logits = self.net(h_t)          # [B, 3]
        return F.softmax(logits, dim=-1)  # [alpha, beta, gamma], sums to 1


class ContextualRecommendationEngine(nn.Module):
    def __init__(self, hidden_size: int = 768, mlp_hidden: int = 128, use_dynamic_weights: bool = True):
        super().__init__()
        self.use_dynamic_weights = use_dynamic_weights
        self.weighting = DynamicUtilityWeighting(hidden_size, mlp_hidden)
        # Fixed-weight fallback for the "w/o Dynamic Weights" ablation.
        self.register_buffer("fixed_weights", torch.tensor([1 / 3, 1 / 3, 1 / 3]))

        # Learned projections so visual/preference/context similarities are
        # computed in a shared, trainable space rather than raw cosine sim
        # on possibly mismatched feature spaces.
        self.visual_proj = nn.Linear(hidden_size, hidden_size)
        self.pref_proj = nn.Linear(hidden_size, hidden_size)
        self.context_proj = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def _batched_cosine_sim(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """query: [B, H], candidates: [B, K, H] -> [B, K]"""
        query = F.normalize(query, dim=-1).unsqueeze(1)   # [B, 1, H]
        candidates = F.normalize(candidates, dim=-1)       # [B, K, H]
        return (query * candidates).sum(dim=-1)

    def forward(
        self,
        h_t: torch.Tensor,                  # [B, H] fused joint representation
        viewed_visual_feat: torch.Tensor,    # [B, H] visual feature of the currently-viewed product
        candidate_visual_feats: torch.Tensor,  # [B, K, H]
        user_pref_vector: torch.Tensor,      # [B, H] historical preference vector P
        candidate_pref_feats: torch.Tensor,  # [B, K, H] item attribute vectors P_i
        context_vector: torch.Tensor,        # [B, H] situational context embedding H_ctx_t
        candidate_context_feats: torch.Tensor,  # [B, K, H] item context tag embeddings C_i
    ) -> dict:
        visual_sim = self._batched_cosine_sim(self.visual_proj(viewed_visual_feat), self.visual_proj(candidate_visual_feats))
        pref_sim = self._batched_cosine_sim(self.pref_proj(user_pref_vector), self.pref_proj(candidate_pref_feats))
        context_sim = self._batched_cosine_sim(self.context_proj(context_vector), self.context_proj(candidate_context_feats))

        if self.use_dynamic_weights:
            weights = self.weighting(h_t)  # [B, 3]
        else:
            weights = self.fixed_weights.unsqueeze(0).expand(h_t.shape[0], -1)

        alpha, beta, gamma = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        utility = alpha * visual_sim + beta * pref_sim + gamma * context_sim  # [B, K]

        return {
            "utility": utility,
            "weights": weights,
            "visual_sim": visual_sim,
            "pref_sim": pref_sim,
            "context_sim": context_sim,
        }

    def rank(self, utility: torch.Tensor, top_k: int) -> torch.Tensor:
        k = min(top_k, utility.shape[-1])
        return torch.topk(utility, k=k, dim=-1).indices
