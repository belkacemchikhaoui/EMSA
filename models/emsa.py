"""Full EMSA model: wires the Multimodal Fusion Encoder, Empathic Dialogue
Manager, and Contextual Recommendation Engine together (Fig. 1).

Also implements the ablation variants used in Table 3 / Figure 6 by
disabling specific sub-components rather than maintaining separate model
classes, so every variant shares exactly the same code path except for the
one component being ablated.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.dialogue_manager import DialoguePolicyNetwork, EmpathyScorer
from models.fusion_encoder import CrossAttentionFusion, MultimodalFusionEncoder, TextEncoder, VisualRegionEncoder
from models.recommendation_engine import ContextualRecommendationEngine

ABLATION_VARIANTS = [
    "full",
    "no_cross_attention",
    "no_empathy_reward",
    "no_dynamic_weights",
    "no_video",
    "no_user_history",
    "single_modal",
]


class ConcatFusion(nn.Module):
    """Simple concatenation+MLP fallback used for the `no_cross_attention`
    ablation, replacing the cross-attention fusion module while keeping
    input/output shapes identical so the rest of the model is unaffected.
    """

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.text_pool = nn.Linear(hidden_size, hidden_size)
        self.visual_pool = nn.Linear(hidden_size, hidden_size)
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, text_feats, visual_feats, text_mask=None, return_attention=False):
        text_pooled = self.text_pool(text_feats.mean(dim=1))
        visual_pooled = self.visual_pool(visual_feats.mean(dim=1))
        h_t = self.combine(torch.cat([text_pooled, visual_pooled], dim=-1))
        return h_t, None, None


class EMSAModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        text_encoder_name: str = "bert-base-uncased",
        vocab_size: int = 30522,
        num_emotion_categories: int = 6,
        affect_top_k: int = 2,
        recommendation_mlp_hidden: int = 128,
        dropout: float = 0.1,
        pretrained_visual: bool = True,
        variant: str = "full",
        freeze_text_encoder: bool = False,
        freeze_visual_encoder: bool = False,
    ):
        super().__init__()
        if variant not in ABLATION_VARIANTS:
            raise ValueError(f"Unknown ablation variant: {variant}. Must be one of {ABLATION_VARIANTS}")
        self.variant = variant

        self.visual_encoder = VisualRegionEncoder(hidden_size, pretrained=pretrained_visual)
        self.text_encoder = TextEncoder(text_encoder_name)
        text_hidden = self.text_encoder.bert.config.hidden_size
        self.text_proj = nn.Linear(text_hidden, hidden_size) if text_hidden != hidden_size else nn.Identity()

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if freeze_visual_encoder:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        if variant == "no_cross_attention":
            self.fusion = ConcatFusion(hidden_size, dropout)
        else:
            self.fusion = CrossAttentionFusion(hidden_size, num_heads, dropout)

        self.dialogue_policy = DialoguePolicyNetwork(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.empathy_scorer = EmpathyScorer(hidden_size, num_emotion_categories, affect_top_k)

        use_dynamic_weights = variant != "no_dynamic_weights"
        self.recommendation_engine = ContextualRecommendationEngine(
            hidden_size, recommendation_mlp_hidden, use_dynamic_weights=use_dynamic_weights
        )

        # Projects raw candidate/history item features (which may come
        # from a non-BERT/ResNet feature space, e.g. pre-extracted 768-d
        # vectors) into the shared hidden space used by the CRE.
        self.item_feature_proj = nn.Linear(hidden_size, hidden_size)

    def encode(self, input_ids, attention_mask, visual_regions, precomputed_visual: bool = None):
        text_feats = self.text_proj(self.text_encoder(input_ids, attention_mask))

        # `visual_regions` is 5D ([B, N, 3, H, W]) for raw pixels (e.g. VOGUE
        # product images loaded on-the-fly) and 3D ([B, N, hidden]) for
        # pre-extracted region features (e.g. MM-SHOP's stored .npz video
        # features). Auto-detect by rank unless the caller overrides it.
        if precomputed_visual is None:
            precomputed_visual = visual_regions.dim() == 3

        if self.variant == "no_video":
            # Visual branch is zeroed out entirely -- the model must rely
            # on text + history alone.
            visual_feats = torch.zeros(
                text_feats.shape[0], 1, text_feats.shape[-1], device=text_feats.device
            )
        elif self.variant == "single_modal":
            # Text-only: visual features are replaced with a constant,
            # uninformative embedding so the fusion module still runs but
            # contributes no signal, matching a "single modality" baseline.
            visual_feats = torch.zeros(
                text_feats.shape[0], 1, text_feats.shape[-1], device=text_feats.device
            )
        elif precomputed_visual:
            visual_feats = visual_regions if visual_regions.shape[-1] == text_feats.shape[-1] else self.item_feature_proj(visual_regions)
        else:
            visual_feats = self.visual_encoder(visual_regions)

        h_t, h_vu, attn = self.fusion(text_feats, visual_feats, text_mask=attention_mask, return_attention=False)
        return {
            "joint_repr": h_t,
            "text_feats": text_feats,
            "visual_feats": visual_feats,
        }

    def recommend(
        self,
        h_t: torch.Tensor,
        viewed_visual_feat: torch.Tensor,
        candidate_features: torch.Tensor,   # [B, K, H]
        user_history_features: torch.Tensor,  # [B, T, H]
        context_vector: torch.Tensor,
    ) -> dict:
        candidate_features = self.item_feature_proj(candidate_features)

        if self.variant == "no_user_history":
            user_pref_vector = torch.zeros_like(h_t)
        else:
            user_pref_vector = self.item_feature_proj(user_history_features).mean(dim=1)

        k = candidate_features.shape[1]
        candidate_context_feats = candidate_features  # tags are not separately encoded here;
        # candidate item embeddings double as their own context representation,
        # consistent with ContextSim comparing item tags C_i to H_ctx_t in the
        # same embedding space as the rest of the utility function.

        out = self.recommendation_engine(
            h_t=h_t,
            viewed_visual_feat=viewed_visual_feat,
            candidate_visual_feats=candidate_features,
            user_pref_vector=user_pref_vector,
            candidate_pref_feats=candidate_features,
            context_vector=context_vector,
            candidate_context_feats=candidate_context_feats,
        )
        return out

    def forward(self, batch: dict, decoder_input_ids: torch.Tensor = None, precomputed_visual: bool = None):
        enc = self.encode(
            batch["input_ids"], batch["attention_mask"], batch["visual_regions"],
            precomputed_visual=precomputed_visual,
        )
        h_t = enc["joint_repr"]

        viewed_visual_feat = enc["visual_feats"].mean(dim=1)
        context_vector = h_t  # situational context is derived from the fused state itself

        rec_out = self.recommend(
            h_t=h_t,
            viewed_visual_feat=viewed_visual_feat,
            candidate_features=batch["candidate_features"],
            user_history_features=batch["user_history_features"],
            context_vector=context_vector,
        )

        policy_out = self.dialogue_policy(h_t, decoder_input_ids)

        return {
            "joint_repr": h_t,
            "utility": rec_out["utility"],
            "utility_weights": rec_out["weights"],
            "act_logits": policy_out["act_logits"],
            "value": policy_out["value"],
            "token_logits": policy_out["token_logits"],
        }
