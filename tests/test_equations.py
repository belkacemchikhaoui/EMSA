"""Sanity tests that verify the core equations from the paper are
implemented correctly at a unit level (shapes, value ranges, and simple
known-input/known-output checks) -- independent of any dataset.
"""
import torch

from models.dialogue_manager import EmpathyScorer, RewardWeights, compute_reward
from models.recommendation_engine import ContextualRecommendationEngine, DynamicUtilityWeighting


def test_empathy_score_in_unit_interval():
    scorer = EmpathyScorer(hidden_size=16, num_emotion_categories=4, affect_top_k=2)
    b = 3
    out = scorer(
        response_embedding=torch.randn(b, 16),
        context_embedding=torch.randn(b, 16),
        user_utterance_embedding=torch.randn(b, 16),
        response_perplexity=torch.tensor([1.0, 2.0, 5.0]),
        bertscore=torch.tensor([0.5, 0.8, 0.2]),
    )
    assert out["empathy_score"].shape == (b,)
    assert torch.all(out["empathy_score"] >= 0) and torch.all(out["empathy_score"] <= 1.01)


def test_dynamic_utility_weights_sum_to_one():
    weighting = DynamicUtilityWeighting(hidden_size=16, mlp_hidden=8)
    h_t = torch.randn(5, 16)
    weights = weighting(h_t)
    assert weights.shape == (5, 3)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(5), atol=1e-5)


def test_recommendation_engine_utility_shape():
    engine = ContextualRecommendationEngine(hidden_size=16, mlp_hidden=8, use_dynamic_weights=True)
    b, k = 2, 5
    out = engine(
        h_t=torch.randn(b, 16),
        viewed_visual_feat=torch.randn(b, 16),
        candidate_visual_feats=torch.randn(b, k, 16),
        user_pref_vector=torch.randn(b, 16),
        candidate_pref_feats=torch.randn(b, k, 16),
        context_vector=torch.randn(b, 16),
        candidate_context_feats=torch.randn(b, k, 16),
    )
    assert out["utility"].shape == (b, k)


def test_reward_matches_equation_7():
    task_success = torch.tensor([1.0, 0.0])
    empathy_t = torch.tensor([0.6, 0.4])
    empathy_prev = torch.tensor([0.5, 0.5])
    ppl = torch.tensor([1.5, 2.0])
    weights = RewardWeights(lambda_empathy=0.3, lambda_coherence=0.2)

    out = compute_reward(task_success, turn_penalty=0.1, empathy_t=empathy_t,
                          empathy_prev=empathy_prev, response_perplexity=ppl, weights=weights)

    r_task = task_success - 0.1
    r_empathy = empathy_t - empathy_prev
    r_coherence = -torch.log(ppl)
    expected = r_task + 0.3 * r_empathy + 0.2 * r_coherence

    assert torch.allclose(out["reward"], expected, atol=1e-5)


def test_fixed_weights_ablation_uses_uniform_mixture():
    engine = ContextualRecommendationEngine(hidden_size=16, mlp_hidden=8, use_dynamic_weights=False)
    h_t = torch.randn(4, 16)
    out = engine(
        h_t=h_t,
        viewed_visual_feat=torch.randn(4, 16),
        candidate_visual_feats=torch.randn(4, 3, 16),
        user_pref_vector=torch.randn(4, 16),
        candidate_pref_feats=torch.randn(4, 3, 16),
        context_vector=torch.randn(4, 16),
        candidate_context_feats=torch.randn(4, 3, 16),
    )
    assert torch.allclose(out["weights"], torch.full((4, 3), 1 / 3), atol=1e-5)
