"""Evaluation metrics used throughout Section 4 of the paper:
  - HR@k, NDCG@k                             (Table 1, Table 3, Fig. 2/9)
  - BLEU-4, ROUGE-L, distinct-2, perplexity   (Table 2, Fig. 3)
  - Empathy score                            (Table 2, computed via models.dialogue_manager.EmpathyScorer)
  - Paired significance tests                (Table 5)

All functions operate on plain numpy/python objects so they can be reused
identically by the training harness, the ablation study, the robustness
analysis, and the cold-start analysis -- there is exactly one
implementation of each metric in the whole repository.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def hit_rate_at_k(utility_scores: np.ndarray, target_indices: np.ndarray, k: int) -> float:
    """utility_scores: [N, K_candidates], target_indices: [N] index of the
    ground-truth item within each row's candidate list.
    """
    topk = np.argsort(-utility_scores, axis=1)[:, :k]
    hits = (topk == target_indices[:, None]).any(axis=1)
    return float(hits.mean())


def ndcg_at_k(utility_scores: np.ndarray, target_indices: np.ndarray, k: int) -> float:
    topk = np.argsort(-utility_scores, axis=1)[:, :k]
    ndcgs = []
    for row_topk, target in zip(topk, target_indices):
        pos = np.where(row_topk == target)[0]
        if len(pos) == 0:
            ndcgs.append(0.0)
        else:
            rank = pos[0]
            ndcgs.append(1.0 / math.log2(rank + 2))
    return float(np.mean(ndcgs))


def per_sample_hit_at_k(utility_scores: np.ndarray, target_indices: np.ndarray, k: int) -> np.ndarray:
    """Returns a per-sample binary hit vector, needed for paired significance tests."""
    topk = np.argsort(-utility_scores, axis=1)[:, :k]
    return (topk == target_indices[:, None]).any(axis=1).astype(float)


# ---------------------------------------------------------------------------
# Dialogue-quality metrics
# ---------------------------------------------------------------------------

def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_n(candidate: str, reference: str, n: int = 4) -> float:
    """A compact, dependency-light BLEU-n implementation (geometric mean of
    1..n-gram precisions with a brevity penalty). For publication-grade
    BLEU scoring, swap this for `sacrebleu` / `nltk.translate.bleu_score`;
    this implementation is deterministic and requires no external corpus
    downloads, which keeps CI/reproducibility simple.
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    if not cand_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for i in range(1, n + 1):
        cand_ngrams = _ngrams(cand_tokens, i)
        ref_ngrams = _ngrams(ref_tokens, i)
        overlap = sum(min(c, ref_ngrams.get(g, 0)) for g, c in cand_ngrams.items())
        total = max(1, sum(cand_ngrams.values()))
        precisions.append(overlap / total)

    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)

    bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(cand_tokens)))
    return geo_mean * bp


def rouge_l(candidate: str, reference: str) -> float:
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    if not cand_tokens or not ref_tokens:
        return 0.0

    m, n = len(cand_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cand_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    beta = 1.2
    return ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)


def distinct_n(responses: list, n: int = 2) -> float:
    all_ngrams = []
    for r in responses:
        tokens = r.lower().split()
        all_ngrams.extend(_ngrams(tokens, n).keys())
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def corpus_perplexity(neg_log_likelihoods: np.ndarray) -> float:
    """neg_log_likelihoods: per-token average NLL over the corpus."""
    return float(np.exp(np.mean(neg_log_likelihoods)))


# ---------------------------------------------------------------------------
# Significance testing (Table 5)
# ---------------------------------------------------------------------------

def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Paired t-test comparing per-sample metric values for model A
    (typically EMSA) against model B (best baseline), as in Table 5.
    Returns the t-statistic and two-tailed p-value.
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return {"t_statistic": float(t_stat), "p_value": float(p_value)}
