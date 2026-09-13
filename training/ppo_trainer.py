"""Proximal Policy Optimization trainer for the Empathic Dialogue Manager's
policy network, per Sec. 3.2 ("The policy is optimized via Proximal Policy
Optimization (PPO) [45] to maximize expected cumulative reward").

Each rollout is one dialogue turn: the policy generates a response given
the current fused state, we score it with the empathy scorer (Eq. 2-6),
combine that with task-success and coherence rewards (Eq. 7), and take a
clipped PPO policy-gradient step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.dialogue_manager import RewardWeights, compute_reward

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    lambda_empathy: float = 0.3
    lambda_coherence: float = 0.2
    turn_penalty: float = 0.1


def compute_response_perplexity(token_logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Token-level perplexity of the generated response against itself
    (teacher-forced), normalized to a bounded range via a log-sigmoid so it
    behaves well inside Eq. (6)/(7) rather than diverging for long
    sequences.
    """
    mask = (target_ids != pad_id).float()
    logp = F.log_softmax(token_logits, dim=-1)
    token_logp = torch.gather(logp, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    seq_logp = (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    ppl = torch.exp(-seq_logp)
    # Normalize into (0, ~few] range as required by Eq. (6)'s PPL^-1 term.
    return torch.clamp(ppl / ppl.detach().mean().clamp(min=1e-6), min=1e-3, max=50.0)


class PPOTrainer:
    def __init__(self, model, empathy_scorer, optimizer, config: PPOConfig, device: torch.device, pad_id: int):
        self.model = model
        self.empathy_scorer = empathy_scorer
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.pad_id = pad_id
        self.reward_weights = RewardWeights(config.lambda_empathy, config.lambda_coherence)

    def rollout_and_update(self, batch: dict, empathy_prev: torch.Tensor) -> dict:
        """Single-turn rollout (dialogues here are treated as bandit-style
        one-step episodes per turn, consistent with the reward being
        defined turn-locally in Eq. 7; multi-turn credit assignment across
        a full dialogue is handled by accumulating `empathy_prev` across
        calls from the calling training loop).
        """
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        with torch.no_grad():
            enc = self.model.encode(batch["input_ids"], batch["attention_mask"], batch["visual_regions"])
            h_t = enc["joint_repr"]
            old_policy_out = self.model.dialogue_policy(h_t)
            old_act_logits = old_policy_out["act_logits"]
            old_value = old_policy_out["value"]
            old_act_dist = torch.distributions.Categorical(logits=old_act_logits)
            action = old_act_dist.sample()
            old_log_prob = old_act_dist.log_prob(action)

            generated = self.model.dialogue_policy.generate(
                h_t, bos_id=101, eos_id=102, max_len=self.model.dialogue_policy.max_response_length
            )

        gen_out = self.model.dialogue_policy(h_t, generated)
        response_ppl = compute_response_perplexity(gen_out["token_logits"], generated, self.pad_id)

        response_embedding = gen_out["token_logits"].softmax(-1).mean(dim=1) @ self.model.dialogue_policy.token_embedding.weight
        context_embedding = h_t  # situational context approximated by the fused state
        user_utt_embedding = enc["text_feats"].mean(dim=1)
        # BERTScore against a gold response is computed by the calling
        # evaluation harness offline (it requires an external scorer); at
        # train time we approximate appropriateness via perplexity alone
        # for the reward signal, matching common practice of using cheap
        # proxies during RL rollouts and full metrics only at eval time.
        bertscore_proxy = torch.sigmoid(-response_ppl.detach() + 1.0)

        empathy_out = self.empathy_scorer(
            response_embedding, context_embedding, user_utt_embedding,
            response_ppl.detach(), bertscore_proxy, target_emotion=batch.get("emotion_label"),
        )
        empathy_t = empathy_out["empathy_score"]

        with torch.no_grad():
            utility = self.model.recommend(
                h_t, enc["visual_feats"].mean(dim=1), batch["candidate_features"],
                batch["user_history_features"], h_t,
            )["utility"]
            predicted_item = utility.argmax(dim=-1)
            task_success = (predicted_item == batch["target_index"]).float()

        reward_out = compute_reward(
            task_success, self.config.turn_penalty, empathy_t, empathy_prev, response_ppl.detach(), self.reward_weights,
        )
        reward = reward_out["reward"].detach()
        advantage = (reward - old_value.detach())
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            policy_out = self.model.dialogue_policy(h_t)
            new_dist = torch.distributions.Categorical(logits=policy_out["act_logits"])
            new_log_prob = new_dist.log_prob(action)
            entropy = new_dist.entropy().mean()

            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(policy_out["value"], reward)

            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            metrics = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "mean_reward": reward.mean().item(),
                "mean_empathy": empathy_t.mean().item(),
                "task_success_rate": task_success.mean().item(),
            }

        return {"metrics": metrics, "empathy_t": empathy_t.detach()}
