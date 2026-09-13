"""Entry point for empathy-aware PPO fine-tuning of EMSA's dialogue policy
(Sec. 3.2, Eq. 7). Run this *after* `training/train.py` has produced a
supervised checkpoint for the recommendation objective -- PPO here only
fine-tunes the dialogue policy head plus lightly updates the shared
encoder, using the empathy score as part of the reward signal.
"""
from __future__ import annotations

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from data.common import collate_variable_candidates
from models import EMSAModel
from models.dialogue_manager import EmpathyScorer
from training.ppo_trainer import PPOConfig, PPOTrainer
from training.train import build_datasets
from training.utils import get_device, load_checkpoint, load_config, save_checkpoint, set_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["vogue", "pixelrec", "mmshop"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--num_ppo_epochs", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg.get("device", "cuda"))

    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    train_ds, _, _ = build_datasets(args.dataset, cfg, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["ppo"]["rollout_batch_size"], shuffle=True,
        collate_fn=collate_variable_candidates, num_workers=cfg["training"]["num_workers"],
    )

    m = cfg["model"]
    model = EMSAModel(
        hidden_size=m["hidden_size"], num_heads=m["num_attention_heads"],
        num_layers=m["num_transformer_layers"], text_encoder_name=m["text_encoder"],
        num_emotion_categories=cfg["empathy"]["num_emotion_categories"],
        affect_top_k=cfg["empathy"]["affect_top_k"],
        recommendation_mlp_hidden=cfg["recommendation"]["utility_hidden_size"],
        dropout=m["dropout_rate"], variant="full",
    ).to(device)

    sup_ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"emsa_{args.dataset}.pt")
    if os.path.exists(sup_ckpt_path):
        checkpoint = load_checkpoint(sup_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Initialized from supervised checkpoint: %s", sup_ckpt_path)
    else:
        logger.warning("No supervised checkpoint found at %s; starting PPO from a randomly "
                        "initialized model. Run training/train.py first for a meaningful result.",
                        sup_ckpt_path)

    empathy_scorer = EmpathyScorer(
        hidden_size=m["hidden_size"],
        num_emotion_categories=cfg["empathy"]["num_emotion_categories"],
        affect_top_k=cfg["empathy"]["affect_top_k"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(empathy_scorer.parameters()), lr=cfg["training"]["learning_rate"],
    )
    ppo_cfg = PPOConfig(
        clip_epsilon=cfg["ppo"]["clip_epsilon"], gamma=cfg["ppo"]["gamma"], gae_lambda=cfg["ppo"]["gae_lambda"],
        value_coef=cfg["ppo"]["value_coef"], entropy_coef=cfg["ppo"]["entropy_coef"],
        ppo_epochs=cfg["ppo"]["ppo_epochs"], lambda_empathy=cfg["ppo"]["lambda_empathy"],
        lambda_coherence=cfg["ppo"]["lambda_coherence"],
    )
    trainer = PPOTrainer(model, empathy_scorer, optimizer, ppo_cfg, device, pad_id=tokenizer.pad_token_id)

    for epoch in range(args.num_ppo_epochs):
        empathy_prev = torch.zeros(cfg["ppo"]["rollout_batch_size"], device=device)
        epoch_metrics = []
        for batch in train_loader:
            if batch["input_ids"].shape[0] != empathy_prev.shape[0]:
                empathy_prev = torch.zeros(batch["input_ids"].shape[0], device=device)
            out = trainer.rollout_and_update(batch, empathy_prev)
            empathy_prev = out["empathy_t"]
            epoch_metrics.append(out["metrics"])

        mean_reward = sum(m_["mean_reward"] for m_ in epoch_metrics) / max(1, len(epoch_metrics))
        mean_empathy = sum(m_["mean_empathy"] for m_ in epoch_metrics) / max(1, len(epoch_metrics))
        logger.info("[PPO %s] epoch %d/%d mean_reward=%.4f mean_empathy=%.4f",
                     args.dataset, epoch + 1, args.num_ppo_epochs, mean_reward, mean_empathy)

    out_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"emsa_ppo_{args.dataset}.pt")
    save_checkpoint(
        {"model_state_dict": model.state_dict(), "empathy_scorer_state_dict": empathy_scorer.state_dict()},
        out_path,
    )
    logger.info("Saved PPO-tuned checkpoint to %s", out_path)


if __name__ == "__main__":
    main()
