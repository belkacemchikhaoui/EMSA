"""Supervised training entry point for the recommendation objective.

Trains EMSA (or a chosen baseline) on a chosen dataset using a standard
ranking loss (cross-entropy over the candidate set, with the true item as
the positive class), plus the affect-classification auxiliary loss where a
ground-truth emotion label is available. This is run once per (model,
dataset) pair; `evaluation/run_all.py` then loads every resulting
checkpoint to produce the comparison tables/figures.

The empathy-aware dialogue *policy* (response generation + PPO) is trained
separately by `training/ppo_trainer.py`, since it optimizes a different
objective (Eq. 7) over generated sequences rather than a ranking loss.
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from baselines import build_baseline
from data import build_loader
from data.common import EMSADataset, collate_variable_candidates
from models import EMSAModel
from training.utils import get_device, load_config, save_checkpoint, set_seed

logger = logging.getLogger(__name__)


def build_model(model_name: str, cfg: dict, device: torch.device):
    m = cfg["model"]
    if model_name == "emsa":
        model = EMSAModel(
            hidden_size=m["hidden_size"],
            num_heads=m["num_attention_heads"],
            num_layers=m["num_transformer_layers"],
            text_encoder_name=m["text_encoder"],
            num_emotion_categories=cfg["empathy"]["num_emotion_categories"],
            affect_top_k=cfg["empathy"]["affect_top_k"],
            recommendation_mlp_hidden=cfg["recommendation"]["utility_hidden_size"],
            dropout=m["dropout_rate"],
            variant="full",
            freeze_text_encoder=m.get("freeze_text_encoder", False),
            freeze_visual_encoder=m.get("freeze_visual_encoder", False),
        )
    else:
        model = build_baseline(
            model_name,
            hidden_size=m["hidden_size"],
            text_encoder_name=m["text_encoder"],
            dropout=m["dropout_rate"],
            num_emotion_categories=cfg["empathy"]["num_emotion_categories"],
        )
    return model.to(device)


def build_datasets(dataset_name: str, cfg: dict, tokenizer: BertTokenizer):
    loader_kwargs = {"seed": cfg["training"]["seed"]}
    split_kwargs = {"seed": cfg["training"]["seed"]}
    if dataset_name == "pixelrec":
        pr_cfg = cfg.get("pixelrec", {})
        loader_kwargs["num_negatives"] = pr_cfg.get("num_negatives", 99)
        loader_kwargs["min_user_interactions"] = pr_cfg.get("min_user_interactions", 3)
        split_kwargs["max_users"] = pr_cfg.get("max_users")

    loader = build_loader(dataset_name, cfg["paths"][f"{dataset_name}_root"], **loader_kwargs)
    train_samples, val_samples, test_samples = loader.split(**split_kwargs)
    item_feature_lookup = loader.build_item_feature_lookup()

    max_len = cfg["model"]["max_text_length"]
    train_ds = EMSADataset(train_samples, tokenizer, item_feature_lookup, max_len)
    val_ds = EMSADataset(val_samples, tokenizer, item_feature_lookup, max_len)
    test_ds = EMSADataset(test_samples, tokenizer, item_feature_lookup, max_len)
    return train_ds, val_ds, test_ds


def train_one_model(model_name: str, dataset_name: str, cfg: dict) -> str:
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg.get("device", "cuda"))

    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    train_ds, val_ds, test_ds = build_datasets(dataset_name, cfg, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["training"]["num_workers"], collate_fn=collate_variable_candidates,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["eval_batch_size"], shuffle=False,
        num_workers=cfg["training"]["num_workers"], collate_fn=collate_variable_candidates,
    )

    model = build_model(model_name, cfg, device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg["training"]["warmup_steps"], total_steps)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience = 0
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"{model_name}_{dataset_name}.pt")

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["mixed_precision"] and device.type == "cuda")

    for epoch in range(cfg["training"]["num_epochs"]):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_name}/{dataset_name}] epoch {epoch + 1}/{cfg['training']['num_epochs']}", leave=False)
        for batch in pbar:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                if model_name == "emsa":
                    out = model(batch)
                else:
                    out = model(batch)
                loss = criterion(out["utility"], batch["target_index"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                out = model(batch)
                val_loss += criterion(out["utility"], batch["target_index"]).item()
        val_loss /= max(1, len(val_loader))

        logger.info("[%s/%s] epoch %d/%d done in %.1fs  train_loss=%.4f val_loss=%.4f",
                     model_name, dataset_name, epoch + 1, cfg["training"]["num_epochs"],
                     time.time() - epoch_start, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            save_checkpoint(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "config": cfg},
                ckpt_path,
            )
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["vogue", "pixelrec", "mmshop"])
    parser.add_argument("--model", default="emsa")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = train_one_model(args.model, args.dataset, cfg)
    logger.info("Saved best checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    main()
