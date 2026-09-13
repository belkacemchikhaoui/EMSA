"""Ablation study (Table 3 / Fig. 6): trains and evaluates each EMSA
variant defined in `models.emsa.ABLATION_VARIANTS` and reports the
relative HR@k drop versus the full model.

Each variant is trained from scratch with the same recipe as the full
model (see `training/train.py`) -- ablation results are only meaningful if
every variant gets a fair, independent training run rather than reusing
weights from the full model. To keep 7 independent training runs
tractable, ablation training uses `cfg["ablation"]["num_epochs"]` (a
smaller epoch budget than the main run, since ablations only need to show
a directional effect) plus early stopping, mixed precision, and per-epoch
timing so progress is visible rather than a single silent log line per
epoch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.common import collate_variable_candidates
from evaluation.metrics import hit_rate_at_k
from models import ABLATION_VARIANTS, EMSAModel
from training.train import build_datasets
from training.utils import get_device, load_config, save_checkpoint, set_seed
from transformers import BertTokenizer, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def train_variant(variant: str, dataset_name: str, cfg: dict, device: torch.device) -> dict:
    set_seed(cfg["training"]["seed"])
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    train_ds, val_ds, test_ds = build_datasets(dataset_name, cfg, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        collate_fn=collate_variable_candidates, num_workers=cfg["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["eval_batch_size"], shuffle=False,
        collate_fn=collate_variable_candidates, num_workers=cfg["training"]["num_workers"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["training"]["eval_batch_size"], shuffle=False,
        collate_fn=collate_variable_candidates, num_workers=cfg["training"]["num_workers"],
    )

    m = cfg["model"]
    ablation_cfg = cfg.get("ablation", {})
    num_epochs = ablation_cfg.get("num_epochs", cfg["training"]["num_epochs"])
    patience_limit = ablation_cfg.get("early_stopping_patience", cfg["training"]["early_stopping_patience"])

    model = EMSAModel(
        hidden_size=m["hidden_size"], num_heads=m["num_attention_heads"],
        num_layers=m["num_transformer_layers"], text_encoder_name=m["text_encoder"],
        num_emotion_categories=cfg["empathy"]["num_emotion_categories"],
        affect_top_k=cfg["empathy"]["affect_top_k"],
        recommendation_mlp_hidden=cfg["recommendation"]["utility_hidden_size"],
        dropout=m["dropout_rate"], variant=variant,
        freeze_text_encoder=m.get("freeze_text_encoder", False),
        freeze_visual_encoder=m.get("freeze_visual_encoder", False),
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg["training"]["warmup_steps"], total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["mixed_precision"] and device.type == "cuda")

    best_val_loss = float("inf")
    best_state = None
    patience = 0
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"ablation_{variant}_{dataset_name}.pt")

    logger.info("[ablation:%s] starting: %d train batches/epoch, up to %d epochs, patience=%d",
                variant, len(train_loader), num_epochs, patience_limit)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{variant}] epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch in pbar:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
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

        epoch_time = time.time() - epoch_start
        logger.info(
            "[ablation:%s] epoch %d/%d done in %.1fs  train_loss=%.4f  val_loss=%.4f",
            variant, epoch + 1, num_epochs, epoch_time, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                logger.info("[ablation:%s] early stopping at epoch %d/%d", variant, epoch + 1, num_epochs)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint({"model_state_dict": model.state_dict(), "variant": variant}, ckpt_path)

    model.eval()
    all_utility, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(batch)
            all_utility.append(out["utility"].cpu().numpy())
            all_targets.append(batch["target_index"].cpu().numpy())
    utility = np.concatenate(all_utility, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    result = {"checkpoint": ckpt_path}
    for k in cfg["recommendation"]["top_k"]:
        result[f"hr@{k}"] = hit_rate_at_k(utility, targets, k)
    return result


def run_ablation(dataset_name: str, cfg: dict) -> dict:
    device = get_device(cfg.get("device", "cuda"))
    results = {}
    for variant in ABLATION_VARIANTS:
        logger.info("=== Running ablation variant: %s (%s) ===", variant, dataset_name)
        start = time.time()
        results[variant] = train_variant(variant, dataset_name, cfg, device)
        logger.info("=== Finished variant %s in %.1fs ===", variant, time.time() - start)

    full_hr5 = results["full"].get("hr@5")
    for variant, res in results.items():
        if variant == "full" or full_hr5 in (None, 0):
            continue
        res["hr@5_drop_pct"] = 100.0 * (full_hr5 - res.get("hr@5", 0)) / full_hr5

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["output_dir"], f"ablation_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote ablation results to %s", out_path)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["vogue", "pixelrec", "mmshop"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_ablation(args.dataset, cfg)


if __name__ == "__main__":
    main()
