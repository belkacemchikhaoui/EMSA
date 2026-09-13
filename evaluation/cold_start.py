"""Cold-start analysis (Fig. 8): HR@5 as a function of the number of
historical interactions available for each user, computed by truncating
each test sample's `user_history_features` to the first N entries.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

from evaluation.metrics import hit_rate_at_k
from training.utils import get_device, load_checkpoint

logger = logging.getLogger(__name__)


def _truncate_history(batch: dict, n: int) -> dict:
    batch = dict(batch)
    hist = batch["user_history_features"]
    if hist.shape[1] > n:
        batch["user_history_features"] = hist[:, :n, :]
    elif hist.shape[1] < n:
        pad = torch.zeros(hist.shape[0], n - hist.shape[1], hist.shape[2], dtype=hist.dtype, device=hist.device)
        batch["user_history_features"] = torch.cat([hist, pad], dim=1)
    return batch


@torch.no_grad()
def run_cold_start_analysis(model_name: str, dataset_name: str, cfg: dict, model_builder, test_loader) -> dict:
    device = get_device(cfg.get("device", "cuda"))
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"{model_name}_{dataset_name}.pt")
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    model = model_builder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = {}
    for n_interactions in cfg["cold_start"]["interaction_counts"]:
        all_utility, all_targets = [], []
        for batch in test_loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch = _truncate_history(batch, n_interactions)
            out = model(batch)
            all_utility.append(out["utility"].cpu().numpy())
            all_targets.append(batch["target_index"].cpu().numpy())
        utility = np.concatenate(all_utility, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        hr5 = hit_rate_at_k(utility, targets, 5)
        results[str(n_interactions)] = hr5
        logger.info("[%s/%s] n_interactions=%d -> HR@5=%.4f", model_name, dataset_name, n_interactions, hr5)

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["output_dir"], f"cold_start_{model_name}_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
