"""Robustness analysis (Fig. 7): performance under synthetic video
occlusion and simulated ASR (speech-recognition) noise.

Video occlusion is simulated by zeroing out a random fraction of visual
regions. ASR noise is simulated with a simple token-corruption model
(random deletion/substitution) applied to the tokenized query text at the
given error rate, which stands in for a real ASR system's word-error
rate when no paired audio is available.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.common import collate_variable_candidates
from evaluation.metrics import hit_rate_at_k
from training.utils import get_device, load_checkpoint, load_config

logger = logging.getLogger(__name__)


def apply_video_occlusion(visual_regions: torch.Tensor, occlusion_rate: float, generator: torch.Generator) -> torch.Tensor:
    if occlusion_rate <= 0:
        return visual_regions
    mask = torch.rand(visual_regions.shape[:2], generator=generator) >= occlusion_rate
    mask = mask.to(visual_regions.device)
    shape = [mask.shape[0], mask.shape[1]] + [1] * (visual_regions.dim() - 2)
    return visual_regions * mask.view(*shape).float()


def apply_asr_noise(input_ids: torch.Tensor, error_rate: float, pad_id: int, mask_id: int,
                     generator: torch.Generator) -> torch.Tensor:
    if error_rate <= 0:
        return input_ids
    noisy = input_ids.clone()
    corrupt_mask = torch.rand(input_ids.shape, generator=generator) < error_rate
    corrupt_mask &= input_ids != pad_id
    noisy[corrupt_mask] = mask_id
    return noisy


@torch.no_grad()
def evaluate_under_noise(model, test_loader, device, occlusion_rate: float, asr_error_rate: float,
                          top_k: int, pad_id: int, mask_id: int, seed: int = 42) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    all_utility, all_targets = [], []
    for batch in test_loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch["visual_regions"] = apply_video_occlusion(batch["visual_regions"], occlusion_rate, generator)
        batch["input_ids"] = apply_asr_noise(batch["input_ids"], asr_error_rate, pad_id, mask_id, generator)
        out = model(batch)
        all_utility.append(out["utility"].cpu().numpy())
        all_targets.append(batch["target_index"].cpu().numpy())
    utility = np.concatenate(all_utility, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return hit_rate_at_k(utility, targets, top_k)


def run_robustness_analysis(model_name: str, dataset_name: str, cfg: dict, model_builder, test_loader,
                             tokenizer) -> dict:
    device = get_device(cfg.get("device", "cuda"))
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"{model_name}_{dataset_name}.pt")
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    model = model_builder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = {"video_occlusion": {}, "asr_noise": {}}
    top5 = 5
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    for occ in cfg["robustness"]["video_occlusion_levels"]:
        hr = evaluate_under_noise(model, test_loader, device, occ, 0.0, top5, pad_id, mask_id, cfg["training"]["seed"])
        results["video_occlusion"][str(occ)] = hr
        logger.info("[%s/%s] occlusion=%.2f -> HR@5=%.4f", model_name, dataset_name, occ, hr)

    for rate in cfg["robustness"]["asr_error_rates"]:
        hr = evaluate_under_noise(model, test_loader, device, 0.0, rate, top5, pad_id, mask_id, cfg["training"]["seed"])
        results["asr_noise"][str(rate)] = hr
        logger.info("[%s/%s] asr_error=%.2f -> HR@5=%.4f", model_name, dataset_name, rate, hr)

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["output_dir"], f"robustness_{model_name}_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
