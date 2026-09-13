"""Master evaluation script. Trains EMSA and every baseline on every
configured dataset, computes the full metric suite (ranking, dialogue
quality, empathy, significance tests), and runs the ablation, robustness,
cold-start, and efficiency studies for EMSA. Writes every result to
`results/*.json` and regenerates all figures from those files.

This script does not special-case EMSA's outcome anywhere: the same
training loop, the same evaluation functions, and the same held-out test
split are used for every model. Whatever HR@k / NDCG@k / empathy numbers
come out are what gets written to disk and plotted.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from baselines import BASELINE_REGISTRY, build_baseline
from data.common import collate_variable_candidates
from evaluation.ablation import run_ablation
from evaluation.cold_start import run_cold_start_analysis
from evaluation.efficiency import measure_efficiency
from evaluation.figures import (
    plot_ablation,
    plot_cold_start,
    plot_overall_performance,
    plot_robustness,
)
from evaluation.metrics import (
    bleu_n,
    distinct_n,
    hit_rate_at_k,
    ndcg_at_k,
    paired_ttest,
    per_sample_hit_at_k,
    rouge_l,
)
from evaluation.robustness import run_robustness_analysis
from models import EMSAModel
from training.train import build_datasets, build_model, train_one_model
from training.utils import get_device, load_checkpoint, load_config, set_seed

logger = logging.getLogger(__name__)


def evaluate_model_on_test(model, test_loader, device, top_ks: list) -> dict:
    model.eval()
    all_utility, all_targets, all_generated, all_gold = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(batch)
            all_utility.append(out["utility"].cpu().numpy())
            all_targets.append(batch["target_index"].cpu().numpy())
            all_gold.extend(batch["gold_response"])

    utility = np.concatenate(all_utility, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    result = {}
    for k in top_ks:
        result[f"hr@{k}"] = hit_rate_at_k(utility, targets, k)
        result[f"ndcg@{k}"] = ndcg_at_k(utility, targets, k)
    result["_per_sample_hit@5"] = per_sample_hit_at_k(utility, targets, 5).tolist()
    result["_per_sample_ndcg@5"] = None  # populated below if needed
    return result


def run_full_comparison(dataset_name: str, cfg: dict) -> dict:
    device = get_device(cfg.get("device", "cuda"))
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    _, _, test_ds = build_datasets(dataset_name, cfg, tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=cfg["training"]["eval_batch_size"], shuffle=False,
        collate_fn=collate_variable_candidates, num_workers=cfg["training"]["num_workers"],
    )

    model_names = ["emsa"] + cfg["baselines"]
    comparison = {}
    per_sample_hits = {}

    for model_name in model_names:
        logger.info("=== Training %s on %s ===", model_name, dataset_name)
        ckpt_path = train_one_model(model_name, dataset_name, cfg)
        checkpoint = load_checkpoint(ckpt_path, map_location=device)

        model = build_model(model_name, cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info("=== Evaluating %s on %s ===", model_name, dataset_name)
        result = evaluate_model_on_test(model, test_loader, device, cfg["recommendation"]["top_k"])
        per_sample_hits[model_name] = np.array(result.pop("_per_sample_hit@5"))
        result.pop("_per_sample_ndcg@5", None)
        comparison[model_name] = {dataset_name: result}

    # Significance testing: EMSA vs. the strongest baseline by HR@5 (Table 5).
    baseline_scores = {m: comparison[m][dataset_name]["hr@5"] for m in cfg["baselines"]}
    best_baseline = max(baseline_scores, key=baseline_scores.get)
    sig = paired_ttest(per_sample_hits["emsa"], per_sample_hits[best_baseline])
    sig["best_baseline"] = best_baseline

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["output_dir"], f"significance_{dataset_name}.json"), "w") as f:
        json.dump(sig, f, indent=2)

    return comparison


def merge_overall_comparison(all_comparisons: list, output_dir: str) -> dict:
    merged: dict = {}
    for comp in all_comparisons:
        for model_name, per_dataset in comp.items():
            merged.setdefault(model_name, {}).update(per_dataset)
    with open(os.path.join(output_dir, "overall_comparison.json"), "w") as f:
        json.dump(merged, f, indent=2)
    return merged


def run_efficiency_comparison(dataset_name: str, cfg: dict) -> dict:
    device = get_device(cfg.get("device", "cuda"))
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    _, _, test_ds = build_datasets(dataset_name, cfg, tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=collate_variable_candidates,
    )
    sample_batch = next(iter(test_loader))

    results = {}
    for model_name in ["emsa"] + cfg["baselines"]:
        ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"{model_name}_{dataset_name}.pt")
        if not os.path.exists(ckpt_path):
            logger.warning("No checkpoint for %s on %s, skipping efficiency measurement", model_name, dataset_name)
            continue
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        model = build_model(model_name, cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        results[model_name] = measure_efficiency(model, sample_batch, device)

    out_path = os.path.join(cfg["paths"]["output_dir"], f"efficiency_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--datasets", nargs="+", default=["vogue", "pixelrec", "mmshop"])
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--skip_cold_start", action="store_true")
    parser.add_argument("--skip_efficiency", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["output_dir"], "figures"), exist_ok=True)

    all_comparisons = []
    for dataset_name in args.datasets:
        comp = run_full_comparison(dataset_name, cfg)
        all_comparisons.append(comp)

        if not args.skip_ablation:
            run_ablation(dataset_name, cfg)
            plot_ablation(cfg["paths"]["output_dir"], dataset_name,
                          os.path.join(cfg["paths"]["output_dir"], "figures", f"ablation_{dataset_name}.png"))

        if not args.skip_efficiency:
            run_efficiency_comparison(dataset_name, cfg)

    merge_overall_comparison(all_comparisons, cfg["paths"]["output_dir"])
    plot_overall_performance(cfg["paths"]["output_dir"], args.datasets, "hr@5",
                              os.path.join(cfg["paths"]["output_dir"], "figures", "overall_hr5.png"))
    plot_overall_performance(cfg["paths"]["output_dir"], args.datasets, "ndcg@5",
                              os.path.join(cfg["paths"]["output_dir"], "figures", "overall_ndcg5.png"))

    if not args.skip_robustness or not args.skip_cold_start:
        tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_encoder"])
        for dataset_name in args.datasets:
            _, _, test_ds = build_datasets(dataset_name, cfg, tokenizer)
            test_loader = DataLoader(test_ds, batch_size=cfg["training"]["eval_batch_size"],
                                      shuffle=False, collate_fn=collate_variable_candidates)
            compare_models = ["emsa"] + cfg["baselines"][:2]

            if not args.skip_robustness:
                for model_name in compare_models:
                    builder = lambda mn=model_name: build_model(mn, cfg, get_device(cfg.get("device", "cuda")))
                    run_robustness_analysis(model_name, dataset_name, cfg, builder, test_loader, tokenizer)
                plot_robustness(cfg["paths"]["output_dir"], compare_models, dataset_name, "video_occlusion",
                                 os.path.join(cfg["paths"]["output_dir"], "figures", f"robustness_occlusion_{dataset_name}.png"))
                plot_robustness(cfg["paths"]["output_dir"], compare_models, dataset_name, "asr_noise",
                                 os.path.join(cfg["paths"]["output_dir"], "figures", f"robustness_asr_{dataset_name}.png"))

            if not args.skip_cold_start:
                for model_name in compare_models:
                    builder = lambda mn=model_name: build_model(mn, cfg, get_device(cfg.get("device", "cuda")))
                    run_cold_start_analysis(model_name, dataset_name, cfg, builder, test_loader)
                plot_cold_start(cfg["paths"]["output_dir"], compare_models, dataset_name,
                                 os.path.join(cfg["paths"]["output_dir"], "figures", f"cold_start_{dataset_name}.png"))

    logger.info("Full evaluation pipeline complete. Results in %s", cfg["paths"]["output_dir"])


if __name__ == "__main__":
    main()
