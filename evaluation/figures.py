"""Regenerates every figure in the paper (Fig. 2-10) directly from the
JSON/CSV files written by the other `evaluation/*.py` modules. No figure
in this file is ever drawn from a hand-typed number -- if a required
results file is missing, the corresponding figure is skipped with a
logged warning rather than silently substituted with placeholder data.
"""
from __future__ import annotations

import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _load_json(path: str):
    if not os.path.exists(path):
        logger.warning("Results file not found, skipping dependent figure: %s", path)
        return None
    with open(path, "r") as f:
        return json.load(f)


def plot_overall_performance(results_dir: str, datasets: list, metric_key: str, out_path: str):
    """Fig. 2 / Fig. 9 style grouped bar chart across datasets."""
    data = _load_json(os.path.join(results_dir, "overall_comparison.json"))
    if data is None:
        return

    models = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(datasets)
    x = range(len(models))
    for i, ds in enumerate(datasets):
        values = [data[m].get(ds, {}).get(metric_key) for m in models]
        if any(v is None for v in values):
            logger.warning("Missing %s/%s for some models; figure may be incomplete", ds, metric_key)
        positions = [xi + i * width for xi in x]
        ax.bar(positions, [v or 0 for v in values], width=width, label=ds)

    ax.set_xticks([xi + width * (len(datasets) - 1) / 2 for xi in x])
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric_key.upper())
    ax.set_title(f"Model performance comparison ({metric_key.upper()})")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_ablation(results_dir: str, dataset: str, out_path: str):
    """Fig. 6 style ablation bar chart."""
    data = _load_json(os.path.join(results_dir, f"ablation_{dataset}.json"))
    if data is None:
        return
    variants = list(data.keys())
    hr5 = [data[v].get("hr@5", 0) for v in variants]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(variants, hr5)
    ax.set_ylabel("HR@5")
    ax.set_title(f"Ablation study: {dataset}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_robustness(results_dir: str, model_names: list, dataset: str, noise_type: str, out_path: str):
    """Fig. 7 style robustness curve. noise_type in {"video_occlusion", "asr_noise"}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    any_plotted = False
    for model_name in model_names:
        data = _load_json(os.path.join(results_dir, f"robustness_{model_name}_{dataset}.json"))
        if data is None:
            continue
        series = data.get(noise_type, {})
        sorted_keys = sorted(series.keys(), key=float)
        xs = [float(k) for k in sorted_keys]
        ys = [series[k] for k in sorted_keys]
        ax.plot(xs, ys, marker="o", label=model_name)
        any_plotted = True
    if not any_plotted:
        logger.warning("No robustness results found for %s; skipping figure", dataset)
        plt.close(fig)
        return
    ax.set_xlabel(noise_type.replace("_", " ").title())
    ax.set_ylabel("HR@5")
    ax.set_title(f"Robustness under {noise_type.replace('_', ' ')} ({dataset})")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_cold_start(results_dir: str, model_names: list, dataset: str, out_path: str):
    """Fig. 8 style cold-start curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    any_plotted = False
    for model_name in model_names:
        data = _load_json(os.path.join(results_dir, f"cold_start_{model_name}_{dataset}.json"))
        if data is None:
            continue
        xs = sorted(int(k) for k in data.keys())
        ys = [data[str(x)] for x in xs]
        ax.plot(xs, ys, marker="o", label=model_name)
        any_plotted = True
    if not any_plotted:
        logger.warning("No cold-start results found for %s; skipping figure", dataset)
        plt.close(fig)
        return
    ax.set_xlabel("Number of historical interactions")
    ax.set_ylabel("HR@5")
    ax.set_title(f"Cold-start performance ({dataset})")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_training_curves(train_losses: list, val_losses: list, val_hr_history: list, out_path: str):
    """Fig. 10 style training-progress plot, drawn from logged per-epoch
    values that the training loop appends as it runs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(val_hr_history, label="Val HR@10", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HR@10")
    axes[1].set_title("Validation HR@10")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
