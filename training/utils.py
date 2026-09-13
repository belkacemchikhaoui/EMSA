from __future__ import annotations

import os

# Must be set before numpy/torch/scipy are imported anywhere in the process.
# Prevents a duplicate-OpenMP/MKL-runtime crash (seen on Windows as an
# access violation inside libifcoremd.dll, or as a silent hang when it
# happens inside a spawned DataLoader worker) that can occur when multiple
# packages (numpy, scipy, torch) each bundle their own copy of the
# Intel MKL/OpenMP runtime. training.utils is imported by every entry
# point in this repository, so setting this here covers all of them.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# huggingface_hub's newer "Xet" fast-transfer backend has known reliability
# issues as of mid-2026 (404s on xet-read-token, downloads stalling
# indefinitely mid-transfer -- see huggingface_hub GitHub issues #4349,
# #4508, #3266). Disabling it falls back to the older, slower but far more
# reliable plain-HTTP download path for model/tokenizer downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import logging
import random

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "cuda":
        logging.getLogger(__name__).warning("CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None) -> dict:
    return torch.load(path, map_location=map_location)
