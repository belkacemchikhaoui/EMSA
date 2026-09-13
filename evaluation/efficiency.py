"""Computational efficiency comparison (Table 4): inference latency,
peak memory, FLOPs, and an energy proxy, measured identically for every
model on the same hardware and batch so the comparison is apples-to-apples.
"""
from __future__ import annotations

import time

import torch


def _try_thop_profile(model: torch.nn.Module, batch: dict):
    """Uses `thop` if installed to report FLOPs; otherwise returns None so
    the caller can report "n/a" rather than a fabricated number.
    """
    try:
        from thop import profile
    except ImportError:
        return None
    try:
        macs, _ = profile(model, inputs=(batch,), verbose=False)
        return macs * 2  # FLOPs ~= 2 * MACs
    except Exception:
        return None


@torch.no_grad()
def measure_efficiency(model: torch.nn.Module, sample_batch: dict, device: torch.device,
                        num_warmup: int = 5, num_trials: int = 30) -> dict:
    model.eval().to(device)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample_batch.items()}

    for _ in range(num_warmup):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(num_trials):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / num_trials * 1000

    peak_mem_gb = None
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    flops = _try_thop_profile(model, batch)

    # A rough energy proxy: NVIDIA GPUs expose instantaneous power draw via
    # NVML if `pynvml` is installed; without it we report None rather than
    # inventing a number, per the "no fabricated results" policy for this
    # repository.
    energy_j = None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        energy_j = power_w * (elapsed_ms / 1000.0)
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return {
        "inference_ms": elapsed_ms,
        "peak_memory_gb": peak_mem_gb,
        "flops_g": (flops / 1e9) if flops is not None else None,
        "energy_j": energy_j,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
