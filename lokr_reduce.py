"""
LoKr rank reduction: SVD-truncate full-size lokr_w2 tensors into decomposed
lokr_w2_a/lokr_w2_b pairs, shrinking LyCORIS LoKr-format LoRA files.

Usage:
    python lokr_reduce.py model.safetensors output_dir/
    python lokr_reduce.py model.safetensors output_dir/ --experiments svd --svd-keep-ratio 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lora_reduce import load_with_metadata, save_variant


def group_modules_lokr(tensors: dict) -> dict:
    """Group tensors by base module name, returning w1/w2/alpha keys per module.

    Only recognizes full-format keys (.lokr_w1, .lokr_w2, .alpha). Already-decomposed
    keys (.lokr_w2_a, .lokr_w2_b, .lokr_w1_a, .lokr_w1_b) and tucker keys (.lokr_t2)
    are not recognized here, so modules using them end up without a "w2_key" (or
    "w1_key"), which callers use to skip them as ineligible.
    """
    modules = {}
    for key in tensors:
        if key.endswith(".lokr_w1"):
            base = key[: -len(".lokr_w1")]
            modules.setdefault(base, {})["w1_key"] = key
        elif key.endswith(".lokr_w2"):
            base = key[: -len(".lokr_w2")]
            modules.setdefault(base, {})["w2_key"] = key
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            modules.setdefault(base, {})["alpha_key"] = key
    return modules


def _svd_w2(w2: torch.Tensor, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD of a full lokr_w2 matrix. Returns (U, S, Vh) such that w2 ≈ U @ diag(S) @ Vh."""
    w2_f32 = w2.to(device, dtype=torch.float32)
    return torch.linalg.svd(w2_f32, full_matrices=False)
