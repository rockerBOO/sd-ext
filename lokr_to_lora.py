#!/usr/bin/env python3
import argparse
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LOKR_SUFFIX_RE = re.compile(r"\.(lokr_w1|lokr_w1_a|lokr_w1_b|lokr_w2|lokr_w2_a|lokr_w2_b|alpha)$")
DEFAULT_RANK = 32


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def group_lokr_keys(state_dict):
    """Group LoKr tensors by base module key, keyed by part name (w1, w1_a, w1_b, w2, w2_a, w2_b, alpha)."""
    groups = {}
    passthrough = {}
    for key, value in state_dict.items():
        m = LOKR_SUFFIX_RE.search(key)
        if not m:
            passthrough[key] = value
            continue
        base_key = key[: m.start()]
        part = m.group(1).replace("lokr_", "")
        groups.setdefault(base_key, {})[part] = value
    return groups, passthrough


def resolve_factor(parts, name):
    """Return the full w1/w2 matrix, reconstructing it from _a/_b factors if decomposed."""
    if name in parts:
        return parts[name]
    a, b = parts.get(f"{name}_a"), parts.get(f"{name}_b")
    if a is not None and b is not None:
        return a.float() @ b.float()
    return None


def to_kohya_key(base_key, prefix="lora_unet"):
    """diffusion_model.blocks.0.attn.wq -> lora_unet_blocks_0_attn_wq"""
    base_key = base_key.removeprefix("diffusion_model.")
    return f"{prefix}_{base_key.replace('.', '_')}"


def convert_lokr_to_lora(input_path, output_path, target_rank=None, device=None, kohya_keys=True):
    device = device or get_device()
    print(f"Using device: {device}")

    print(f"Loading {input_path}...")
    state_dict = load_file(input_path)
    groups, new_state_dict = group_lokr_keys(state_dict)

    total_blocks = len(groups)
    print(f"Found {total_blocks} LoKr blocks. Starting conversion...")
    start_time = time.time()
    processed = 0
    converted = 0

    for base_key, parts in groups.items():
        processed += 1
        w1 = resolve_factor(parts, "w1")
        w2 = resolve_factor(parts, "w2")

        if w1 is None or w2 is None:
            print(f"\nSkipping {base_key}: missing w1/w2 (found keys: {sorted(parts.keys())})")
            continue

        out_key = to_kohya_key(base_key) if kohya_keys else base_key

        try:
            w1 = w1.to(device).float()
            w2 = w2.to(device).float()

            # LoKr's effective delta is scale * kron(w1, w2), where scale = alpha / dim
            # and dim is the rank of whichever factor is decomposed into _a/_b (the LoKr
            # training "dim" hyperparameter). If neither factor is decomposed there's no
            # dim to divide by, so the stored alpha doesn't scale anything (scale = 1).
            # kron(w1, w2) itself is always unscaled raw data, regardless of decomposition.
            dim_orig = None
            if "w2_a" in parts:
                dim_orig = parts["w2_a"].shape[1]
            elif "w1_a" in parts:
                dim_orig = parts["w1_a"].shape[1]
            alpha_orig = parts["alpha"].item() if "alpha" in parts else None
            true_scale = (alpha_orig / dim_orig) if (alpha_orig is not None and dim_orig) else 1.0

            weight = torch.kron(w1, w2)

            rank = target_rank if target_rank else min(DEFAULT_RANK, min(weight.shape))
            # kron(w1, w2) has rank <= rank(w1) * rank(w2); an exact full SVD is wasted
            # work for the tall/wide matrices LoKr produces, so use randomized low-rank SVD.
            q = min(rank + 8, *weight.shape)
            try:
                u, s, v = torch.svd_lowrank(weight, q=q, niter=4)
            except Exception:
                weight = weight.cpu()
                u, s, v = torch.svd_lowrank(weight, q=q, niter=4)
                u, s, v = u.to(device), s.to(device), v.to(device)
            vh = v.transpose(-2, -1)

            rank = min(rank, s.shape[0])

            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank, :]

            sqrt_s = torch.diag(torch.sqrt(s))

            # U * sqrt(S) -> lora_up [out_dim, rank]
            # sqrt(S) * Vh -> lora_down [rank, in_dim]
            # Bake the LoKr's true scale into lora_up and emit alpha = rank, so the
            # kohya-side alpha/rank multiplier is 1 and doesn't re-scale an already-scaled
            # matrix by the original (now-meaningless-at-the-new-rank) alpha/dim ratio.
            lora_up = torch.matmul(u, sqrt_s) * true_scale
            lora_down = torch.matmul(sqrt_s, vh)

            new_state_dict[f"{out_key}.lora_down.weight"] = lora_down.cpu().to(dtype=torch.float16)
            new_state_dict[f"{out_key}.lora_up.weight"] = lora_up.cpu().to(dtype=torch.float16)
            new_state_dict[f"{out_key}.alpha"] = torch.tensor([rank], dtype=torch.float16)

            converted += 1
            elapsed = time.time() - start_time
            print(
                f"[{processed}/{total_blocks}] {out_key}: {tuple(weight.shape)} -> rank {rank} "
                f"(scale {true_scale:.4f}, {elapsed:.1f}s elapsed)",
                flush=True,
            )

            del w1, w2, weight, u, s, vh, lora_down, lora_up
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError converting block {out_key}: {e}")
            continue

    save_file(new_state_dict, output_path)
    print(f"\nConverted {converted}/{total_blocks} blocks. Total time: {time.time() - start_time:.2f} seconds.")
    print(f"Saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a LoKr safetensors file into a LoRA safetensors file via SVD.")
    parser.add_argument("input", type=Path, help="Path to the input LoKr safetensors file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Path to the output LoRA safetensors file")
    parser.add_argument(
        "-r", "--rank", type=int, default=None, help=f"Target LoRA rank (default: {DEFAULT_RANK}, capped to the weight's smaller dim)"
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Device to run SVD on (default: auto-detect)"
    )
    parser.add_argument(
        "--raw-keys",
        action="store_true",
        help="Keep dotted module-path keys (e.g. diffusion_model.blocks.0...) instead of "
        "converting to kohya-style lora_unet_... keys expected by musubi-tuner/sd-scripts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_file = args.output or args.input.with_name(f"{args.input.stem}_lora.safetensors")
    device = torch.device(args.device) if args.device else None
    convert_lokr_to_lora(args.input, output_file, args.rank, device, kohya_keys=not args.raw_keys)
