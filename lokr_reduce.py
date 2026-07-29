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


def apply_lokr_svd(
    tensors: dict, modules: dict, keep_ratio: float = 0.5, device: str = "cpu"
) -> tuple[dict, dict[str, tuple[int, int]]]:
    """Truncate each module's full lokr_w2 to keep_ratio of its rank via SVD.

    Decomposes lokr_w2 (out x in) into lokr_w2_a (out x r) and lokr_w2_b (r x in),
    removing the lokr_w2 key. Sets the module's alpha to float(r): once lokr_w2 is
    decomposed, ComfyUI's LoKr loader computes alpha_effective = stored_alpha / r, so
    alpha must equal r to reproduce today's effective scale of 1.0 (ComfyUI ignores
    alpha entirely while lokr_w2 is stored full). Only 2D lokr_w2 tensors on modules
    with both w1_key and w2_key present are eligible; everything else passes through
    unchanged and is omitted from rank_changes.

    Returns (new_tensors, {base: (orig_min_dim, new_rank)}) for eligible modules only.
    """
    out = dict(tensors)
    rank_changes: dict[str, tuple[int, int]] = {}

    for base, data in modules.items():
        if "w1_key" not in data or "w2_key" not in data:
            continue
        w2_key = data["w2_key"]
        w2 = tensors[w2_key]
        if w2.dim() != 2:
            continue

        out_dim, in_dim = w2.shape
        min_dim = min(out_dim, in_dim)
        r = max(1, round(min_dim * keep_ratio))

        if r >= min_dim:
            continue

        U, S, Vh = _svd_w2(w2, device=device)

        sqrt_S = torch.sqrt(S[:r])
        w2_a = (U[:, :r] * sqrt_S).contiguous().to(dtype=w2.dtype)
        w2_b = (sqrt_S.unsqueeze(1) * Vh[:r, :]).contiguous().to(dtype=w2.dtype)

        del out[w2_key]
        out[f"{base}.lokr_w2_a"] = w2_a
        out[f"{base}.lokr_w2_b"] = w2_b

        alpha_dtype = tensors[data["alpha_key"]].dtype if "alpha_key" in data else torch.float32
        out[f"{base}.alpha"] = torch.tensor(float(r), dtype=alpha_dtype)

        rank_changes[base] = (min_dim, r)

    return out, rank_changes


def apply_lokr_effective_rank(
    tensors: dict, modules: dict, device: str = "cpu"
) -> tuple[dict, dict[str, tuple[int, int]]]:
    """Truncate each module's full lokr_w2 to its entropy-based effective rank.

    Same decomposition and alpha-rewrite behavior as apply_lokr_svd, but the target
    rank r = round(exp(entropy(s²/Σs²))) is computed per-module from lokr_w2's own
    singular value spectrum instead of a fixed ratio — no threshold to tune.

    Returns (new_tensors, {base: (orig_min_dim, new_rank)}) for eligible modules only.
    """
    out = dict(tensors)
    rank_changes: dict[str, tuple[int, int]] = {}

    for base, data in modules.items():
        if "w1_key" not in data or "w2_key" not in data:
            continue
        w2_key = data["w2_key"]
        w2 = tensors[w2_key]
        if w2.dim() != 2:
            continue

        out_dim, in_dim = w2.shape
        min_dim = min(out_dim, in_dim)

        U, S, Vh = _svd_w2(w2, device=device)

        s2 = S.cpu().float().numpy() ** 2
        total = s2.sum()
        if total <= 0:
            continue

        p = s2 / total
        nonzero = p > 0
        entropy = float(-np.sum(p[nonzero] * np.log(p[nonzero])))
        effective_rank = float(np.exp(entropy))
        r = int(min(min_dim, max(1, round(effective_rank))))

        rank_changes[base] = (min_dim, r)

        if r >= min_dim:
            continue

        sqrt_S = torch.sqrt(S[:r])
        w2_a = (U[:, :r] * sqrt_S).contiguous().to(dtype=w2.dtype)
        w2_b = (sqrt_S.unsqueeze(1) * Vh[:r, :]).contiguous().to(dtype=w2.dtype)

        del out[w2_key]
        out[f"{base}.lokr_w2_a"] = w2_a
        out[f"{base}.lokr_w2_b"] = w2_b

        alpha_dtype = tensors[data["alpha_key"]].dtype if "alpha_key" in data else torch.float32
        out[f"{base}.alpha"] = torch.tensor(float(r), dtype=alpha_dtype)

    return out, rank_changes


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _print_rank_summary(label: str, rank_changes: dict[str, tuple[int, int]], modules: dict):
    changed = {b: rc for b, rc in rank_changes.items() if rc[0] != rc[1]}
    eligible = len(rank_changes)
    total_modules = len(modules)
    skipped = total_modules - eligible
    print(f"  {label}: {len(changed)}/{eligible} eligible modules reduced ({skipped} skipped: not full/2D)")

    savings = []
    for base, (orig_min_dim, new_rank) in changed.items():
        savings.append((base, orig_min_dim, new_rank))
    savings.sort(key=lambda x: x[1] - x[2], reverse=True)
    for base, orig_min_dim, new_rank in savings[:8]:
        print(f"    {base}: rank {orig_min_dim} -> {new_rank}")
    if len(savings) > 8:
        print(f"    ... ({len(savings) - 8} more)")
    print()


def main():
    parser = argparse.ArgumentParser(description="LoKr rank reduction: SVD-truncate full lokr_w2 tensors")
    parser.add_argument("input", help="Input safetensors LoKr file")
    parser.add_argument("output_dir", help="Directory to write output variants")
    parser.add_argument(
        "--experiments",
        nargs="*",
        metavar="NAME",
        default=None,
        help="Run only these experiments (default: all). Names: svd, effective_rank",
    )
    parser.add_argument(
        "--svd-keep-ratio",
        type=float,
        default=0.5,
        help="SVD truncation: fraction of rank to keep (default: 0.5)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU processing")
    args = parser.parse_args()

    _exps = set(args.experiments) if args.experiments is not None else None

    def want(name: str) -> bool:
        return _exps is None or name in _exps

    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print(f"Loading: {input_path}")
    tensors, metadata = load_with_metadata(input_path)
    print(f"  Metadata keys: {list(metadata.keys())}")
    modules = group_modules_lokr(tensors)
    print(f"  Modules found: {len(modules)}")

    if want("svd"):
        reduced, rank_changes = apply_lokr_svd(tensors, modules, keep_ratio=args.svd_keep_ratio, device=device)
        out_path = output_dir / f"{stem}-svd.safetensors"
        extra_meta = {
            "lokr_reduce.source": input_path.name,
            "lokr_reduce.method": "lokr_svd",
            "lokr_reduce.keep_ratio": str(args.svd_keep_ratio),
            "lokr_reduce.modules_reduced": str(len({b for b, rc in rank_changes.items() if rc[0] != rc[1]})),
        }
        save_variant(reduced, out_path, "lokr SVD truncation", metadata, extra_meta)
        _print_rank_summary(f"LoKr SVD truncation (keep_ratio={args.svd_keep_ratio})", rank_changes, modules)

    if want("effective_rank"):
        reduced, rank_changes = apply_lokr_effective_rank(tensors, modules, device=device)
        out_path = output_dir / f"{stem}-effrank.safetensors"
        extra_meta = {
            "lokr_reduce.source": input_path.name,
            "lokr_reduce.method": "lokr_effective_rank",
            "lokr_reduce.modules_reduced": str(len({b for b, rc in rank_changes.items() if rc[0] != rc[1]})),
        }
        save_variant(reduced, out_path, "lokr effective-rank truncation", metadata, extra_meta)
        _print_rank_summary("LoKr effective-rank truncation", rank_changes, modules)

    print("Done.")


if __name__ == "__main__":
    main()
