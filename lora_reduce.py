"""
LoRA reduction experiments via alpha scaling.

Produces variants of a LoRA with reduced effective scale:
  - Global alpha reduction (all layers scaled equally)
  - Selective alpha reduction (only outlier layers above a threshold)

Usage:
    python lora_reduce.py model.safetensors output_dir/
    python lora_reduce.py model.safetensors output_dir/ --global-scales 0.75 0.5
    python lora_reduce.py model.safetensors output_dir/ --selective-scale 0.5 --threshold 1.5
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def parse_module_name(module_name: str) -> tuple[str, int, str]:
    m = re.match(r".*_(double|single)_blocks_(\d+)_(.*)", module_name)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return "other", -1, module_name


def group_modules(tensors: dict) -> dict:
    """Group tensors by base module name, returning up/down/alpha per module.

    Recognizes both sd-scripts keys (.lora_up.weight/.lora_down.weight/.alpha)
    and PEFT keys (.lora_B.weight/.lora_A.weight). PEFT's A is the down-projection
    and B is the up-projection, matching sd-scripts' down/up roles. PEFT files
    typically carry no per-layer alpha tensor; callers already treat a missing
    alpha_key as scale = 1.0 (see compute_effective_scales and friends).
    """
    modules = {}
    for key in tensors:
        if ".lora_up.weight" in key:
            base = key.replace(".lora_up.weight", "")
            modules.setdefault(base, {})["up_key"] = key
        elif ".lora_down.weight" in key:
            base = key.replace(".lora_down.weight", "")
            modules.setdefault(base, {})["down_key"] = key
        elif ".lora_B.weight" in key:
            base = key.replace(".lora_B.weight", "")
            modules.setdefault(base, {})["up_key"] = key
        elif ".lora_A.weight" in key:
            base = key.replace(".lora_A.weight", "")
            modules.setdefault(base, {})["down_key"] = key
        elif ".alpha" in key:
            base = key.replace(".alpha", "")
            modules.setdefault(base, {})["alpha_key"] = key
    return modules


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_effective_scales(tensors: dict, modules: dict, device: str = "cpu") -> dict[str, float]:
    """Returns {base_module_name: effective_scale}."""
    scales = {}
    for base, data in modules.items():
        if "up_key" not in data or "down_key" not in data:
            continue
        up = tensors[data["up_key"]].to(device, dtype=torch.float32)
        down = tensors[data["down_key"]].to(device, dtype=torch.float32)
        alpha = tensors[data["alpha_key"]].item() if "alpha_key" in data else 1.0
        rank = down.shape[0]
        product = up.view(up.shape[0], -1) @ down.view(down.shape[0], -1)
        scales[base] = torch.norm(product).item() * (alpha / rank)
    return scales


def apply_alpha_scale(tensors: dict, modules: dict, scale_map: dict[str, float]) -> dict:
    """Returns a new tensors dict with alphas scaled per scale_map."""
    out = dict(tensors)  # shallow copy, safe since we only replace alpha tensors
    for base, factor in scale_map.items():
        if base not in modules or "alpha_key" not in modules[base]:
            continue
        alpha_key = modules[base]["alpha_key"]
        old_alpha = tensors[alpha_key].item()
        new_alpha = old_alpha * factor
        out[alpha_key] = torch.tensor(new_alpha, dtype=tensors[alpha_key].dtype)
    return out


def apply_norm_scale(tensors: dict, modules: dict, scale_map: dict[str, float]) -> dict:
    """Scale up/down weight norms by sqrt(factor) each, keeping alpha unchanged.

    Scaling both by sqrt(f) gives: norm(new_up @ new_down) = f * norm(up @ down),
    so effective_scale changes by exactly f while alpha is preserved.
    """
    out = dict(tensors)
    for base, factor in scale_map.items():
        if base not in modules:
            continue
        data = modules[base]
        if "up_key" not in data or "down_key" not in data:
            continue
        sqrt_f = float(factor ** 0.5)
        up_key = data["up_key"]
        down_key = data["down_key"]
        out[up_key] = (tensors[up_key].to(dtype=torch.float32) * sqrt_f).to(dtype=tensors[up_key].dtype)
        out[down_key] = (tensors[down_key].to(dtype=torch.float32) * sqrt_f).to(dtype=tensors[down_key].dtype)
    return out


def _low_rank_svd(up: torch.Tensor, down: torch.Tensor, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Efficient SVD of W = up @ down using the (rank × rank) core trick.

    W has rank ≤ r without ever materializing the full (out × in) matrix.
    Returns U_up, U_c, S_c, Vh_c, Vh_dn so that:
        W = (U_up @ U_c) @ diag(S_c) @ (Vh_c @ Vh_dn)
    All core matrices are (r × r); U_up is (out × r), Vh_dn is (r × in).
    """
    rank = down.shape[0]
    up_2d = up.view(up.shape[0], rank).to(device)
    down_2d = down.view(rank, -1).to(device)

    U_up, S_up, Vh_up = torch.linalg.svd(up_2d, full_matrices=False)   # out×r, r, r×r
    U_dn, S_dn, Vh_dn = torch.linalg.svd(down_2d, full_matrices=False)  # r×r, r, r×in

    # Core: (r×r) matrix whose singular values equal those of W
    C = (Vh_up * S_up.unsqueeze(1)) @ (U_dn * S_dn.unsqueeze(0))  # diag(S_up)@Vh_up @ U_dn@diag(S_dn)... wait
    # Correct: C = diag(S_up) @ Vh_up @ U_dn @ diag(S_dn)
    C = torch.diag(S_up) @ Vh_up @ U_dn @ torch.diag(S_dn)

    U_c, S_c, Vh_c = torch.linalg.svd(C, full_matrices=False)  # r×r, r, r×r

    return U_up, U_c, S_c, Vh_c, Vh_dn


def apply_svd_truncation(tensors: dict, modules: dict, keep_ratio: float = 0.75, device: str = "cpu") -> dict:
    """Truncate LoRA rank via low-rank SVD, keeping top-k singular value components.

    Uses the (rank×rank) core trick — never materializes the full (out×in) matrix.
    Alpha is scaled proportionally to preserve alpha/rank ratio.

    NOTE (LoRA+): SVD decomposes the product into orthonormal factors, which rebalances
    the up/down norm ratio toward 1.0. If this LoRA was trained with LoRA+ (lr_B >> lr_A),
    the up-dominant factorization is intentional and SVD will destroy it. The effective
    output (up@down) is preserved, but further LoRA+ training on top may behave differently.
    """
    out = dict(tensors)

    for base, data in modules.items():
        if "up_key" not in data or "down_key" not in data:
            continue
        up_key = data["up_key"]
        down_key = data["down_key"]

        up = tensors[up_key].to(dtype=torch.float32)
        down = tensors[down_key].to(dtype=torch.float32)
        orig_up_shape = up.shape

        rank = down.shape[0]
        k = max(1, round(rank * keep_ratio))

        if k >= rank:
            continue

        U_up, U_c, S_c, Vh_c, Vh_dn = _low_rank_svd(up, down, device=device)

        sqrt_S = torch.sqrt(S_c[:k])
        new_up = (U_up @ U_c[:, :k]) * sqrt_S          # (out, k)
        new_down = sqrt_S.unsqueeze(1) * (Vh_c[:k, :] @ Vh_dn)  # (k, in)

        # Restore extra dims for conv weights (e.g. (out, k, 1, 1))
        if orig_up_shape.numel() != orig_up_shape[0] * rank:
            extra = orig_up_shape[2:]
            new_up = new_up.view(new_up.shape[0], k, *([1] * len(extra)))
            new_down = new_down.view(k, down.shape[1], *extra)

        out[up_key] = new_up.to(dtype=tensors[up_key].dtype)
        out[down_key] = new_down.to(dtype=tensors[down_key].dtype)

        # Scale alpha proportionally to preserve alpha/rank ratio
        if "alpha_key" in data:
            old_alpha = tensors[data["alpha_key"]].item()
            new_alpha = old_alpha * (k / rank)
            out[data["alpha_key"]] = torch.tensor(new_alpha, dtype=tensors[data["alpha_key"]].dtype)

    return out


def apply_block_pruning(
    tensors: dict, modules: dict, block_type: str, sub_filter: str | None = None
) -> tuple[dict, list[str]]:
    """Remove all LoRA layers matching a block type and optional sub-stream filter.

    block_type: 'double' | 'single'
    sub_filter: optional substring to match within sub_layer, e.g. 'txt' or 'img'
                removes only txt_attn/txt_mlp (sub_filter='txt') or img_attn/img_mlp ('img')

    Examples:
        apply_block_pruning(..., 'double')          # remove all double_blocks layers
        apply_block_pruning(..., 'single')          # remove all single_blocks layers
        apply_block_pruning(..., 'double', 'txt')   # remove only txt stream in double blocks
        apply_block_pruning(..., 'double', 'img')   # remove only img stream in double blocks
    """
    out = dict(tensors)
    pruned = []

    for base, data in modules.items():
        m = re.match(r".*_(double|single)_blocks_\d+_(.*)", base)
        if not m:
            continue
        if m.group(1) != block_type:
            continue
        if sub_filter is not None and not m.group(2).startswith(sub_filter):
            continue

        out.pop(data.get("up_key", ""), None)
        out.pop(data.get("down_key", ""), None)
        out.pop(data.get("alpha_key", ""), None)
        pruned.append(base)

    return out, pruned


def apply_layer_pruning(
    tensors: dict, modules: dict, eff_scales: dict[str, float], threshold_x_median: float = 2.0
) -> tuple[dict, list[str]]:
    """Zero out entire LoRA layers whose effective scale exceeds threshold × median.

    Rather than scaling down outlier layers, this removes them completely — testing
    whether high-magnitude layers are contributing signal or causing artifacts.
    Sets both up and down weights to near-zero (1e-8 to preserve file structure).
    Alpha is preserved.

    Returns (new_tensors, list_of_pruned_layer_names).
    """
    if not eff_scales:
        return dict(tensors), []

    median = float(np.median(list(eff_scales.values())))
    cutoff = median * threshold_x_median

    out = dict(tensors)
    pruned = []

    for base, data in modules.items():
        if base not in eff_scales:
            continue
        if eff_scales[base] <= cutoff:
            continue
        if "up_key" not in data or "down_key" not in data:
            continue

        # Remove keys entirely — no need to carry zero weights
        out.pop(data["up_key"], None)
        out.pop(data["down_key"], None)
        out.pop(data.get("alpha_key", ""), None)
        pruned.append(base)

    return out, pruned


def apply_energy_pruning(
    tensors: dict, modules: dict, min_energy_fraction: float = 0.01, device: str = "cpu"
) -> tuple[dict, dict[str, tuple[int, int]]]:
    """Per-layer adaptive rank reduction: drop singular value components below an energy threshold.

    For each layer, computes the full SV spectrum and keeps only components whose
    individual s² / total_s² >= min_energy_fraction. This is 'uncertainty pruning':
    weak components (likely noise or underfit directions) are removed, strong ones kept.
    Rank varies per layer — high-utilization layers keep more, sparse ones get compressed more.

    NOTE (LoRA+): same caveat as SVD truncation — rebalances factorization.
    Returns (new_tensors, {base: (orig_rank, new_rank)}).
    """
    out = dict(tensors)
    rank_changes: dict[str, tuple[int, int]] = {}

    for base, data in modules.items():
        if "up_key" not in data or "down_key" not in data:
            continue
        up_key, down_key = data["up_key"], data["down_key"]
        up = tensors[up_key].to(dtype=torch.float32)
        down = tensors[down_key].to(dtype=torch.float32)
        orig_up_shape = up.shape
        rank = down.shape[0]

        U_up, U_c, S_c, Vh_c, Vh_dn = _low_rank_svd(up, down, device=device)

        s2 = S_c.cpu().float().numpy() ** 2
        total = s2.sum()
        if total <= 0:
            continue

        # Keep components above energy threshold
        keep_mask = s2 / total >= min_energy_fraction
        k = max(1, int(keep_mask.sum()))

        rank_changes[base] = (rank, k)

        if k >= rank:
            continue  # nothing to prune

        sqrt_S = torch.sqrt(S_c[:k])
        new_up = (U_up @ U_c[:, :k]) * sqrt_S
        new_down = sqrt_S.unsqueeze(1) * (Vh_c[:k, :] @ Vh_dn)

        if orig_up_shape.numel() != orig_up_shape[0] * rank:
            extra = orig_up_shape[2:]
            new_up = new_up.view(new_up.shape[0], k, *([1] * len(extra)))
            new_down = new_down.view(k, down.shape[1], *extra)

        out[up_key] = new_up.to(dtype=tensors[up_key].dtype)
        out[down_key] = new_down.to(dtype=tensors[down_key].dtype)

        if "alpha_key" in data:
            old_alpha = tensors[data["alpha_key"]].item()
            out[data["alpha_key"]] = torch.tensor(
                old_alpha * (k / rank), dtype=tensors[data["alpha_key"]].dtype
            )

    return out, rank_changes


def apply_spectral_norm_scale(tensors: dict, modules: dict, target: float | None = None, device: str = "cpu") -> tuple[dict, float, dict[str, float]]:
    """Scale weights so the spectral norm (largest singular value of up@down) matches target.

    Uses the low-rank core trick — spectral norm of W = largest singular value of
    the (rank×rank) core C, so no large matrix is ever materialized.
    Alpha is preserved unchanged.
    Returns (new_tensors, target_used, {base: spectral_norm}).
    """
    spectral_norms: dict[str, float] = {}
    for base, data in modules.items():
        if "up_key" not in data or "down_key" not in data:
            continue
        up = tensors[data["up_key"]].to(dtype=torch.float32)
        down = tensors[data["down_key"]].to(dtype=torch.float32)
        _, _, S_c, _, _ = _low_rank_svd(up, down, device=device)
        spectral_norms[base] = S_c[0].item()

    if target is None:
        target = float(np.median(list(spectral_norms.values())))

    out = dict(tensors)
    for base, spec_norm in spectral_norms.items():
        if spec_norm <= 0:
            continue
        sqrt_f = float((target / spec_norm) ** 0.5)
        up_key = modules[base]["up_key"]
        down_key = modules[base]["down_key"]
        out[up_key] = (tensors[up_key].to(dtype=torch.float32) * sqrt_f).to(dtype=tensors[up_key].dtype)
        out[down_key] = (tensors[down_key].to(dtype=torch.float32) * sqrt_f).to(dtype=tensors[down_key].dtype)

    return out, target, spectral_norms


def lora_key_to_base_key(module_name: str) -> str | None:
    """Convert LoRA base module name to base model weight key."""
    m = re.match(r".*_(double|single)_blocks_(\d+)_(.*)", module_name)
    if not m:
        return None
    block_type, block_idx, sub_layer = m.group(1), m.group(2), m.group(3)
    if "_" in sub_layer:
        last = sub_layer.rfind("_")
        sub_layer_key = sub_layer[:last] + "." + sub_layer[last + 1:]
    else:
        sub_layer_key = sub_layer
    return f"{block_type}_blocks.{block_idx}.{sub_layer_key}.weight"


def _power_iter_top_left_vec(W: torch.Tensor, n_iter: int = 32) -> torch.Tensor:
    v = torch.randn(W.shape[1], dtype=W.dtype, device=W.device)
    v = v / torch.norm(v)
    u = torch.zeros(W.shape[0], dtype=W.dtype, device=W.device)
    for _ in range(n_iter):
        u = W @ v
        u = u / torch.norm(u)
        v = W.T @ u
        v = v / torch.norm(v)
    return u


def apply_spec_ratio_reduction(
    tensors: dict, modules: dict, base_path: str, target_ratio: float = 0.9, device: str = "cpu"
) -> tuple[dict, dict[str, float]]:
    """Scale any layer where spectral_norm(LoRA) / spectral_norm(base) > target_ratio.

    Brings spec_ratio down to target_ratio by scaling both up and down by sqrt(factor).
    Alpha preserved. Layers already below target are untouched.
    Returns (new_tensors, {base: spec_ratio_before}).
    """
    out = dict(tensors)
    spec_ratios: dict[str, float] = {}

    with safe_open(base_path, framework="pt") as f:
        base_keys = set(f.keys())
        for base_name, data in modules.items():
            if "up_key" not in data or "down_key" not in data:
                continue
            base_key = lora_key_to_base_key(base_name)
            if base_key is None or base_key not in base_keys:
                continue

            up = tensors[data["up_key"]].to(dtype=torch.float32)
            down = tensors[data["down_key"]].to(dtype=torch.float32)

            # LoRA spectral norm via core trick
            _, _, S_c, _, _ = _low_rank_svd(up, down, device=device)
            lora_spec = S_c[0].item()

            # Base spectral norm via power iteration
            W_base = f.get_tensor(base_key).to(device, dtype=torch.float32)
            u_base = _power_iter_top_left_vec(W_base)
            v_base = W_base.T @ u_base
            v_base = v_base / torch.norm(v_base)
            base_spec = (u_base @ (W_base @ v_base)).item()

            spec_ratio = lora_spec / base_spec if base_spec > 0 else float("inf")
            spec_ratios[base_name] = spec_ratio

            if spec_ratio > target_ratio:
                factor = (target_ratio / spec_ratio) ** 2  # spectral scales as factor, so weight scales as sqrt
                sqrt_f = factor ** 0.5
                out[data["up_key"]] = (tensors[data["up_key"]].to(dtype=torch.float32) * sqrt_f).to(dtype=tensors[data["up_key"]].dtype)
                out[data["down_key"]] = (tensors[data["down_key"]].to(dtype=torch.float32) * sqrt_f).to(dtype=tensors[data["down_key"]].dtype)

    return out, spec_ratios


def apply_spectral_orthogonalization(
    tensors: dict, modules: dict, base_path: str, n_base_vecs: int = 4, device: str = "cpu"
) -> tuple[dict, int]:
    """Project LoRA delta to be orthogonal to the base weight's top-n left singular vectors.

    For each layer:
      W_lora = up @ down
      P = U_base[:, :n] @ U_base[:, :n].T  (projection onto base's top-n directions)
      W_orth = (I - P) @ W_lora = ((I - P) @ up) @ down

    Only up is modified — down stays identical. Alpha unchanged.
    Returns (new_tensors, n_layers_modified).
    """
    out = dict(tensors)
    n_modified = 0

    with safe_open(base_path, framework="pt") as f:
        base_keys = set(f.keys())
        for base_name, data in modules.items():
            if "up_key" not in data or "down_key" not in data:
                continue
            base_key = lora_key_to_base_key(base_name)
            if base_key is None or base_key not in base_keys:
                continue

            up_key = data["up_key"]
            up = tensors[up_key].to(device, dtype=torch.float32)
            W_base = f.get_tensor(base_key).to(device, dtype=torch.float32)
            orig_shape = up.shape

            # Compute top-n left singular vectors of base weight via iterated power method
            up_2d = up.view(up.shape[0], -1)
            U_base_vecs = []
            W_deflated = W_base.clone()
            for _ in range(n_base_vecs):
                u = _power_iter_top_left_vec(W_deflated)
                U_base_vecs.append(u)
                # Deflate: remove this direction
                W_deflated = W_deflated - u.unsqueeze(1) * (u @ W_deflated)

            U_base = torch.stack(U_base_vecs, dim=1)  # (out, n)

            # Project up to be orthogonal: up_orth = (I - U U^T) @ up
            proj = U_base @ (U_base.T @ up_2d)  # (out, rank)
            up_orth = up_2d - proj

            out[up_key] = up_orth.view(orig_shape).to(dtype=tensors[up_key].dtype)
            n_modified += 1

    return out, n_modified


def load_with_metadata(path: Path) -> tuple[dict, dict[str, str]]:
    """Load tensors and metadata from a safetensors file."""
    tensors = load_file(str(path))
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
    return tensors, metadata


def save_variant(tensors: dict, path: Path, label: str, metadata: dict[str, str], extra_meta: dict[str, str]):
    merged = {**metadata, **extra_meta}
    save_file(tensors, str(path), metadata=merged)
    print(f"  Saved {label}: {path.name}")


def main():
    parser = argparse.ArgumentParser(description="LoRA reduction experiments via alpha scaling")
    parser.add_argument("input", help="Input safetensors LoRA file")
    parser.add_argument("output_dir", help="Directory to write output variants")
    parser.add_argument(
        "--global-scales",
        type=float,
        nargs="+",
        default=[0.75, 0.5],
        help="Global alpha scale factors to produce (default: 0.75 0.5)",
    )
    parser.add_argument(
        "--selective-scale",
        type=float,
        default=0.5,
        help="Alpha scale factor applied to outlier layers only (default: 0.5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Outlier threshold as multiple of median effective scale (default: 1.5)",
    )
    parser.add_argument(
        "--svd-keep-ratio",
        type=float,
        default=0.75,
        help="Fraction of singular values to keep in SVD truncation (default: 0.75)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU processing (default: use CUDA if available)",
    )
    parser.add_argument(
        "--base-model",
        help="Path to base model safetensors for spec-ratio and orthogonalization experiments",
    )
    parser.add_argument(
        "--spec-ratio-target",
        type=float,
        default=0.9,
        help="Target spectral ratio for spec-ratio reduction (default: 0.9)",
    )
    parser.add_argument(
        "--orth-vecs",
        type=int,
        default=4,
        help="Number of base singular vectors to orthogonalize against (default: 4)",
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=2.0,
        help="Layer pruning: zero out layers above this × median eff_scale (default: 2.0)",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.01,
        help="Energy pruning: drop SV components below this fraction of total layer energy (default: 0.01)",
    )
    parser.add_argument(
        "--max-norm",
        type=float,
        default=None,
        help="Max-norm cap: clip any layer's effective_scale to this value (default: off). "
             "Use e.g. 2× median as a starting point.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        metavar="NAME",
        help=(
            "Run only these experiments (default: all). Names: "
            "global, selective, norm-global, norm-selective, spectral, svd, "
            "svd-norm-selective, norm-selective-global, spec-ratio, spec-ratio-svd, orth, "
            "layer-prune, energy, block-prune, "
            "pruned-norm-selective, energy-norm-selective, pruned-energy, max-norm"
        ),
    )
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
    modules = group_modules(tensors)
    eff_scales = compute_effective_scales(tensors, modules, device=device)

    all_scales = np.array(list(eff_scales.values()))
    median = float(np.median(all_scales))
    print(f"  Layers: {len(eff_scales)}  Median effective scale: {median:.2f}  Max: {np.max(all_scales):.2f}")

    # Identify outliers
    outliers = {base: s for base, s in eff_scales.items() if s > median * args.threshold}
    print(f"  Outliers (>{args.threshold}x median): {len(outliers)}")
    for base, s in sorted(outliers.items(), key=lambda x: -x[1]):
        _, _, _ = parse_module_name(base)
        print(f"    {base}  eff={s:.2f}  ({s/median:.2f}x)")

    print()

    keep = args.svd_keep_ratio
    threshold = args.prune_threshold
    energy_thresh = args.energy_threshold

    # --- Experiment: Global alpha reduction ---
    if want("global"):
        for factor in args.global_scales:
            scale_map = {base: factor for base in modules}
            reduced = apply_alpha_scale(tensors, modules, scale_map)
            out_path = output_dir / f"{stem}-global{factor:.2f}.safetensors"
            new_scales = np.array([eff_scales[base] * factor for base in eff_scales])
            extra_meta = {
                "lora_reduce.source": input_path.name,
                "lora_reduce.method": "global_alpha",
                "lora_reduce.scale": str(factor),
                "lora_reduce.layers_affected": str(len(modules)),
                "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
                "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
            }
            save_variant(reduced, out_path, f"global ×{factor}", metadata, extra_meta)
            print(f"    median: {np.median(new_scales):.2f}  max: {np.max(new_scales):.2f}")
        print()

    # --- Experiment: Selective outlier reduction ---
    if want("selective"):
        scale_map = {base: args.selective_scale for base in outliers}
        reduced = apply_alpha_scale(tensors, modules, scale_map)
        out_path = output_dir / f"{stem}-selective{args.selective_scale:.2f}.safetensors"
        new_scales = np.array([eff_scales[base] * scale_map.get(base, 1.0) for base in eff_scales])
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "selective_alpha",
            "lora_reduce.scale": str(args.selective_scale),
            "lora_reduce.threshold": str(args.threshold),
            "lora_reduce.layers_affected": str(len(outliers)),
            "lora_reduce.outlier_layers": ", ".join(sorted(outliers.keys())),
            "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
        }
        save_variant(reduced, out_path, f"selective outliers ×{args.selective_scale}", metadata, extra_meta)
        new_scales = np.array([eff_scales[base] * scale_map.get(base, 1.0) for base in eff_scales])
        print(f"    median: {np.median(new_scales):.2f}  max: {np.max(new_scales):.2f}  (was {median:.2f} / {np.max(all_scales):.2f})")
        print()

    # --- Experiment: Global norm scaling (alpha preserved) ---
    if want("norm-global"):
        norm_target = median
        scale_map = {base: norm_target / eff_scales[base] for base in eff_scales if eff_scales[base] > 0}
        reduced = apply_norm_scale(tensors, modules, scale_map)
        out_path = output_dir / f"{stem}-norm-global.safetensors"
        new_scales = np.array([eff_scales[base] * scale_map.get(base, 1.0) for base in eff_scales])
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "norm_global",
            "lora_reduce.norm_target": f"{norm_target:.4f}",
            "lora_reduce.layers_affected": str(len(scale_map)),
            "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
        }
        save_variant(reduced, out_path, "norm global → median", metadata, extra_meta)
        print(f"    median: {np.median(new_scales):.2f}  max: {np.max(new_scales):.2f}  (was {median:.2f} / {np.max(all_scales):.2f})")
        print()

    # --- Experiment: Selective norm scaling of outliers only (alpha preserved) ---
    if want("norm-selective"):
        scale_map = {base: median / eff_scales[base] for base in outliers if eff_scales[base] > 0}
        reduced = apply_norm_scale(tensors, modules, scale_map)
        out_path = output_dir / f"{stem}-norm-selective.safetensors"
        new_scales = np.array([eff_scales[base] * scale_map.get(base, 1.0) for base in eff_scales])
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "norm_selective",
            "lora_reduce.norm_target": f"{median:.4f}",
            "lora_reduce.threshold": str(args.threshold),
            "lora_reduce.layers_affected": str(len(scale_map)),
            "lora_reduce.outlier_layers": ", ".join(sorted(scale_map.keys())),
            "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
        }
        save_variant(reduced, out_path, "norm selective outliers → median", metadata, extra_meta)
        print(f"    median: {np.median(new_scales):.2f}  max: {np.max(new_scales):.2f}  (was {median:.2f} / {np.max(all_scales):.2f})")
        print()

    # --- Experiment: Spectral normalization (alpha preserved) ---
    if want("spectral"):
        reduced, spec_target, spec_norms = apply_spectral_norm_scale(tensors, modules, device=device)
        out_path = output_dir / f"{stem}-spectral.safetensors"
        new_eff = {base: eff_scales[base] * (spec_target / spec_norms[base]) for base in spec_norms}
        new_scales = np.array(list(new_eff.values()))
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "spectral_norm",
            "lora_reduce.spectral_target": f"{spec_target:.4f}",
            "lora_reduce.layers_affected": str(len(spec_norms)),
            "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
        }
        save_variant(reduced, out_path, "spectral → median", metadata, extra_meta)
        print(f"    spectral target: {spec_target:.2f}  median eff: {np.median(new_scales):.2f}  max: {np.max(new_scales):.2f}")
        print()

    # --- Experiment: SVD truncation ---
    if want("svd"):
        reduced = apply_svd_truncation(tensors, modules, keep_ratio=keep, device=device)
        out_path = output_dir / f"{stem}-svd{keep:.2f}.safetensors"
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "svd_truncation",
            "lora_reduce.svd_keep_ratio": str(keep),
            "lora_reduce.layers_affected": str(len(modules)),
        }
        save_variant(reduced, out_path, f"SVD keep {keep:.0%}", metadata, extra_meta)
        for base, data in list(modules.items())[:3]:
            orig_rank = tensors[data["down_key"]].shape[0]
            new_rank = reduced[data["down_key"]].shape[0]
            print(f"    {base}: rank {orig_rank} → {new_rank}")
        print(f"    ... (keep_ratio={keep})")
        print()

    # --- Combo: SVD truncation → norm-selective outlier cap ---
    if want("svd-norm-selective"):
        svd_tensors = apply_svd_truncation(tensors, modules, keep_ratio=keep, device=device)
        svd_modules = group_modules(svd_tensors)
        svd_scales = compute_effective_scales(svd_tensors, svd_modules, device=device)
        svd_median = float(np.median(list(svd_scales.values())))
        svd_outliers = {base: s for base, s in svd_scales.items() if s > svd_median * args.threshold}
        scale_map = {base: svd_median / svd_scales[base] for base in svd_outliers if svd_scales[base] > 0}
        combo_tensors = apply_norm_scale(svd_tensors, svd_modules, scale_map)
        out_path = output_dir / f"{stem}-svd{keep:.2f}-norm-selective.safetensors"
        new_scales = np.array([svd_scales[base] * scale_map.get(base, 1.0) for base in svd_scales])
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "svd_then_norm_selective",
            "lora_reduce.svd_keep_ratio": str(keep),
            "lora_reduce.norm_target": f"{svd_median:.4f}",
            "lora_reduce.layers_affected": str(len(scale_map)),
            "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
        }
        save_variant(combo_tensors, out_path, f"SVD keep {keep:.0%} → norm-selective outliers", metadata, extra_meta)
        print(f"  Combo svd{keep:.2f}+norm-selective: median {np.median(new_scales):.2f}  max {np.max(new_scales):.2f}  ({len(scale_map)} outliers capped)")
        print()

    # --- Combo: norm-selective → global alpha 0.75 ---
    if want("norm-selective-global"):
        ns_scale_map = {base: median / eff_scales[base] for base in outliers if eff_scales[base] > 0}
        ns_tensors = apply_norm_scale(tensors, modules, ns_scale_map)
        ns_scales = {base: eff_scales[base] * ns_scale_map.get(base, 1.0) for base in eff_scales}
        global_scale_map = {base: 0.75 for base in modules}
        combo_tensors = apply_alpha_scale(ns_tensors, modules, global_scale_map)
        new_scales = np.array([ns_scales[base] * 0.75 for base in ns_scales])
        out_path = output_dir / f"{stem}-norm-selective-global0.75.safetensors"
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "norm_selective_then_global0.75",
            "lora_reduce.norm_target": f"{median:.4f}",
            "lora_reduce.global_scale": "0.75",
            "lora_reduce.layers_affected": str(len(modules)),
            "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
        }
        save_variant(combo_tensors, out_path, "norm-selective outliers → global ×0.75", metadata, extra_meta)
        print(f"  Combo norm-selective+global0.75: median {np.median(new_scales):.2f}  max {np.max(new_scales):.2f}")
        print()

    if args.base_model:
        # --- Combo: spec-ratio → SVD truncation ---
        if want("spec-ratio-svd"):
            sr_tensors, spec_ratios = apply_spec_ratio_reduction(
                tensors, modules, args.base_model,
                target_ratio=args.spec_ratio_target, device=device,
            )
            sr_modules = group_modules(sr_tensors)
            combo_tensors = apply_svd_truncation(sr_tensors, sr_modules, keep_ratio=keep, device=device)
            n_sr = sum(1 for r in spec_ratios.values() if r > args.spec_ratio_target)
            out_path = output_dir / f"{stem}-spec-ratio{args.spec_ratio_target:.2f}-svd{keep:.2f}.safetensors"
            extra_meta = {
                "lora_reduce.source": input_path.name,
                "lora_reduce.method": "spec_ratio_then_svd",
                "lora_reduce.spec_ratio_target": str(args.spec_ratio_target),
                "lora_reduce.svd_keep_ratio": str(keep),
                "lora_reduce.spec_layers_affected": str(n_sr),
                "lora_reduce.base_model": Path(args.base_model).name,
            }
            save_variant(combo_tensors, out_path, f"spec-ratio ≤{args.spec_ratio_target} → SVD keep {keep:.0%}", metadata, extra_meta)
            print(f"  Combo spec-ratio{args.spec_ratio_target:.2f}+svd{keep:.2f}: {n_sr} spec-ratio layers reduced, then rank truncated")
            print()

        # --- Experiment: Spec-ratio-targeted reduction ---
        if want("spec-ratio"):
            reduced, spec_ratios = apply_spec_ratio_reduction(
                tensors, modules, args.base_model,
                target_ratio=args.spec_ratio_target, device=device,
            )
            n_reduced = sum(1 for r in spec_ratios.values() if r > args.spec_ratio_target)
            out_path = output_dir / f"{stem}-spec-ratio{args.spec_ratio_target:.2f}.safetensors"
            extra_meta = {
                "lora_reduce.source": input_path.name,
                "lora_reduce.method": "spec_ratio_reduction",
                "lora_reduce.spec_ratio_target": str(args.spec_ratio_target),
                "lora_reduce.layers_affected": str(n_reduced),
                "lora_reduce.base_model": Path(args.base_model).name,
            }
            save_variant(reduced, out_path, f"spec-ratio ≤{args.spec_ratio_target}", metadata, extra_meta)
            for base_name, r in sorted(spec_ratios.items(), key=lambda x: -x[1])[:5]:
                marker = " ←" if r > args.spec_ratio_target else ""
                print(f"    {base_name}  spec_ratio={r:.3f}{marker}")
            print(f"    ({n_reduced} layers reduced)")
            print()

        # --- Experiment: Spectral orthogonalization ---
        if want("orth"):
            reduced, n_mod = apply_spectral_orthogonalization(
                tensors, modules, args.base_model,
                n_base_vecs=args.orth_vecs, device=device,
            )
            out_path = output_dir / f"{stem}-orth{args.orth_vecs}.safetensors"
            extra_meta = {
                "lora_reduce.source": input_path.name,
                "lora_reduce.method": "spectral_orthogonalization",
                "lora_reduce.orth_n_vecs": str(args.orth_vecs),
                "lora_reduce.layers_affected": str(n_mod),
                "lora_reduce.base_model": Path(args.base_model).name,
            }
            save_variant(reduced, out_path, f"orthogonalize top-{args.orth_vecs} base vecs", metadata, extra_meta)
            print(f"    {n_mod} layers orthogonalized against top-{args.orth_vecs} base singular vectors")
            print()

    # --- Experiment: Layer pruning ---
    if want("layer-prune"):
        reduced, pruned_layers = apply_layer_pruning(tensors, modules, eff_scales, threshold_x_median=threshold)
        out_path = output_dir / f"{stem}-pruned{threshold:.1f}x.safetensors"
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "layer_pruning",
            "lora_reduce.prune_threshold": f"{threshold:.2f}x_median",
            "lora_reduce.layers_pruned": str(len(pruned_layers)),
            "lora_reduce.pruned_layers": ", ".join(sorted(pruned_layers)),
        }
        save_variant(reduced, out_path, f"layer-prune >{threshold:.1f}×median", metadata, extra_meta)
        print(f"  Layer pruning (>{threshold:.1f}×median={median*threshold:.1f}): {len(pruned_layers)} layers removed")
        for name in sorted(pruned_layers, key=lambda n: -eff_scales[n]):
            print(f"    {name}  eff={eff_scales[name]:.2f}")
        print()

    # --- Experiment: Energy pruning ---
    if want("energy"):
        reduced, rank_changes = apply_energy_pruning(tensors, modules, min_energy_fraction=energy_thresh, device=device)
        out_path = output_dir / f"{stem}-energy{energy_thresh:.3f}.safetensors"
        changed = {b: (o, n) for b, (o, n) in rank_changes.items() if o != n}
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "energy_pruning",
            "lora_reduce.energy_threshold": str(energy_thresh),
            "lora_reduce.layers_affected": str(len(changed)),
        }
        save_variant(reduced, out_path, f"energy-prune threshold={energy_thresh}", metadata, extra_meta)
        rank_reductions = [(b, o, n) for b, (o, n) in rank_changes.items() if o != n]
        rank_reductions.sort(key=lambda x: x[1] - x[2], reverse=True)
        print(f"  Energy pruning (min_energy={energy_thresh}): {len(changed)}/{len(rank_changes)} layers had rank reduced")
        for name, orig_r, new_r in rank_reductions[:8]:
            short = name.replace("lora_unet_", "")
            print(f"    {short}: rank {orig_r} → {new_r}")
        if len(rank_reductions) > 8:
            print(f"    ... ({len(rank_reductions) - 8} more)")
        print()

    # --- Experiments: Block-type pruning ---
    if want("block-prune"):
        for btype, sub in [("double", None), ("single", None), ("double", "txt"), ("double", "img")]:
            reduced, pruned_layers = apply_block_pruning(tensors, modules, btype, sub)
            suffix = f"prune-{btype}" if sub is None else f"prune-{btype}-{sub}"
            out_path = output_dir / f"{stem}-{suffix}.safetensors"
            label = f"all {btype}_blocks" if sub is None else f"{btype}_blocks {sub}-stream"
            extra_meta = {
                "lora_reduce.source": input_path.name,
                "lora_reduce.method": "block_pruning",
                "lora_reduce.block_type": btype,
                "lora_reduce.sub_filter": sub or "all",
                "lora_reduce.layers_pruned": str(len(pruned_layers)),
            }
            save_variant(reduced, out_path, f"prune {label}", metadata, extra_meta)
            remaining = len(modules) - len(pruned_layers)
            print(f"  Prune {label}: {len(pruned_layers)} layers removed, {remaining} remain")
        print()

    # --- Combo: pruned + norm-selective ---
    if want("pruned-norm-selective"):
        pruned_tensors, pruned_layers = apply_layer_pruning(tensors, modules, eff_scales, threshold_x_median=threshold)
        pruned_modules = group_modules(pruned_tensors)
        pruned_scales = compute_effective_scales(pruned_tensors, pruned_modules, device=device)
        pruned_median = float(np.median(list(pruned_scales.values()))) if pruned_scales else median
        pruned_outliers = {b: s for b, s in pruned_scales.items() if s > pruned_median * args.threshold}
        ns_scale_map = {b: pruned_median / pruned_scales[b] for b in pruned_outliers if pruned_scales[b] > 0}
        combo_tensors = apply_norm_scale(pruned_tensors, pruned_modules, ns_scale_map)
        out_path = output_dir / f"{stem}-pruned{threshold:.1f}x-norm-selective.safetensors"
        new_scales = np.array([pruned_scales[b] * ns_scale_map.get(b, 1.0) for b in pruned_scales])
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "pruned_then_norm_selective",
            "lora_reduce.prune_threshold": f"{threshold:.2f}x_median",
            "lora_reduce.layers_pruned": str(len(pruned_layers)),
            "lora_reduce.norm_outliers_capped": str(len(ns_scale_map)),
            "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
        }
        save_variant(combo_tensors, out_path, f"layer-prune >{threshold:.1f}× → norm-selective", metadata, extra_meta)
        print(f"  Combo pruned+norm-selective: {len(pruned_layers)} layers removed, {len(ns_scale_map)} capped  median {np.median(new_scales):.2f}  max {np.max(new_scales):.2f}")
        print()

    # --- Combo: energy pruning + norm-selective ---
    if want("energy-norm-selective"):
        ep_tensors, ep_rank_changes = apply_energy_pruning(tensors, modules, min_energy_fraction=energy_thresh, device=device)
        ep_modules = group_modules(ep_tensors)
        ep_scales = compute_effective_scales(ep_tensors, ep_modules, device=device)
        ep_median = float(np.median(list(ep_scales.values()))) if ep_scales else median
        ep_outliers = {b: s for b, s in ep_scales.items() if s > ep_median * args.threshold}
        ep_ns_map = {b: ep_median / ep_scales[b] for b in ep_outliers if ep_scales[b] > 0}
        combo_tensors = apply_norm_scale(ep_tensors, ep_modules, ep_ns_map)
        out_path = output_dir / f"{stem}-energy{energy_thresh:.3f}-norm-selective.safetensors"
        new_scales = np.array([ep_scales[b] * ep_ns_map.get(b, 1.0) for b in ep_scales])
        ep_changed = sum(1 for o, n in ep_rank_changes.values() if o != n)
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "energy_then_norm_selective",
            "lora_reduce.energy_threshold": str(energy_thresh),
            "lora_reduce.energy_layers_affected": str(ep_changed),
            "lora_reduce.norm_outliers_capped": str(len(ep_ns_map)),
            "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
        }
        save_variant(combo_tensors, out_path, f"energy-prune {energy_thresh} → norm-selective", metadata, extra_meta)
        print(f"  Combo energy+norm-selective: {ep_changed} rank-compressed, {len(ep_ns_map)} capped  median {np.median(new_scales):.2f}  max {np.max(new_scales):.2f}")
        print()

    # --- Combo: pruned + energy pruning ---
    if want("pruned-energy"):
        pruned_tensors2, pruned_layers2 = apply_layer_pruning(tensors, modules, eff_scales, threshold_x_median=threshold)
        pruned_modules2 = group_modules(pruned_tensors2)
        combo_tensors, ep2_rank_changes = apply_energy_pruning(pruned_tensors2, pruned_modules2, min_energy_fraction=energy_thresh, device=device)
        out_path = output_dir / f"{stem}-pruned{threshold:.1f}x-energy{energy_thresh:.3f}.safetensors"
        ep2_changed = sum(1 for o, n in ep2_rank_changes.values() if o != n)
        combo_modules = group_modules(combo_tensors)
        combo_scales_arr = np.array(list(compute_effective_scales(combo_tensors, combo_modules, device=device).values()))
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "pruned_then_energy",
            "lora_reduce.prune_threshold": f"{threshold:.2f}x_median",
            "lora_reduce.layers_pruned": str(len(pruned_layers2)),
            "lora_reduce.energy_threshold": str(energy_thresh),
            "lora_reduce.energy_layers_affected": str(ep2_changed),
            "lora_reduce.eff_scale_median": f"{np.median(combo_scales_arr):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(combo_scales_arr):.4f}",
        }
        save_variant(combo_tensors, out_path, f"layer-prune >{threshold:.1f}× → energy-prune {energy_thresh}", metadata, extra_meta)
        print(f"  Combo pruned+energy: {len(pruned_layers2)} layers removed, {ep2_changed} rank-compressed  median {np.median(combo_scales_arr):.2f}  max {np.max(combo_scales_arr):.2f}")
        print()

    # --- Experiment: Max-norm cap ---
    if want("max-norm") and args.max_norm is not None:
        cap = args.max_norm
        scale_map = {base: cap / s for base, s in eff_scales.items() if s > cap and s > 0}
        capped = apply_norm_scale(tensors, modules, scale_map)
        out_path = output_dir / f"{stem}-maxnorm{cap:.2f}.safetensors"
        new_scales = np.array([eff_scales[base] * scale_map.get(base, 1.0) for base in eff_scales])
        extra_meta = {
            "lora_reduce.source": input_path.name,
            "lora_reduce.method": "max_norm",
            "lora_reduce.max_norm": str(cap),
            "lora_reduce.layers_affected": str(len(scale_map)),
            "lora_reduce.eff_scale_median": f"{np.median(new_scales):.4f}",
            "lora_reduce.eff_scale_max": f"{np.max(new_scales):.4f}",
        }
        save_variant(capped, out_path, f"max-norm cap @ {cap:.2f}", metadata, extra_meta)
        print(f"  Max-norm cap @ {cap:.2f}: {len(scale_map)} layers capped  median {np.median(new_scales):.2f}  max {np.max(new_scales):.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
