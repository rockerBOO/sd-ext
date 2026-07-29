import torch

from lora_reduce import apply_effective_rank_pruning, group_modules


def test_group_modules_sd_scripts_format():
    tensors = {
        "lora_unet_x.lora_up.weight": None,
        "lora_unet_x.lora_down.weight": None,
        "lora_unet_x.alpha": None,
    }
    modules = group_modules(tensors)
    assert modules == {
        "lora_unet_x": {
            "up_key": "lora_unet_x.lora_up.weight",
            "down_key": "lora_unet_x.lora_down.weight",
            "alpha_key": "lora_unet_x.alpha",
        }
    }


def test_group_modules_peft_format():
    tensors = {
        "base_model.model.x.lora_B.weight": None,
        "base_model.model.x.lora_A.weight": None,
    }
    modules = group_modules(tensors)
    assert modules == {
        "base_model.model.x": {
            "up_key": "base_model.model.x.lora_B.weight",
            "down_key": "base_model.model.x.lora_A.weight",
        }
    }


def test_group_modules_peft_format_no_alpha_key_present():
    tensors = {
        "base_model.model.x.lora_B.weight": None,
        "base_model.model.x.lora_A.weight": None,
    }
    modules = group_modules(tensors)
    assert "alpha_key" not in modules["base_model.model.x"]


def _orthogonal(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    return q


def _make_layer(singular_values: list[float], seed_offset: int = 0):
    """Build up/down (both n x n, n = len(singular_values)) whose product
    up @ down has exactly `singular_values` as its singular values.

    up = U (orthogonal), down = diag(S) @ Vh (Vh orthogonal), so
    up @ down = U @ diag(S) @ Vh is already a valid SVD of the product.
    """
    n = len(singular_values)
    up = _orthogonal(n, seed=1 + seed_offset)
    vh = _orthogonal(n, seed=2 + seed_offset)
    s = torch.tensor(singular_values, dtype=torch.float32)
    down = torch.diag(s) @ vh
    return up, down


def test_effective_rank_collapses_dominant_layer_to_rank_one():
    rank = 8
    up, down = _make_layer([10.0] + [0.01] * (rank - 1))
    tensors = {
        "x.lora_up.weight": up,
        "x.lora_down.weight": down,
    }
    modules = group_modules(tensors)

    reduced, rank_changes = apply_effective_rank_pruning(tensors, modules, device="cpu")

    orig_rank, new_rank = rank_changes["x"]
    assert orig_rank == rank
    assert new_rank == 1
    assert reduced["x.lora_up.weight"].shape == (rank, 1)
    assert reduced["x.lora_down.weight"].shape == (1, rank)


def test_effective_rank_leaves_uniform_layer_unchanged():
    rank = 8
    up, down = _make_layer([1.0] * rank, seed_offset=10)
    tensors = {
        "x.lora_up.weight": up,
        "x.lora_down.weight": down,
    }
    modules = group_modules(tensors)

    reduced, rank_changes = apply_effective_rank_pruning(tensors, modules, device="cpu")

    orig_rank, new_rank = rank_changes["x"]
    assert orig_rank == rank
    assert new_rank == rank
    # Uniform spectrum: k >= rank, so tensors pass through unchanged.
    assert torch.equal(reduced["x.lora_up.weight"], tensors["x.lora_up.weight"])
    assert torch.equal(reduced["x.lora_down.weight"], tensors["x.lora_down.weight"])


def test_effective_rank_rescales_alpha_proportionally():
    rank = 8
    up, down = _make_layer([10.0] + [0.01] * (rank - 1), seed_offset=20)
    tensors = {
        "x.lora_up.weight": up,
        "x.lora_down.weight": down,
        "x.alpha": torch.tensor(8.0),
    }
    modules = group_modules(tensors)

    reduced, rank_changes = apply_effective_rank_pruning(tensors, modules, device="cpu")

    _, new_rank = rank_changes["x"]
    assert reduced["x.alpha"].item() == 8.0 * (new_rank / rank)


def test_effective_rank_handles_conv2d_shape():
    rank = 8
    up, down = _make_layer([10.0] + [0.01] * (rank - 1), seed_offset=30)
    # Conv2d LoRA layout: up is (out, rank, 1, 1), down is (rank, in, k, k)
    up_conv = up.view(rank, rank, 1, 1)
    down_conv = down.view(rank, rank, 1, 1)
    tensors = {
        "x.lora_up.weight": up_conv,
        "x.lora_down.weight": down_conv,
    }
    modules = group_modules(tensors)

    reduced, rank_changes = apply_effective_rank_pruning(tensors, modules, device="cpu")

    _, new_rank = rank_changes["x"]
    assert reduced["x.lora_up.weight"].shape == (rank, new_rank, 1, 1)
    assert reduced["x.lora_down.weight"].shape == (new_rank, rank, 1, 1)


def test_effective_rank_handles_conv2d_shape_with_nontrivial_kernel():
    """Conv2d LoRA where up has a 1x1 kernel but down has a 3x3 kernel
    (the realistic conv-LoRA shape: up is (out, rank, 1, 1), down is
    (rank, in, 3, 3)). Regression test: the reshape must derive the down
    tensor's kernel dims from `down`'s own shape, not from up's shape
    (whose kernel dims are always 1x1), or it corrupts/crashes.
    """
    rank = 9
    up, down = _make_layer([10.0] + [0.01] * (rank - 1), seed_offset=40)
    # up: (out=9, rank=9) -> (9, 9, 1, 1)
    up_conv = up.view(rank, rank, 1, 1)
    # down: (rank=9, 9) -> (9, in=1, 3, 3)  (9 = in * 3 * 3 = 1 * 9)
    down_conv = down.view(rank, 1, 3, 3)
    tensors = {
        "x.lora_up.weight": up_conv,
        "x.lora_down.weight": down_conv,
    }
    modules = group_modules(tensors)

    reduced, rank_changes = apply_effective_rank_pruning(tensors, modules, device="cpu")

    _, new_rank = rank_changes["x"]
    assert reduced["x.lora_up.weight"].shape == (rank, new_rank, 1, 1)
    assert reduced["x.lora_down.weight"].shape == (new_rank, 1, 3, 3)
