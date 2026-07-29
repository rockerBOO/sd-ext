import torch

from lokr_reduce import _svd_w2, apply_lokr_svd, group_modules_lokr


def test_group_modules_lokr_full_format():
    tensors = {
        "diffusion_model.blocks.0.attn.wk.lokr_w1": None,
        "diffusion_model.blocks.0.attn.wk.lokr_w2": None,
        "diffusion_model.blocks.0.attn.wk.alpha": None,
    }
    modules = group_modules_lokr(tensors)
    assert modules == {
        "diffusion_model.blocks.0.attn.wk": {
            "w1_key": "diffusion_model.blocks.0.attn.wk.lokr_w1",
            "w2_key": "diffusion_model.blocks.0.attn.wk.lokr_w2",
            "alpha_key": "diffusion_model.blocks.0.attn.wk.alpha",
        }
    }


def test_group_modules_lokr_no_alpha_key_present():
    tensors = {
        "x.lokr_w1": None,
        "x.lokr_w2": None,
    }
    modules = group_modules_lokr(tensors)
    assert modules == {
        "x": {
            "w1_key": "x.lokr_w1",
            "w2_key": "x.lokr_w2",
        }
    }
    assert "alpha_key" not in modules["x"]


def test_group_modules_lokr_ignores_decomposed_keys():
    tensors = {
        "y.lokr_w1": None,
        "y.lokr_w2_a": None,
        "y.lokr_w2_b": None,
        "y.alpha": None,
    }
    modules = group_modules_lokr(tensors)
    assert modules == {
        "y": {
            "w1_key": "y.lokr_w1",
            "alpha_key": "y.alpha",
        }
    }
    assert "w2_key" not in modules["y"]


def test_svd_w2_reconstructs_matrix():
    torch.manual_seed(0)
    w2 = torch.randn(12, 8)
    U, S, Vh = _svd_w2(w2, device="cpu")
    reconstructed = U @ torch.diag(S) @ Vh
    assert torch.allclose(reconstructed, w2, atol=1e-4)


def _make_lokr_w2(out_dim: int, in_dim: int, singular_values: list[float], seed: int) -> torch.Tensor:
    """Build a w2 matrix with exactly `singular_values` as its singular values."""
    g = torch.Generator().manual_seed(seed)
    r = len(singular_values)
    u, _ = torch.linalg.qr(torch.randn(out_dim, r, generator=g))
    v, _ = torch.linalg.qr(torch.randn(in_dim, r, generator=g))
    s = torch.tensor(singular_values, dtype=torch.float32)
    return u @ torch.diag(s) @ v.T


def test_apply_lokr_svd_truncates_to_keep_ratio():
    w2 = _make_lokr_w2(16, 12, [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01], seed=1)
    tensors = {
        "x.lokr_w1": torch.eye(4),
        "x.lokr_w2": w2,
    }
    modules = group_modules_lokr(tensors)

    reduced, rank_changes = apply_lokr_svd(tensors, modules, keep_ratio=0.5, device="cpu")

    orig_min_dim, new_rank = rank_changes["x"]
    assert orig_min_dim == 12
    assert new_rank == 6  # round(12 * 0.5)
    assert "x.lokr_w2" not in reduced
    assert reduced["x.lokr_w2_a"].shape == (16, 6)
    assert reduced["x.lokr_w2_b"].shape == (6, 12)


def test_apply_lokr_svd_sets_alpha_to_rank():
    w2 = _make_lokr_w2(16, 12, [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01], seed=2)
    tensors = {
        "x.lokr_w1": torch.eye(4),
        "x.lokr_w2": w2,
        "x.alpha": torch.tensor(9.9992e9),
    }
    modules = group_modules_lokr(tensors)

    reduced, rank_changes = apply_lokr_svd(tensors, modules, keep_ratio=0.5, device="cpu")

    _, new_rank = rank_changes["x"]
    assert reduced["x.alpha"].item() == float(new_rank)


def test_apply_lokr_svd_adds_alpha_when_missing():
    w2 = _make_lokr_w2(16, 12, [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01], seed=3)
    tensors = {
        "x.lokr_w1": torch.eye(4),
        "x.lokr_w2": w2,
    }
    modules = group_modules_lokr(tensors)

    reduced, rank_changes = apply_lokr_svd(tensors, modules, keep_ratio=0.5, device="cpu")

    _, new_rank = rank_changes["x"]
    assert "x.alpha" in reduced
    assert reduced["x.alpha"].item() == float(new_rank)


def test_apply_lokr_svd_skips_ineligible_modules():
    tensors = {
        # already decomposed: no w2_key
        "y.lokr_w1": torch.eye(4),
        "y.lokr_w2_a": torch.randn(16, 4),
        "y.lokr_w2_b": torch.randn(4, 12),
        # 4D conv w2: has w2_key, but not 2D
        "z.lokr_w1": torch.eye(4),
        "z.lokr_w2": torch.randn(16, 12, 3, 3),
    }
    modules = group_modules_lokr(tensors)

    reduced, rank_changes = apply_lokr_svd(tensors, modules, keep_ratio=0.5, device="cpu")

    assert "y" not in rank_changes
    assert "z" not in rank_changes
    # tensors pass through completely unchanged
    assert torch.equal(reduced["y.lokr_w2_a"], tensors["y.lokr_w2_a"])
    assert torch.equal(reduced["z.lokr_w2"], tensors["z.lokr_w2"])


def test_apply_lokr_svd_no_reduction_when_ratio_is_one():
    w2 = _make_lokr_w2(16, 12, [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01], seed=4)
    tensors = {
        "x.lokr_w1": torch.eye(4),
        "x.lokr_w2": w2,
    }
    modules = group_modules_lokr(tensors)

    reduced, rank_changes = apply_lokr_svd(tensors, modules, keep_ratio=1.0, device="cpu")

    assert "x" not in rank_changes
    assert torch.equal(reduced["x.lokr_w2"], tensors["x.lokr_w2"])
