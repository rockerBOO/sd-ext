import torch

from lokr_reduce import _svd_w2, group_modules_lokr


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
