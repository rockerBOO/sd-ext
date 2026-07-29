from lora_reduce import group_modules


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
