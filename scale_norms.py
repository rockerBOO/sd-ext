import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, safe_open, save_file

from library import model_util


def load_state_dict(file_name, dtype=None):
    if model_util.is_safetensors(file_name):
        sd = load_file(file_name)
        with safe_open(file_name, framework="pt") as f:
            metadata = f.metadata()
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = None

    # for key in list(sd.keys()):
    #     if type(sd[key]) == torch.Tensor:
    #         sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict, metadata):
    if Path(file_name).suffix == ".safetensors":
        save_file(state_dict, file_name, metadata=metadata)
    else:
        print("Error: Pickle checkpoints not yet supported for saving")


def apply_max_norm(state_dict, max_norm_value, device):
    downkeys = []
    upkeys = []
    alphakeys = []
    norms = []
    keys_scaled = 0

    for key in state_dict.keys():
        if "lora_down" in key and "weight" in key:
            downkeys.append(key)
            upkeys.append(key.replace("lora_down", "lora_up"))
            alphakeys.append(key.replace("lora_down.weight", "alpha"))

    for i in range(len(downkeys)):
        down = state_dict[downkeys[i]].to(device)
        up = state_dict[upkeys[i]].to(device)
        alpha = state_dict[alphakeys[i]].to(device)
        dim = down.shape[0]
        scale = alpha / dim

        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            updown = (
                (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
            )
        elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
            updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(
                1, 0, 2, 3
            )
        else:
            updown = up @ down

        updown *= scale

        norm = updown.norm().clamp(min=max_norm_value / 2)
        desired = torch.clamp(norm, max=max_norm_value)

        ratio = desired.cpu() / norm.cpu()
        sqrt_ratio = ratio**0.5
        if ratio != 1:
            keys_scaled += 1
            state_dict[upkeys[i]] *= sqrt_ratio
            state_dict[downkeys[i]] *= sqrt_ratio
        scalednorm = updown.norm() * ratio
        norms.append(scalednorm.item())

    return keys_scaled, sum(norms) / len(norms), max(norms), state_dict


def main(args):
    if args.output is not None:
        output = args.output
    else:
        if args.overwrite is True:
            output = args.model
        else:
            raise RuntimeError("Invalid model to save to")

    lora_sd, metadata = load_state_dict(args.model)

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    max_norm = args.max_norm

    keys_scaled, normed, max_norm, scaled_lora_state_dict = apply_max_norm(
        lora_sd,
        max_norm,
        device,
    )

    print(f"Keys scaled: {keys_scaled}")
    print(f"Scaled max norm ({args.max_norm}) average: {normed}")
    print(f"Max norm: {max_norm}")

    save_to_file(output, scaled_lora_state_dict, metadata)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Check the norm values for the weights in a LoRA model"
    )

    argparser.add_argument("model", help="LoRA model to check the norms of")

    argparser.add_argument(
        "--max_norm",
        type=float,
        required=True,
        help="Max normalization to apply to the tensors",
    )
    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="WARNING overwrites original file. "
        + "Overwrite the model with scaled normed version",
    )

    argparser.add_argument("--output", help="Output file to this file")

    argparser.add_argument("--device", help="Device to run the calculations on")

    args = argparser.parse_args()
    main(args)
