import argparse
import ast
from pathlib import Path

import torch
from safetensors.torch import load_file, safe_open, save_file

try:
    from library import model_util
except ModuleNotFoundError:
    print(
        "Requires to be with the Kohya-ss sd-scripts"
        + "https://github.com/kohya-ss/sd-scripts"
    )
    print("Copy this script into your Kohya-ss sd-scripts directory")
    import sys

    sys.exit(2)


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


# scaling max norm code credit from https://github.com/kohya-ss/sd-scripts
def apply_max_norm(state_dict, max_norm, device, scale_map={}):
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
        max_norm_value = max_norm
        for key in scale_map.keys():
            if key in downkeys[i]:
                max_norm_value = scale_map[key]

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
            updown = torch.nn.functional.conv2d(
                down.permute(1, 0, 2, 3), up
            ).permute(1, 0, 2, 3)
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


def parse_dict(input_str):
    """Convert string input into a dictionary."""
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal (dict)
        return ast.literal_eval(input_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid dictionary format: {input_str}"
        )


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
        lora_sd, max_norm, device, scale_map=args.scale_map
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
        "--scale_map",
        type=parse_dict,
        default="{}",
        help="scale map",
    )

    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="WARNING overwrites original file. "
        + "Overwrite the model with scaled normed version",
    )

    argparser.add_argument("--output", help="Output file to this file")

    argparser.add_argument(
        "--device", help="Device to run the calculations on"
    )

    args = argparser.parse_args()
    main(args)
