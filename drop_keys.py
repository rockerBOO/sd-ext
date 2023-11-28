import argparse
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


def drop_keys(state_dict, keys):
    dropped = []
    dict_keys = [key for key in state_dict.keys()]
    for key in dict_keys:
        for k in keys:
            if k in key:
                # drop key
                dropped.append(key)
                del state_dict[key]

    return state_dict, dropped


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

    keys = args.keys

    cleaned_lora_sd, dropped = drop_keys(lora_sd, keys)

    for dropped_key in dropped:
        print(f"Dropped: {dropped_key}")

    print(f"Dropped keys: {len(dropped)}")
    # print(f"Keys scaled: {keys_scaled}")
    # print(f"Scaled max norm ({args.max_norm}) average: {normed}")
    # print(f"Max norm: {max_norm}")

    save_to_file(output, cleaned_lora_sd, metadata)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Check the norm values for the weights in a LoRA model"
    )

    argparser.add_argument("model", help="LoRA model to check the norms of")

    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="WARNING overwrites original file. "
        + "Overwrite the model with scaled normed version",
    )

    argparser.add_argument("--output", help="Output file to this file")

    argparser.add_argument("--device", help="Device to run the calculations on")
    argparser.add_argument("--keys", nargs="+", help="Keys to drop")

    args = argparser.parse_args()
    main(args)
