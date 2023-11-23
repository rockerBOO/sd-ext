import argparse

import torch
from safetensors.torch import load_file, safe_open

try:
    from library import model_util
except ModuleNotFoundError:
    print("Requires to be with the Kohya-ss sd-scripts https://github.com/kohya-ss/sd-scripts")
    print("Copy this script into your Kohya-ss sd-scripts directory")
    import sys

    sys.exit(2)


def load_state_dict(file_name, dtype):
    if model_util.is_safetensors(file_name):
        sd = load_file(file_name)
        with safe_open(file_name, framework="pt") as f:
            metadata = f.metadata()
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = None

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def get_norms(state_dict, max_norm, device):
    downkeys = []
    upkeys = []
    alphakeys = []
    pre_norms = []
    post_norms = []
    longest_key = 0
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

        save_key = downkeys[i].replace(".lora_down", "")
        longest_key = len(save_key) if len(save_key) > longest_key else longest_key

        pre_norms.append({save_key: updown.norm().item()})

        if max_norm is not None:
            norm = updown.norm().clamp(min=max_norm / 2)
            desired = torch.clamp(norm, max=max_norm)

            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            post_norms.append({save_key: scalednorm.item()})

    return pre_norms, post_norms, keys_scaled, longest_key


def main(args):
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    lora_sd, metadata = load_state_dict(args.model, torch.float32)

    pre_norms, post_norms, keys_scaled, longest_key = get_norms(
        lora_sd, args.max_norm, device
    )

    def average(norms):
        total = 0

        for norm in norms:
            for v in norm.values():
                total += v

        return total / len(norms)

    for i, norm in enumerate(pre_norms):
        for k, v in norm.items():
            if args.max_norm is not None:
                print(f"{k:<{longest_key}} {v:<19} {post_norms[i][k]}")
            else:
                print(f"{k:<{longest_key}} {v}")

    print(f"Tensor norm average:                       {average(pre_norms)}")

    if args.max_norm is not None:
        print(
            f"Scaled by max normalization ({args.max_norm}) average: {average(post_norms)}"
        )
        print(f"Keys scaled: {keys_scaled}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Check the norm values for the weights in a LoRA model"
    )

    argparser.add_argument("model", help="LoRA model to check the norms of")
    argparser.add_argument(
        "--device", default=None, help="Device to run the calculations on"
    )
    argparser.add_argument(
        "--max_norm",
        type=float,
        default=None,
        help="Calculate scaled norms using max normalization",
    )

    args = argparser.parse_args()
    main(args)
