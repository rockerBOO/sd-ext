from safetensors import safe_open
from safetensors.torch import save_file

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("model")
argparser.add_argument("base_name")


args = argparser.parse_args()

with safe_open(args.model, framework="pt") as f:
    tensors = dict()

    for key in f.keys():
        if args.base_name in key:
            tensors[key] = f.get_tensor(key)

    save_file(tensors, f"{args.base_name}.safetensors")
