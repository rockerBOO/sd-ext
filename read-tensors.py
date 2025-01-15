import argparse
import torch

from safetensors import safe_open


def main(args):
    file = safe_open(args.path, "pt")

    print(args.path)

    for i, key in enumerate(file.keys()):
        t = file.get_tensor(key)
        print(key)

        if t.dtype == torch.int64:
            if len(t.shape) == 1 and t.shape[0] == 1:
                print(t.dtype, t.size(), "items", t.item())
            else:
                print(t.dtype, t.size(), "tensor", t)

        elif len(t.shape) == 0:
            print("No tensor")
            continue
        else:
            print(t.dtype, t.size(), "norm", t.norm().item())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "path", help="Path to the safetensors file to read the tensors"
    )
    args = argparser.parse_args()
    main(args)
