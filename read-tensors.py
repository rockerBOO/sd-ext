import argparse
import torch

from safetensors import safe_open


def main(args):
    with safe_open(args.path, "pt") as file:
        for i, key in enumerate(file.keys()):
            t = file.get_tensor(key)
            # print(key, t.size(), t.norm())
            print(key)

            if t.dtype == torch.int64:
                if len(t.shape) == 1 and t.shape[0] == 1:
                    print(t.size(), 'items', t.item())
                else:
                    print(t.size(), 'tensor', t)
            elif len(t.shape) == 0:
                print("No tensor")
                continue
            else:
                print(t.size())
                print(t.size(), 'norm', t.norm().item())



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "path", help="Path to the safetensors file to read the metadata"
    )
    args = argparser.parse_args()
    main(args)
