import argparse
import json
from pathlib import Path
from functools import reduce

import torch
from safetensors import safe_open


# https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings/45846841#45846841
def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."),
        ["", "K", "M", "B", "T"][magnitude],
    )


def create_nested_dict(key_str, value, split_on="."):
    parts = key_str.split(split_on)
    result = {}
    current = result
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return result


def merge_dicts(dicts):
    result = {}
    for d in dicts:
        stack = [(d, result)]
        while stack:
            src, dst = stack.pop()
            for key, value in src.items():
                if isinstance(value, dict):
                    dst.setdefault(key, {})
                    stack.append((value, dst[key]))
                else:
                    dst[key] = value
    return result


def main(args):
    file = safe_open(args.path, "pt")

    print(args.path)

    total_parameters = 0
    norms: list[float] = []

    file_path = Path(args.path)

    results = {}

    for i, key in enumerate(file.keys()):
        t = file.get_tensor(key)

        try:
            assert isinstance(t, torch.Tensor)
        except:
            print(f"{key} not a tensor")
        print(key)

        size = list(t.size())
        norm = None
        dtype = t.dtype

        parameters = 0
        if len(size) > 0:
            parameters = reduce(lambda x, y: x * y, size)

        total_parameters += parameters

        if t.dtype == torch.int64:
            if len(t.shape) == 1 and t.shape[0] == 1:
                print(dtype, size, parameters, "items", t.item())
                norm = t.item()
            else:
                print(dtype, size, parameters, "tensor", t)
                norm = t.item()
        elif t.dtype == torch.float8_e4m3fn:
            norm = None
            print(dtype, size, parameters)
        elif len(t.shape) == 0:
            print("No tensor")
            continue
        else:
            norm = t.norm().item()
            print(dtype, size, parameters, "norm", norm)

        if norm is not None:
            norms.append(norm)

        result = create_nested_dict(
            key,
            {
                "dtype": str(dtype),
                "size": size,
                "parameters": parameters,
                "norm": norm,
            },
            split_on=args.split_on
        )
        results = merge_dicts([results, result])

    results = {"results": results, "name": file_path.name, "metadata": file.metadata()}

    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            json.dump(results, f)

    print(f"Average norms: {sum(norms) / len(norms)}")
    print(
        f"Total parameters: {human_format(total_parameters)} {total_parameters}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", help="Path to the safetensors file to read the tensors"
    )
    parser.add_argument("--split_on", default=".", help="split key element for compiling json dict")
    parser.add_argument("--output_json", default=None, help="json output of tensor information")
    args = parser.parse_args()
    main(args)
