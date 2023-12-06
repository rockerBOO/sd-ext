import argparse
import os
from typing import List
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def convert_file(
    pt_filename: str,
    sf_filename: str
):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]

    metadata = {"format": "pt"}
    # Force tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    save_file(loaded, sf_filename, metadata=metadata)
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically weights to `safetensors` format.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument("pt")
    parser.add_argument("sf")

    args = parser.parse_args()

    pt = Path(args.pt)
    sf = Path(args.sf)

    assert pt.exists()

    convert_file(pt, args.sf)

    print(f"Converted {pt} to {sf} successfully")
