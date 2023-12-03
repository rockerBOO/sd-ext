# MIT License
# Copyright Dave Lage (rockerBOO)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Originally from http://github.com/cyber-meow/anime_screenshot_pipeline/blob/f5a2f40e78a38b2eda4fca16c6d09cf32b38c1bd/utilities/batch_bundle_convert.py
# MIT License
# Copyright (c) 2022 CyberMeow

import argparse
from typing import List
from pathlib import Path

import torch
from torch import load, save
from safetensors import safe_open
from safetensors.torch import save_file


def bundle_state_dict(state_dicts: List[dict]) -> dict:
    bundle_dict = {}
    for embed_name, state_dict in state_dicts:
        for _key, value in state_dict.items():
            bundle_dict[f"bundle_emb.{embed_name}"] = value

    return bundle_dict


def load_state_dict(file: Path) -> dict:
    state_dict = {}
    print(file)
    with safe_open(file, framework="pt") as f:
        for key in f.keys():
            print(key)
            state_dict[key] = f.get_tensor(key)

    return state_dict


def save_state_dict(lora_file: Path, bundled: dict, outfile: Path, metadata={}):
    lora_state_dict = {**load_state_dict(lora_file), **bundled}
    save_file(
        lora_state_dict,
        outfile,
        metadata={"bundled": "Bundled using rockerBOO sd-ext", **metadata},
    )


def get_files(
    try_paths: List[Path], ext=[".safetensors"], recursive=False
) -> List[Path]:
    paths = []
    for try_path in try_paths:
        if try_path.is_dir():
            for file in try_path.iterdir():
                if file.suffix in ext:
                    paths.append(file)
                elif file.is_dir() and recursive:
                    paths.extend(get_files([file], ext, recursive))

        else:
            if try_path.suffix in ext:
                paths.append(try_path)

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool for packing and unpacking LoRA and embeddings."
    )
    parser.add_argument(
        "--lora_path",
        default=[],
        type=str,
        nargs="+",
        help="Paths to LoRA model files.",
    )
    parser.add_argument(
        "--emb_path", default=[], type=str, nargs="+", help="Paths to embedding files."
    )
    parser.add_argument(
        "--dst_dir",
        default=None,
        type=str,
        help="Destination directory for output files.",
    )
    parser.add_argument(
        "--lora_ext",
        default=[".safetensors"],
        type=str,
        nargs="+",
        help="Extensions for LoRA files.",
    )
    parser.add_argument(
        "--emb_ext",
        default=[".safetensors"],
        type=str,
        nargs="+",
        help="Extensions for embedding files.",
    )

    parser.add_argument("--verbose", default=1, type=int, help="Verbosity level.")
    args = parser.parse_args()

    lora_paths = [Path(lora_path) for lora_path in args.lora_path]
    embedding_paths = [Path(embedding_path) for embedding_path in args.emb_path]
    lora_files = get_files(lora_paths, args.lora_ext)
    embedding_files = get_files(embedding_paths, args.emb_ext)

    # print(lora_files)
    # print(embedding_files)
    # Now we want to make a bundle between each embedding into each lora file

    for lora_file in lora_files:
        for embedding_file in embedding_files:
            embedding_dict = load_state_dict(embedding_file)
            # print([k for k in embedding_dict.keys()])
            # import sys
            # sys.exit(2)
            bundled = bundle_state_dict([(embedding_file.stem, embedding_dict)])
            outfile = Path(Path(args.dst_dir) / f"bundle-{lora_file.stem}.safetensors")
            save_state_dict(
                lora_file,
                bundled,
                outfile,
                {
                    "lora_file": str(lora_file.name),
                    "embedding_file": str(embedding_file.name),
                },
            )
