import argparse
import json

from safetensors import safe_open


def main(args):
    with safe_open(args.path, "pt") as file:
        parsed = file.metadata()
        print(json.dumps(parsed, indent=4))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "path", help="Path to the safetensors file to read the metadata"
    )
    args = argparser.parse_args()
    main(args)
