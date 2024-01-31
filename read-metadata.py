import argparse
import json

from safetensors import safe_open


def main(args):
    with safe_open(args.path, "pt") as file:
        metadata = file.metadata()

        if metadata is not None:
            for key, value in metadata.items():
                if len(value) > 0 and (value[0] == "{" or value[0] == "["):
                    print(f"{key}:", json.dumps(json.loads(value), indent=4))
                else:
                    print(f"{key}:", value)

        # print(json.dumps(parsed, indent=4))

        if args.keys is True:
            for key in file.keys():
                if args.read_tensor:
                    t = file.get_tensor(key)
                    print(key, t.size(), t.norm())
                else:
                    print(key)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "path", help="Path to the safetensors file to read the metadata"
    )
    argparser.add_argument(
        "--keys",
        action="store_true",
        help="List out all the keys in this safetensors file",
    )
    argparser.add_argument(
        "--read_tensor",
        action="store_true",
        help="List out all the keys in this safetensors file",
    )
    args = argparser.parse_args()
    main(args)
