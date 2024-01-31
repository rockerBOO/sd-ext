import argparse
import json

import pickle


def main(args):
    with open(args.path) as f:
        results = pickle.load(f)

        for key in results.keys():
            print(key)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""
Warning

The pickle module is not secure. Only unpickle data you trust.

It is possible to construct malicious pickle data which will execute arbitrary 
code during unpickling. Never unpickle data that could have come from an 
untrusted source, or that could have been tampered with.


    """
    )
    argparser.add_argument("path", help="Path to the pickle file to read")
    argparser.add_argument(
        "--keys",
        action="store_true",
        help="List out all the keys in this safetensors file",
    )
    args = argparser.parse_args()
    main(args)
