import argparse
import numpy 

def main(args):
    results = numpy.load(args.path, allow_pickle=True)
    for key in results.keys():
        print(key)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""
Read a numpy file and view some metadata about it.

    """
    )
    argparser.add_argument("path", help="Path to the numpy file to read")
    argparser.add_argument(
        "--keys",
        action="store_true",
        help="List out all the keys in this numpy file",
    )
    args = argparser.parse_args()
    main(args)
