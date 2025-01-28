import argparse
import re

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def convert_module_ids(
    safetensors_file,
    input_start_block,
    input_end_block,
    output_start_block,
    output_file,
):
    """
    Converts the module IDs of a Safetensors file from one block to another block.

    Args:
    - safetensors_file (str): Path to the Safetensors file to modify.
    - input_start_block (int): Start block number of the input range.
    - input_end_block (int): End block number of the input range.
    - output_start_block (int): Start block number of the output range.

    Returns:
    None
    """

    print(f"Loading {safetensors_file}")
    # Load the Safetensors file
    weights = load_file(safetensors_file)

    with safe_open(safetensors_file, framework="pt") as f:
        metadata = f.metadata()

    # Create a mapping of input block IDs to output block IDs
    block_mapping = {
        i: output_start_block + (i - input_start_block)
        for i in range(input_start_block, input_end_block + 1)
    }

    print(f"Block mapping: {block_mapping}")

    # Iterate through the weights and convert the module IDs
    for key, value in list(weights.items()):
        # Use regular expression to extract the block number from the key
        match = re.search(r"lora_unet_double_blocks_(\d+)_.*", key)
        if match:
            block_number = int(match.group(1))
            if input_start_block <= block_number <= input_end_block:
                print(f"Converting key {key}")
                new_key = re.sub(
                    r"lora_unet_double_blocks_(\d+)_",
                    f"lora_unet_double_blocks_{block_mapping[block_number]}_",
                    key,
                )
                weights[new_key] = value
                del weights[key]

    print(f"Saving to {output_file}")
    # Save the modified weights to a new Safetensors file
    save_file(weights, output_file, metadata=metadata)


def main(args):
    if args.input_file == args.output_file:
        print("Error: Input and output files cannot be the same.")
        return

    convert_module_ids(
        args.input_file,
        args.input_start_block,
        args.input_end_block,
        args.output_start_block,
        args.output_file,
    )


if __name__ == "__main__":
    # Example usage:
    # safetensors_file = 'path_to_your_safetensors_file.safetensors'
    # input_start_block = 5
    # input_end_block = 7
    # output_start_block = 16
    # convert_module_ids(safetensors_file, input_start_block, input_end_block, output_start_block)
    parser = argparse.ArgumentParser(
        description="""
            This script converts the module IDs of a Safetensors file from one block to another block.
            It takes an input Safetensors file, input start and end block numbers, output start block number, 
            and an output Safetensors file path as arguments.

            python flex_to_flux_lora.py --input_file landscape.safetensors --output_file landscape-flex_to_flux.safetensors
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--input_file",
        type=str,
        required=True,
        help="Path to the Safetensors file to modify",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to the output Safetensors file",
    )
    parser.add_argument(
        "-s",
        "--input_start_block",
        type=int,
        default=5,
        help="Start block number of the input range",
    )
    parser.add_argument(
        "-e",
        "--input_end_block",
        type=int,
        default=7,
        help="End block number of the input range",
    )
    parser.add_argument(
        "-b",
        "--output_start_block",
        type=int,
        default=16,
        help="Start block number of the output range",
    )
    args = parser.parse_args()

    main(args)
