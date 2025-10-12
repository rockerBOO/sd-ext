import argparse
from pathlib import Path
import torch
import re
from safetensors.torch import save_file
from safetensors import safe_open


def should_drop_key(key, patterns):
    """Check if a key should be dropped based on exact match, partial match, or regex."""
    for pattern in patterns:
        # Try exact match first
        if key == pattern:
            return True, f"exact match: {pattern}"
        
        # Try partial match (substring)
        if pattern in key:
            return True, f"partial match: {pattern}"
        
        # Try regex match (if pattern looks like regex or has special chars)
        try:
            if re.search(pattern, key):
                return True, f"regex match: {pattern}"
        except re.error:
            # Not a valid regex, skip regex matching for this pattern
            pass
    
    return False, None


def drop_keys_from_safetensors(input_file, patterns_to_drop, output_file):
    """Drop specified keys from a safetensors file and save to output."""
    
    # Track dropped keys
    dropped_keys = []
    new_state_dict = {}
    
    # Open the safetensors file and iterate through keys
    with safe_open(input_file, framework="pt") as f:
        metadata = f.metadata()
        # Get all keys from the file
        all_keys = list(f.keys())
        
        # Filter out keys to drop and load remaining tensors
        for key in all_keys:
            should, reason = should_drop_key(key, patterns_to_drop)
            if should:
                print(reason)
                dropped_keys.append(key)
                print(f"Dropped: {key}")
            else:
                # Load the tensor for this key
                tensor = f.get_tensor(key)
                new_state_dict[key] = tensor
    
    print(f"Total dropped keys: {len(dropped_keys)}")
    print(f"Remaining keys: {len(new_state_dict)}")

    # print("Metadata")
    # print(metadata)
    
    # Save the new state dict
    save_file(new_state_dict, output_file, metadata=metadata)
    print(f"Saved cleaned model to: {output_file}")
    
    return dropped_keys


def main():
    parser = argparse.ArgumentParser(description="Drop keys from a safetensors file using exact match, partial match, or regex")
    parser.add_argument("input", type=str, help="Input safetensors file")
    parser.add_argument("output", type=str, help="Output safetensors file")
    parser.add_argument("--patterns", nargs="+", default=[], help="Patterns to match for dropping (exact, partial, or regex)")
    parser.add_argument("--patterns-file", type=str, help="File containing patterns to drop (one per line)")
    
    args = parser.parse_args()
    
    # Collect patterns to drop
    patterns_to_drop = list(args.patterns)
    
    if args.patterns_file:
        with open(args.patterns_file, 'r') as f:
            file_patterns = [line.strip() for line in f if line.strip()]
            patterns_to_drop.extend(file_patterns)
    
    if not patterns_to_drop:
        print("No patterns specified to drop. Use --patterns or --patterns-file")
        return
    
    print(f"Patterns to match: {patterns_to_drop}")
    
    # Drop keys and save
    drop_keys_from_safetensors(args.input, patterns_to_drop, args.output)


if __name__ == "__main__":
    main()
