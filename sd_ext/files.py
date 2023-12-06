from pathlib import Path
from typing import Union, List

from safetensors import safe_open
import torch


def get_files(
    file_or_dir: str, file_ext: List[Union[str, Path]] = []
) -> List[Path]:
    input_images = Path(file_or_dir)
    files = []
    if input_images.is_dir():
        # file_list = os.listdir(input_images)

        for file in input_images.iterdir():
            # full_file = os.path.join(input_images, file)

            if file.is_dir():
                print(f"dir {file}")
                files.extend(get_files(file, file_ext))
            else:
                print(file.suffix, file_ext)
                if file.suffix in file_ext:
                    files.append(file)
                else:
                    print(f"Not a file {file}")
    else:
        if file.suffix in file_ext:
            files.append(file)

    return files


def load_file(model_file: Path):
    pt_state = {}
    if model_file.suffix == ".safetensors":
        with safe_open(model_file, framework="pt") as f:
            for key in f.keys():
                pt_state[key] = f.get_tensor(key)
    else:
        pt_state = torch.load(model_file, map_location="cpu")

    return pt_state
