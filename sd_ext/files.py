from pathlib import Path
from typing import List, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def get_files(
    file_or_dir: Union[str, Path, list[str], list[Path]],
    file_ext: List[Union[str, Path]] = [],
    recursive=True
) -> List[Path]:
    files = []
    if isinstance(file_or_dir, list):
        for f in file_or_dir:
            files.extend(get_files(f, file_ext))
        return files

    if isinstance(file_or_dir, str):
        input_files = Path(file_or_dir)
    else:
        input_files = file_or_dir

    if input_files.is_dir():
        # file_list = os.listdir(input_images)

        for file in input_files.iterdir():
            # full_file = os.path.join(input_files, file)

            if file.is_dir() and recursive is True:
                files.extend(get_files(file, file_ext))
            else:
                if file.suffix in file_ext:
                    files.append(file)
    else:
        if input_files.suffix in file_ext:
            files.append(input_files)

    return files


def get_files_generator(
    file_or_dir: Union[str, Path], file_ext: List[Union[str, Path]] = []
) -> List[Path]:
    input_images = Path(file_or_dir)
    if input_images.is_dir():
        # file_list = os.listdir(input_images)

        for file in input_images.iterdir():
            # full_file = os.path.join(input_images, file)

            if file.is_dir():
                yield get_files(file, file_ext)
            else:
                if file.suffix in file_ext:
                    yield file
                else:
                    print(f"Not a file {file}")
    else:
        if file.suffix in file_ext:
            yield file


def load_file(model_file: Path):
    if isinstance(model_file, str):
        model_file = Path(model_file)

    pt_state = {}
    if model_file.suffix == ".safetensors":
        with safe_open(model_file, framework="pt") as f:
            for key in f.keys():
                pt_state[key] = f.get_tensor(key)
            metadata = f.metadata()
    else:
        pt_state = torch.load(model_file, map_location="cpu")
        metadata = {}

    return pt_state, metadata


def save_model(state_dict: dict, outfile: Union[str, Path], metadata={}):
    """
    Save the model to safetensors or pickle file
    """
    if isinstance(outfile, str):
        outfile = Path(outfile)

    if outfile.suffix == ".safetensors":
        save_file(
            state_dict,
            outfile,
            metadata,
        )
    else:
        torch.save({**state_dict, **metadata}, outfile, map_location="cpu")


# TODO: May not be working...
def merge_file(file: Union[str, Path], state_dict: dict, metadata={}):
    """
    Merge a state dict with another model and save. Overwrites data with merged data
    """
    if isinstance(file, str):
        file = Path(file)

    with safe_open(file, framework="pt") as f:
        current_dict = {{k: f.get_tensor(k)} for k in f.keys()}
        save_file({**current_dict, **state_dict}, file, metadata)
