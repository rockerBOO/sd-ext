from datasets import Image, load_dataset


def load_image_dataset_from_dir(data_dir):
    ds = load_dataset(
        "imagefolder", data_dir=data_dir, split="train"
    ).cast_column("image", Image(decode=False))

    return ds
