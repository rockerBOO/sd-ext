from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import argparse
from pathlib import Path


def main(args):
    model_id = "vikhyatk/moondream2"
    revision = "2024-04-02"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=torch.float16,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    image_paths = []

    for image in args.images:
        path = Path(image)
        if path.is_dir():
            image_paths.extend([image for image in path.iter_dir()])
        else:
            image_paths.append(image)

    images = [Image.open(image) for image in image_paths]

    prompts = ["Describe this image with simplified language."] * len(images)
    answers = model.batch_answer(images, prompts, tokenizer)

    print(answers)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="""
    Generate captions using moondream VQA model.

    Uses fp16 model on CUDA.

    $ python moondream.py /path/to/image.png

    $ python moondream.py /path/to/image.png /path/to/image2.png

    $ python moondream.py /path/to/dir/images
    """, formatter_class=argparse.RawTextHelpFormatter)

    argparser.add_argument(
        "images",
        nargs="+",
        help="List of image or image directories",
    )

    args = argparser.parse_args()
    main(args)
