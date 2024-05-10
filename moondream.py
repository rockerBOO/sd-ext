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
            image_paths.extend([image for image in path.iterdir()])
        else:
            image_paths.append(image)

    image_paths = [
        image_path
        for image_path in image_paths
        if image_path.suffix in [".jpg", ".png"]
    ]

    images = [(image, Image.open(image)) for image in image_paths]

    prompts = [
        "Describe this image with simplified language. Consider the pose and composition of the image.  Consider the camera, perspective, lighting of the image. Describe it like a caption to the image."
    ] * len(images)

    batchsize = 2

    for i in range(0, len(images), batchsize):
        answers = model.batch_answer(
            [image for _, image in images[i : i + batchsize]],
            prompts,
            tokenizer,
        )

        print(
            [
                (str(image[0]), answer)
                for answer, image in zip(answers, images[i : i + batchsize])
            ]
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""
    ! Work In Progress, may have errors or be incomplete...

    ---

    Generate captions using moondream VQA model.

    Uses fp16 model on CUDA.

    $ python moondream.py /path/to/image.png

    $ python moondream.py /path/to/image.png /path/to/image2.png

    $ python moondream.py /path/to/dir/images
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    argparser.add_argument(
        "images",
        nargs="+",
        help="List of image or image directories",
    )

    args = argparser.parse_args()
    main(args)
