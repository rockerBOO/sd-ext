from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import pillow_avif
import torch
import argparse
from pathlib import Path


@torch.inference_mode()
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
        if image_path.suffix
        in [".jpg", ".jpeg", ".webp", ".avif", ".bmp", ".png"]
    ]

    images = [(image, Image.open(image)) for image in image_paths]

    prompts = [
        f"Ignore color. Consider if this image is close up or other perspective. Consider the image name {image_name.stem}. Describe woman as a simplified caption while image, consider composition, lighting, image quality, perspective. Be consise."
        for (image_name, _) in images
    ]

    batchsize = 2

    for i in range(0, len(images), batchsize):
        answers = model.batch_answer(
            [image for _, image in images[i : i + batchsize]],
            [prompt for prompt in prompts[i : i + batchsize]],
            tokenizer,
        )

        print(
            [
                (str(image[0]), answer)
                for answer, image in zip(answers, images[i : i + batchsize])
            ]
        )

        if args.output:
            for answer, image in zip(answers, images[i : i + batchsize]):
                with open(image[0].with_suffix(".txt"), "w") as f:
                    f.write(answer)


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
    argparser.add_argument("--output", default=False)

    args = argparser.parse_args()
    main(args)
