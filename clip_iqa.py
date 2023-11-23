# Copyright © 2023 Dave Lage
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import random

import torch
from datasets import load_dataset, Image

from torchvision import transforms
from pathlib import Path
import PIL


TRANSFORMS = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(299, 299), antialias=True),
        transforms.PILToTensor(),
    ]
)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(data):
    return [
        {
            "image": PIL.Image.open(d["image"]["path"]),
            "image_file": d["image"]["path"],
        }
        for d in data
    ]


def main(args):
    seed = args.seed
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    from torchmetrics.multimodal import CLIPImageQualityAssessment

    prompts = args.prompts.split(";")
    prompts = [[r.strip() for r in p.strip().split(".") if r != ""] for p in prompts]
    results = []

    for p in prompts:
        # we are dealing with built in options
        if len(p) == 1:
            results.append(p[0])
        else:
            results.append(tuple(p))
    prompts = tuple(results)

    metric = CLIPImageQualityAssessment(prompts=prompts)
    metric.to(device)

    print(metric)

    ds = load_dataset("imagefolder", data_dir=args.data_dir, split="train").cast_column(
        "image", Image(decode=False)
    )

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )

    print(f"Images: {len(ds)} batches: {len(dataloader)}")

    for i, images in enumerate(dataloader):
        for image in images:
            results = metric(torch.stack([TRANSFORMS(image["image"])]).to(device))

            print(f"{Path(image['image_file']).name}")
            for key, value in results.items():
                print(f"\t{f'{key}:':<10} {value.item():.3%}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""CLIP Image Quality Assessment

    --data_dir "/path/to/images"
              Directory of images to run the prompts on.
            
    --prompts "quality;brightness;A super good photo. A super bad photo;"

              Prompts as contrast with "positive" and "negative". 
              Positive and negative separated by `.` Eg: Good. Bad.
              Prompts separated by `;` quality;brightness;
              Prompts input is a string so make sure to `"`quote`"` it.

              See <https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_iqa.html> 
                 for more details.

              Built in options (use as a single):

                quality: “Good photo.” vs “Bad photo.”
                brightness: “Bright photo.” vs “Dark photo.”
                noisiness: “Clean photo.” vs “Noisy photo.”
                colorfullness: “Colorful photo.” vs “Dull photo.”
                sharpness: “Sharp photo.” vs “Blurry photo.”
                contrast: “High contrast photo.” vs “Low contrast photo.”
                complexity: “Complex photo.” vs “Simple photo.”
                natural: “Natural photo.” vs “Synthetic photo.”
                happy: “Happy photo.” vs “Sad photo.”
                scary: “Scary photo.” vs “Peaceful photo.”
                new: “New photo.” vs “Old photo.”
                warm: “Warm photo.” vs “Cold photo.”
                real: “Real photo.” vs “Abstract photo.”
                beutiful: “Beautiful photo.” vs “Ugly photo.”
                lonely: “Lonely photo.” vs “Sociable photo.”
                relaxing: “Relaxing photo.” vs “Stressful photo.”
    """, formatter_class=argparse.RawTextHelpFormatter
    )

    argparser.add_argument(
        "--seed", type=int, default=1234, help="Seed for random and torch"
    )
    argparser.add_argument(
        "--data_dir",
        required=True,
        help="Data dir",
    )

    argparser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="list of prompts separated by comma",
    )

    argparser.add_argument("--device", default=None, help="Set device to use")

    args = argparser.parse_args()

    main(args)
