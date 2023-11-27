# Copyright © 2023 Dave Lage (rockerBOO)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import PIL
import torch
from datasets import Image, load_dataset
from torchvision import transforms

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
    print(f"Prompts: {prompts}")

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

    if args.output_scores is not None:
        output_file = Path(args.output_scores)

        csv_filename = (
            output_file / "clip_iqa.csv" if output_file.is_dir() else output_file
        )
        with open(csv_filename, "w") as csv_file:
            print(f"Saving CSV to {csv_filename.absolute()}")
            score_writer = csv.DictWriter(
                csv_file, fieldnames=["image_file", "prompt", "score"]
            )
            score_writer.writeheader()

            scores = get_scores(dataloader, metric, device, score_writer)
    else:
        scores = get_scores(dataloader, metric, device, None)

    average_scores = defaultdict()
    for image_file, score_prompts in scores.items():
        for prompt, score in score_prompts:
            average_scores.setdefault(prompt, []).append(score)

    print(f"Average CLIP IQA scores for {len(dataloader)} in {args.data_dir}")
    for prompt, scores in average_scores.items():
        print(f"{prompt:<20} {sum(scores) / len(scores)}")


def get_scores(dataloader, metric, device, score_writer):
    clip_iqa_scores = defaultdict()
    for i, images in enumerate(dataloader):
        for image in images:
            results = metric(torch.stack([TRANSFORMS(image["image"])]).to(device))
            scores = []
            image_file = Path(image["image_file"])
            # print(f"{image_file.name}")
            for key, value in results.items():
                scores.append((key, value.detach().item()))
                # print(f"\t{f'{key}:':<10} {value.item():.3%}")

            clip_iqa_scores.setdefault(image_file.absolute(), []).extend(scores)

    if score_writer is not None:
        for image_file, scores in clip_iqa_scores.items():
            for prompt, score in scores:
                score_writer.writerow(
                    dict(image_file=image_file, prompt=prompt, score=score)
                )

    return clip_iqa_scores


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
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    argparser.add_argument(
        "--seed", type=int, default=1234, help="Seed for random and torch"
    )
    argparser.add_argument(
        "--data_dir",
        required=True,
        help="Data dir",
    )

    argparser.add_argument("--output_scores", help="Output scores to")

    argparser.add_argument(
        "--prompts",
        type=str,
        default="sharpness;brightness;quality;contrast;colorfullness;happy;beutiful",
        help="list of prompts separated by comma",
    )

    argparser.add_argument("--device", default=None, help="Set device to use")

    args = argparser.parse_args()

    main(args)
