# MIT License
# Copyright Dave Lage (rockerBOO)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
from collections import defaultdict

import PIL
import torch
from accelerate.utils import set_seed
from datasets import Dataset
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchvision import transforms
from sd_ext.sd import generate_dataset, setup_sd_generator, sd_arguments
from sd_ext.dataset import load_image_dataset_from_dir

TRANSFORMS = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(299, 299), antialias=True),
        transforms.PILToTensor(),
    ]
)


def collate_image_file(data):
    return [
        {
            "image": PIL.Image.open(d["image"]["path"]),
            "image_file": d["image"]["path"],
        }
        for d in data
    ]


def get_clip_iqa_prompts(prompts):
    """
    CLIP IQA wants a pair of prompts with positive and a negative
    """
    prompts = args.prompts.split(";")
    prompts = [
        [r.strip() for r in p.strip().split(".") if r != ""] for p in prompts
    ]
    results = []

    for p in prompts:
        # we are dealing with built in options
        if len(p) == 1:
            results.append(p[0])
        else:
            results.append(tuple(p))

    prompts = tuple(results)
    print(f"Prompts: {prompts}")

    return prompts


def main(args):
    seed = args.seed

    set_seed(seed)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    metric = CLIPImageQualityAssessment(
        prompts=get_clip_iqa_prompts(args.prompts)
    )
    metric.to(device)

    if args.data_dir is not None:
        ds = load_image_dataset_from_dir(args)
    else:
        sd_generator = setup_sd_generator(args)
        ds = generate_dataset(sd_generator)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        # worker_init_fn=seed_worker,
        # generator=generator,
        collate_fn=collate_image_file,
    )

    print(f"Images: {len(ds)} batches: {len(dataloader)}")

    scores = get_scores(dataloader, metric, device)

    average_scores = defaultdict()
    for image_file, score_prompts in scores.items():
        for prompt, score in score_prompts:
            average_scores.setdefault(prompt, []).append(score)

    print(f"Average CLIP IQA scores for {len(dataloader)} in {args.data_dir}")
    for prompt, scores in average_scores.items():
        print(f"{prompt:<20} {sum(scores) / len(scores)}")


def get_scores(dataloader, metric, device, score_writer):
    clip_iqa_scores = defaultdict()
    for i, batch in enumerate(dataloader):
        images = []
        for image in batch:
            images.append(TRANSFORMS(image["image"]))

        results = metric(torch.stack(images).to(device))

        print(results)
        clip_iqa_scores.append(results)
        # for result, image in (results, batch):
        #     scores = []
        #     image_file = Path(image["image_file"])
        #     print(f"{image_file.name}")
        #     for key, value in result.items():
        #         scores.append((key, value.detach().item()))
        #         # print(f"\t{f'{key}:':<10} {value.item():.3%}")
        #
        #     clip_iqa_scores.setdefault(image_file.absolute(), []).extend(
        #         scores
        #     )

    return clip_iqa_scores


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""CLIP Image Quality Assessment

    Can use a premade set of images to score

    --data_dir "/path/to/images"
              Directory of images to run the prompts on.

    *OR*

    Generate images using Stable Diffusion

    --num_images_to_generate 25
              Number of images to generate

    --lora_files "/path/to/lora.safetensors"
              Path to LoRA file to use when generating
            
    --prompts "quality;brightness;A super good photo. A super bad photo;"

              Prompts as contrast with "positive" and "negative".
              Positive and negative separated by `.` Eg: Good. Bad.
              Prompts separated by `;` quality;brightness;
              Prompts input is a string so make sure to "quote" the prompt.

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

    argparser.add_argument(
        "--to_generate_prompts",
        help="Prompts to generate with. Default we chose a variety of prompts",
    )
    argparser.add_argument("--output_scores", help="Output scores to")

    argparser.add_argument(
        "--num_images_to_generate",
        type=int,
        default=25,
        help="Number of images to generate if using generated images",
    )

    argparser.add_argument(
        "--prompts",
        type=str,
        default="sharpness;brightness;quality;contrast;colorfullness;happy;beutiful",
        help="list of prompts separated by comma",
    )

    argparser.add_argument(
        "--clip_model_name_or_path",
        default="clip_iqa",
        help="CLIP Model to get the CLIP IQA score from",
    )
    argparser.add_argument("--device", default=None, help="Set device to use")

    argparser = sd_arguments(argparser)
    args = argparser.parse_args()

    main(args)
