# Copyright © 2023 Dave Lage (rockerBOO)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import PIL
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import Dataset, Image, load_dataset
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from torchmetrics.multimodal import CLIPImageQualityAssessment
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


def generate_images(batch_size, prompts, sd_pipeline, args):
    print("Prompts")
    for sample_prompt in prompts:
        print(f"\t{sample_prompt}")

    prompts_images = []
    for i in range(len(prompts) // batch_size):
        prompts = prompts[i * batch_size : i * batch_size + batch_size]

        generated_images = sd_pipeline(
            prompts,
            num_images_per_prompt=1,
            num_inference_steps=15,
        ).images

        prompts_images.append((prompts, generated_images))

    return prompts_images


def get_images_from_lora(
    batch_size, lora_file, sd_pipeline, prompts, accelerator=None, args=None
):
    # only supporting safetensor files here
    if lora_file.suffix != ".safetensors":
        print(f"Skipping {lora_file.name} because not safetensors file")
        return None

    sd_pipeline.unload_lora_weights()
    if accelerator is not None:
        accelerator.clear()
    sd_pipeline.load_lora_weights(lora_file, weight_name=lora_file.name)

    prompt_images = generate_images(batch_size, prompts, sd_pipeline, args)

    return prompt_images


def setup_sd_pipeline(args):
    model_ckpt = args.pretrained_model_name_or_path

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_ckpt, subfolder="scheduler"
    )

    pipeline_kwargs = {}

    if args.vae is not None:
        vae_path = Path(args.vae)
        if vae_path.is_file():
            pipeline_kwargs["vae"] = AutoencoderKL.from_single_file(vae_path)
        else:
            pipeline_kwargs["vae"] = AutoencoderKL.from_pretrained(args.vae)

    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        model_ckpt,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True,
        scheduler=scheduler,
        **pipeline_kwargs,
    )

    if args.xformers:
        sd_pipeline.enable_xformers_memory_efficient_attention()

    if args.ti_embedding_file is not None:
        ti_embedding_file = Path(args.ti_embedding_file)
        sd_pipeline.load_textual_inversion(
            args.ti_embedding_file, weight_name=ti_embedding_file.name
        )

    if args.sliced_vae:
        sd_pipeline.enable_vae_slicing()

    if args.cpu_offloading:
        sd_pipeline.enable_sequential_cpu_offload()

    if args.model_offloading:
        sd_pipeline.enable_model_cpu_offload()

    if args.xformers is None:
        sd_pipeline.unet.set_attn_processor(AttnProcessor2_0())
    # sd_pipeline.unet = torch.compile(
    #     sd_pipeline.unet, mode="reduce-overhead", fullgraph=True
    # )
    return sd_pipeline


def generate_dataset(
    sd_pipeline, prompts=None, accelerator=None, args=None, generator=None
):
    images = generate_images_from_loras(sd_pipeline, accelerator, args)

    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    dataset = Dataset.from_list(images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        worker_init_fn=seed_worker,
        generator=generator,
        # collate_fn=lambda f ,
    )

    return dataset, dataloader


def score_dataset(dataloader, metric, device, args):
    if args.output_scores is not None:
        output_file = Path(args.output_scores)

        csv_filename = (
            output_file / "clip_iqa.csv"
            if output_file.is_dir()
            else output_file
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

    return scores


def generate_images_from_loras(sd_pipeline, prompts=None, args=None):
    if prompts is None:
        prompts = get_prompts(args)

    seed = args.seed
    batch_size = args.batch_size

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    sd_pipeline = setup_sd_pipeline(args).to(device)

    accelerator = Accelerator()
    sd_pipeline = accelerator.prepare(sd_pipeline)

    images = []
    with accelerator.autocast(), torch.inference_mode():
        for file in args.lora_files:
            lora_path = Path(file)

            if lora_path.is_dir():
                for lora_file in lora_path.iterdir():
                    lora_images = get_images_from_lora(
                        batch_size,
                        lora_file,
                        sd_pipeline,
                        prompts,
                        accelerator,
                        args,
                    )

                    images.append((lora_file, lora_images))
            else:
                lora_images = get_images_from_lora(
                    batch_size, lora_path, sd_pipeline, accelerator, args
                )
                images.append((lora_file, lora_images))

    return images


def get_prompts(args):
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


def load_dataset_from_dir(args, generator=None):
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    ds = load_dataset(
        "imagefolder", data_dir=args.data_dir, split="train"
    ).cast_column("image", Image(decode=False))

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate_fn,
    )

    return ds, dataloader


def main(args):
    seed = args.seed

    set_seed(seed)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    metric = CLIPImageQualityAssessment(
        prompts=get_prompts(args),
        model_name_or_path=args.clip_model_name_or_path,
        )
    metric.to(device)

    if args.data_dir is not None:
        ds, dataloader = load_dataset_from_dir(args)
    else:
        ds, dataloader = generate_images_from_loras(args)

    print(f"Images: {len(ds)} batches: {len(dataloader)}")

    scores = score_dataset(dataloader, metric, device, args)

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
            results = metric(
                torch.stack([TRANSFORMS(image["image"])]).to(device)
            )
            scores = []
            image_file = Path(image["image_file"])
            # print(f"{image_file.name}")
            for key, value in results.items():
                scores.append((key, value.detach().item()))
                # print(f"\t{f'{key}:':<10} {value.item():.3%}")

            clip_iqa_scores.setdefault(image_file.absolute(), []).extend(
                scores
            )

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
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the image generation in Stable Diffusion",
    )
    argparser.add_argument(
        "--prompts",
        type=str,
        default="sharpness;brightness;quality;contrast;colorfullness;happy;beutiful",
        help="list of prompts separated by comma",
    )

    argparser.add_argument(
        "--pretrained_model_name_or_path",
        default="runwayml/stable-diffusion-v1-5",
        help="Model to load",
    )
    argparser.add_argument(
        "--clip_model_name_or_path",
        default="clip_iqa",
        help="CLIP Model to get the CLIP IQA score from",
    )

    argparser.add_argument(
        "--lora_files",
        default=None,
        nargs="+",
        help="Lora model file or files to load",
    )

    argparser.add_argument(
        "--ti_embedding_file",
        default=None,
        help="Textual inversion file to load",
    )

    argparser.add_argument(
        "--sliced_vae",
        action="store_true",
        help="Sliced VAE enables decoding large batches of images with limited"
        + " VRAM or batches with 32 images or more by decoding the batches of "
        + "latents one image at a time.",
    )

    argparser.add_argument(
        "--cpu_offloading",
        action="store_true",
        help="",
    )
    argparser.add_argument(
        "--model_offloading",
        action="store_true",
        help="",
    )

    argparser.add_argument(
        "--vae", help="VAE to apply to the generated images"
    )

    argparser.add_argument(
        "--xformers", action="store_true", help="Use XFormers"
    )

    argparser.add_argument("--device", default=None, help="Set device to use")

    args = argparser.parse_args()

    main(args)
