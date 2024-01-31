import argparse
import csv
import random
from pathlib import Path

import numpy
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torchmetrics.multimodal.clip_score import CLIPScore
from sd_ext.sd import setup_sd_generator, sd_arguments
from sd_ext.format import to_csv, format_args


@torch.no_grad()
def calculate_clip_scores(all_prompts_images, metric):
    clip_scores = []

    for prompts, latents in all_prompts_images:
        # if isinstance(images, str):
        #     images = numpy.split(numpy.load(images), len(prompts))
        #
        # images = (numpy.stack(images) * 255).astype("uint8")
        # latents = torch.from_numpy(images)

        print("latents", latents.size())
        if len(latents.size()) == 3:
            latents = latents.unsqueeze(0)

        score = (
            metric(
                latents.to(metric.device),
                prompts,
            )
            .detach()
            .item()
        )

        clip_scores.append(score)

    return clip_scores


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def avg(values):
    return sum(values) / len(values)


@torch.no_grad()
def main(args):
    seed = args.seed

    set_seed(seed)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Loading SD model: {args.pretrained_model_name_or_path}")
    sd_generator = setup_sd_generator(args)

    print(f"Loading CLIP score model: {args.clip_model_name_or_path}")
    metric = CLIPScore(model_name_or_path=args.clip_model_name_or_path)

    accelerator = Accelerator()
    sd_pipeline, metric = accelerator.prepare(sd_generator.pipeline, metric)
    sd_generator.pipeline = sd_pipeline

    batch_size = args.batch_size
    clip_scores = []

    print("Processing ")
    with accelerator.autocast(), torch.inference_mode():
        if args.lora_files:
            clip_scores.extend(
                get_clip_scores_from_lora_files(
                    args.lora_files, sd_generator, metric
                )
            )

            print("---")
            print(f"CLIP Score for each LoRA ({len(clip_scores)})")
            for result in clip_scores:
                print(
                    f"CLIP score ({result['lora_file']}): {avg(result['clip_scores'])}"
                )

        else:
            clip_scores.append({
                "clip_score": avg(get_clip_scores(sd_generator, metric)),
                "model": Path(args.pretrained_model_name_or_path).name
            })

    for clip_score in clip_scores:
        print(clip_score)

    if args.csv:
        to_csv(clip_scores, args.csv)


def get_clip_scores_from_lora_files(lora_files, sd_generator, metric):
    clip_scores = []
    for file in lora_files:
        lora_scores = []
        lora_path = Path(file)

        if lora_path.is_dir():
            for lora_file in lora_path.iterdir():
                if lora_file.suffix != ".safetensors":
                    continue

                scores = get_clip_score_from_lora(
                    lora_file,
                    sd_generator,
                    metric,
                )
                if scores is not None:
                    lora_scores.append(
                        {
                            "clip_scores": clip_scores,
                            "model": Path(sd_generator.model).name,
                            "lora_file": lora_file.name,
                        }
                    )

        else:
            if lora_path.suffix != ".safetensors":
                continue
            scores = get_clip_score_from_lora(
                lora_path,
                sd_generator,
                metric,
            )
            if scores is not None:
                lora_scores.append(
                    {
                        "clip_scores": scores,
                        "model": Path(sd_generator.model).name,
                        "lora_file": lora_path.name,
                    }
                )

        metric.reset()

        print(lora_scores)

        if len(lora_scores) > 0:
            avg_scores = sum(
                [sum(score.get("clip_scores")) for score in lora_scores]
            ) / sum([len(score.get("clip_scores")) for score in lora_scores])

            print(f"CLIP score ({lora_path.name}): {avg_scores}")

            clip_scores.extend(lora_scores)
        else:
            print(f"Did not get any scores for LoRA {lora_path.name}")

    return clip_scores


def get_clip_score_from_lora(lora_file, sd_generator, metric):
    sd_generator.pipeline.unload_lora_weights()
    sd_generator.pipeline.load_lora_weights(
        lora_file, weight_name=lora_file.name
    )

    clip_scores = get_clip_scores(sd_generator, metric)

    return clip_scores


def get_clip_scores(sd_generator, metric):
    prompts_latents = generate_latents(sd_generator)

    scores = calculate_clip_scores(prompts_latents, metric)

    return scores


@torch.no_grad()
def generate_latents(sd_generator):
    prompts = load_dataset("nateraw/parti-prompts", split="train")
    prompts = prompts.shuffle(seed=sd_generator.seed)
    sample_prompts = [
        prompts[i]["Prompt"] for i in range(args.num_images_to_score)
    ]


    prompts_images = []
    for i in range(len(sample_prompts) // sd_generator.batch_size):

        prompts = sample_prompts[
            i * sd_generator.batch_size : i * sd_generator.batch_size
            + sd_generator.batch_size
        ]
        
        print("Batch prompts")
        for prompt in prompts:
            print(f"\t{prompt}")

        generated_images = sd_generator.pipeline(
            prompts,
            num_images_per_prompt=1,
            num_inference_steps=15,
            output_type="pt",
            generator=torch.manual_seed(sd_generator.seed),
        ).images

        prompts_images.append((prompts, generated_images))

    return prompts_images


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--num_images_to_score",
        type=int,
        default=25,
        help="Number of images to score if generating from the model",
    )
    # argparser.add_argument(
    #     "--save_scores_from_generated",
    #     action="store_true",
    #     help="Save the score from generated images",
    # )

    argparser.add_argument(
        "--output_clip_score",
        help="Output the CLIP score to this file or directory as clip_score.csv",
    )
    argparser.add_argument(
        "--clip_model_name_or_path",
        default="openai/clip-vit-base-patch16",
        help="CLIP Model to get the CLIP score from",
    )

    argparser = format_args(argparser)
    argparser = sd_arguments(argparser)
    args = argparser.parse_args()

    main(args)
