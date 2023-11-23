from torchmetrics.multimodal.clip_score import CLIPScore
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from accelerate import Accelerator
import random
from pathlib import Path
import argparse
import numpy
from diffusers.models.attention_processor import AttnProcessor2_0


@torch.no_grad()
def calculate_clip_score(all_prompts_images, accelerator, metric):
    # batch_size = 5
    clip_scores = []

    with accelerator.autocast():
        for prompts, images in all_prompts_images:
            if isinstance(images, str):
                images = numpy.split(numpy.load(images), len(prompts))

            images = (numpy.stack(images) * 255).astype("uint8")
            latents = torch.from_numpy(images)

            if len(latents.size()) == 3:
                latents = latents.unsqueeze(0)
            clip_scores.append(
                metric(
                    latents.permute(0, 3, 1, 2).to(metric.device),
                    prompts,
                )
                .detach()
                .item()
            )

    return numpy.sum(clip_scores) / len(clip_scores)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_sd_pipeline(args):
    model_ckpt = args.pretrained_model_name_or_path

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_ckpt, subfolder="scheduler"
    )

    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        model_ckpt,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True,
        scheduler=scheduler,
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


@torch.no_grad()
def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    sd_pipeline = setup_sd_pipeline(args).to(device)

    metric = CLIPScore(model_name_or_path=args.clip_model_name_or_path)

    accelerator = Accelerator()

    sd_pipeline, metric = accelerator.prepare(sd_pipeline, metric)

    batch_size = args.batch_size
    clip_scores = []
    with accelerator.autocast(), torch.inference_mode():
        for file in args.lora_files:
            lora_path = Path(file)

            if lora_path.is_dir():
                for lora_file in lora_path.iterdir():
                    clip_score = get_clip_score_from_lora(
                        lora_file, sd_pipeline, accelerator, batch_size, seed, metric
                    )
                    clip_scores.append((clip_score, lora_file))
            else:
                clip_score = get_clip_score_from_lora(
                    lora_path, sd_pipeline, accelerator, batch_size, seed, metric
                )
                clip_scores.append((clip_score, lora_path))

    print("---")
    print(f"CLIP Score for each LoRA ({len(clip_scores)})")
    for clip_score, lora_file in clip_scores:
        print(f"CLIP score ({lora_file.name}): {clip_score}")
        # CLIP score: 35.7038


def get_clip_score_from_lora(
    lora_file, sd_pipeline, accelerator, batch_size, seed, metric
):
    sd_pipeline.unload_lora_weights()
    accelerator.clear()
    sd_pipeline.load_lora_weights(lora_file, weight_name=lora_file.name)

    prompts_latents = generate_latents(sd_pipeline, batch_size, seed)

    accelerator.free_memory()

    clip_score = calculate_clip_score(prompts_latents, accelerator, metric)

    metric.reset()
    print(f"CLIP score ({lora_file.name}): {clip_score}")

    return clip_score


@torch.no_grad()
def generate_latents(sd_pipeline, batch_size, seed):
    prompts = load_dataset("nateraw/parti-prompts", split="train")
    prompts = prompts.shuffle(seed=seed)
    sample_prompts = [prompts[i]["Prompt"] for i in range(10)]

    prompts_images = []
    for i in range(len(sample_prompts) // batch_size):
        prompts = sample_prompts[i * batch_size : i * batch_size + batch_size]

        print("Prompts")
        for sample_prompt in sample_prompts:
            print(f"\t{sample_prompt}")

        generated_images = sd_pipeline(
            prompts,
            num_images_per_prompt=1,
            num_inference_steps=15,
            output_type="np",
            generator=torch.manual_seed(seed),
        ).images

        if args.save_latents_to_disk:
            latents_file = Path(f"{args.seed}-{i}-latents.npy")
            numpy.save(latents_file, numpy.concatenate(generated_images))
            prompts_images.append((prompts, str(latents_file)))
        else:
            prompts_images.append((prompts, generated_images))

    return prompts_images


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--seed", type=int, default=1234, help="Seed for random and torch"
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the image generation in Stable Diffusion",
    )

    argparser.add_argument(
        "--pretrained_model_name_or_path",
        default="runwayml/stable-diffusion-v1-5",
        help="Model to load",
    )
    argparser.add_argument(
        "--clip_model_name_or_path",
        default="openai/clip-vit-base-patch16",
        help="CLIP Model to get the CLIP score from",
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
        "--save_latents_to_disk",
        action="store_true",
        help="",
    )
    argparser.add_argument("--xformers", action="store_true", help="Use XFormers")
    argparser.add_argument("--device", default=None, help="Set device to use")

    args = argparser.parse_args()

    main(args)
