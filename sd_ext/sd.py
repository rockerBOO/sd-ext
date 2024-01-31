# from .files import get_files
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from pathlib import Path
import torch
from datasets import Dataset
from typing import List
from .random import set_seed


class SDGenerator:
    def __init__(
        self, pipeline, seed, batch_size, device, steps=50, n_iter=1, model=""
    ):
        self.pipeline = pipeline
        self.seed = seed
        self.batch_size = batch_size
        self.device = device
        self.steps = steps
        self.n_iter = n_iter
        self.model = model

    def pipe(self, *args, **kwargs):
        if len(args) > 0:
            prompt = args[0]
            prompt = (
                prompt
                if isinstance(prompt, list)
                else ([prompt] * self.n_iter)
            )
            args = prompt
        else:
            prompt = kwargs["prompt"]
            prompt = (
                prompt
                if isinstance(prompt, list)
                else ([prompt] * self.n_iter)
            )
            kwargs["prompt"] = prompt

        return self.pipeline(
            *args,
            num_image_per_prompt=self.batch_size,
            num_inference_steps=self.steps,
            **kwargs,
        )


def generate_images(sd_generator: SDGenerator, prompts: List[str]):
    batch_size = sd_generator.batch_size
    images = []
    for i in range(len(prompts) // batch_size):
        prompts = prompts[i * batch_size : i * batch_size + batch_size]

        generated_images = sd_generator.pipe(
            prompts,
            num_images_per_prompt=1,
            num_inference_steps=15,
        ).images

        images.append((prompts, generated_images))

    return images


def setup_sd_generator(args):
    model_ckpt = args.pretrained_model_name_or_path

    pipeline_kwargs = {}

    if args.vae is not None:
        vae_path = Path(args.vae)
        if vae_path.is_file():
            pipeline_kwargs["vae"] = AutoencoderKL.from_single_file(vae_path)
        else:
            pipeline_kwargs["vae"] = AutoencoderKL.from_pretrained(args.vae)

    if (
        model_ckpt.endswith(".safetensors")
        or model_ckpt.endswith(".bin")
        or model_ckpt.endswith(".pt")
        or model_ckpt.endswith(".ckpt")
    ):
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )

        sd_pipeline = StableDiffusionPipeline.from_single_file(
            model_ckpt,
            torch_dtype=torch.float16,
            load_safety_checker=False,
            use_safetensors=True,
            scheduler=scheduler,
            **pipeline_kwargs,
        )
    else:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_ckpt, subfolder="scheduler"
        )

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

    # if args.xformers is None:
    #     sd_pipeline.unet.set_attn_processor(AttnProcessor2_0())
    # sd_pipeline.unet = torch.compile(
    #     sd_pipeline.unet, mode="reduce-overhead", fullgraph=True
    # )
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        set_seed(args.seed)

    sd_pipeline.to(args.device)

    return SDGenerator(
        sd_pipeline,
        args.seed,
        args.batch_size,
        args.device,
        steps=args.steps,
        model=args.pretrained_model_name_or_path,
    )


def generate_dataset(sd_generator, prompts=None):
    images = generate_images(sd_generator, prompts)
    dataset = Dataset.from_list(images)

    return dataset


def sd_arguments(argparser):
    argparser.add_argument(
        "--pretrained_model_name_or_path",
        default="runwayml/stable-diffusion-v1-5",
        help="Model to load",
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
        "--steps", default=15, help="Number of steps to do for inference"
    )

    argparser.add_argument(
        "--n_iter", default=1, help="Number of iterations to run "
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

    argparser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed to use for random number generation",
    )

    argparser.add_argument(
        "--device", help="Seed to use for random number generation"
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the image generation in Stable Diffusion",
    )

    return argparser
