import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path

import torch
from segmoe import SegMoEPipeline


def main(args):
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = SegMoEPipeline(args.config, device=device)

    if args.vae_slicing:
        pipeline.pipe.enable_vae_slicing()

    if args.vae_tiling:
        pipeline.pipe.enable_vae_tiling()

    prompt = args.prompt
    negative_prompt = args.neg_prompt
    imgs = pipeline(
        prompt=[prompt] * args.batch_size,
        negative_prompt=[negative_prompt] * args.batch_size,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
    ).images

    output_dir = Path(args.output_dir)

    for i, img in enumerate(imgs):
        img.save(output_dir / f"image-{i}.png")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""
        SegMoE: Segmind Mixture of Diffusion Experts

        python segmoe_run.py --prompt "cosmic canvas, orange city background, painting of a chubby cat" \
          --neg_prompt "nsfw, bad quality, worse quality" \
          --width 1024 \
          --height 1024 \
          --cfg 7.5 \
          --batch_size 1 \
          --steps 25 \
          --config "segmind/SegMoE-SD-4x2-v0"
        """,
        formatter_class=RawTextHelpFormatter,
    )

    argparser.add_argument(
        "--vae_slicing",
        type=bool,
        default=True,
        help="sliced VAE decode that decodes the batch latents one image at a time",
    )
    argparser.add_argument(
        "--vae_tiling",
        type=bool,
        default=True,
        help="splits the image into overlapping tiles, decodes the tiles, and blends the outputs to make the final image",
    )

    argparser.add_argument(
        "--prompt",
        default="cosmic canvas, orange city background, painting of a chubby cat",
    )
    argparser.add_argument(
        "--neg_prompt", default="nsfw, bad quality, worse quality"
    )
    argparser.add_argument("--width", type=int)
    argparser.add_argument("--height", type=int)
    argparser.add_argument("--steps", type=int, default=25)
    argparser.add_argument("--cfg", type=float, default=7.5)
    argparser.add_argument("--batch_size", type=int)
    argparser.add_argument("--output_dir", default="./")
    argparser.add_argument(
        "--config",
        default="segmind/SegMoE-SD-4x2-v0",
        help="""    segmind/SegMoE-2x1-v0
    segmind/SegMoE-4x2-v0
    segmind/SegMoE-sd-4x2-v0
    or path to config.yaml (see https://github.com/segmind/segmoe for examples)
    """,
    )

    args = argparser.parse_args()

    main(args)
