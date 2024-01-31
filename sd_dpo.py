from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
import argparse
from sd_ext.sd import setup_sd_generator, sd_arguments


def main(args):
    # load pipeline
    generator = setup_sd_generator(args)
    # model_id = "runwayml/stable-diffusion-v1-5"
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     model_id, torch_dtype=torch.float16
    # )

    # load finetuned model
    unet_id = "mhdang/dpo-sd1.5-text2image-v1"
    unet = UNet2DConditionModel.from_pretrained(
        unet_id, subfolder="unet", torch_dtype=torch.float16
    )
    unet.to(args.device)
    generator.pipeline.unet = unet
    # pipe = generator.pipe.to("cuda")

    # prompt = "Two cats playing chess on a tree branch"
    prompt = args.prompt 
    images = generator.pipe(prompt, guidance_scale=7.5).images

    for i, image in enumerate(images):
        image.save(f"{prompt}_{i}.png")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--prompt", help="Prompt")
    argparser = sd_arguments(argparser)
    args = argparser.parse_args()
    main(args)
