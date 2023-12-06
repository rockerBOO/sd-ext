import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipe = AutoPipelineForText2Image.from_pretrained(
    "warp-ai/wuerstchen", torch_dtype=torch.float16
).to("cuda")

caption = "Anthropomorphic cat dressed as a fire fighter"
for i, img in enumerate(
    pipe(
        caption,
        width=1024,
        height=1536,
        prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
        prior_guidance_scale=4.0,
        num_images_per_prompt=2,
    ).images
):
    img.save(f"{i}-img.png")
