from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    DPMSolverMultistepScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
import torch

model_id = "stabilityai/sdxl-turbo"

scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
    scheduler=scheduler,
)
print(pipe)
pipe = pipe.to("cuda")

pipe.unet.set_attn_processor(AttnProcessor2_0())
pipe.enable_model_cpu_offload()
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = [
    "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    "a picture of a nude woman with her legs spread on a couch",
]

for i, img in enumerate(
    pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=1).images
):
    img.save(f"{i}-img.png")
