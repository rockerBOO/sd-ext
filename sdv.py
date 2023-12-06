import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from diffusers.models.attention_processor import AttnProcessor2_0

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    variant="fp16",
)

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Load the conditioning image
# image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true"
# )
image = load_image(
    "/home/rockerboo/art/nsfw/pov-analog-skin-texture/00008-1366162927.png"
)
image = image.resize((576, 1024))
# image = image.resize((682, 384))

print(pipe.unet.device)
print(pipe.vae.device)
# pipe.to("cuda")

generator = torch.manual_seed(42)

pipe.unet.set_attn_processor(AttnProcessor2_0())

pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking()
frames = pipe(
    image,
    decode_chunk_size=2,
    generator=generator,
    motion_bucket_id=180,
    noise_aug_strength=0.1,
    num_frames=18,
).frames[0]
# frames = pipe(image, decode_chunk_size=1, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
