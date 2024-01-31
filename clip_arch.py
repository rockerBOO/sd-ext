# WIP print architecture from CLIP model

from dataclasses import dataclass

import open_clip

from sd_ext.torch import get_device

device = get_device(None)


@dataclass
class Args:
    clip_model = "ViT-L/14"
    pretrained_clip = "openai"


args = Args()

print(f"Loading {args.pretrained_clip} CLIP {args.clip_model}...")
clip_model, _, image_processor = open_clip.create_model_and_transforms(
    "ViT-L/14", pretrained="openai"
)
clip_model.to(device)

print(clip_model)
