import torch
from PIL import Image

# ConvNext-Base
# ConvNext-Large
# ConvNext-XXLarge
# ViT-B/32
# ViT-B/16
# ViT-L/14
# ViT-H/14
# ViT-L/14
# ViT-G/14

# CLIP/OpenCLIP
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

VISION_TRANSFORMER_MODELS = [
    "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-H/14", "ViT-G/14"
]


def get_image_latents(image_processor, image):
    """
    Get the CLIP embeddings of the image
    """
    return image_processor(image).unsqueeze(0)


@torch.no_grad()
def get_image_features_from_image(
    image_processor, clip_model, image: Image, device
):
    """
    Get the CLIP embeddings of the image
    """

    return get_image_features_from_latents(
        clip_model, get_image_latents(image_processor).to(device)
    )


@torch.no_grad()
def get_image_features_from_latents(
    clip_model, image: torch.Tensor, device
):
    """
    Get the CLIP embeddings of the image as a tensor
    """
    image_features = clip_model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features


@torch.no_grad()
def get_image_features(image_processor, clip_model, image, device):
    """
    Get the CLIP embeddings of the image
    """
    image = image_processor(image).unsqueeze(0).to(device)
    image_features = clip_model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features


@torch.no_grad()
def get_text_features(processor, clip_model, image, texts, device):
    text_processed = processor(text=texts, return_tensors="pt", padding=True)

    clip_text_features = clip_model.get_text_features(
        text_processed["input_ids"].to(device),
        text_processed["attention_mask"].to(device),
    )
    return clip_text_features / clip_text_features.norm(
        p=2, dim=-1, keepdim=True
    )


def compare_image_text_features(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
):
    logits_per_image = img_features @ text_features
    features = logits_per_image.reshape(
        logits_per_image.shape[0], -1, 2
    ).softmax(-1)[:, :, 0]
    return features
