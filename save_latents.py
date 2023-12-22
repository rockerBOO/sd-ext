import argparse
import json
from pathlib import Path

import numpy
import open_clip
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms

from sd_ext.clip import get_image_features
from sd_ext.files import get_files, save_model
from sd_ext.hash import hash_file

# Set the directory you want to index here. "." is the current directory that you are in
IMAGE_DIR = "."
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".avif"]
IMAGE_JSON = "priv/images.json"

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def get_latent(vae, image):
    with torch.no_grad():
        latents = vae.tiled_encode(
            torch.stack([image]).to(vae.device, dtype=vae.dtype)
        ).latent_dist.sample()

    return latents[0]


def transform(image_file_path):
    return IMAGE_TRANSFORMS(Image.open(image_file_path).convert("RGB"))


def process_latents(
    vae, image_tensor, file, outdir, save_as="tensor", metadata={}
):
    latent = get_latent(vae, image_tensor)
    if save_as == "tensor":
        sha1_hash = hash_file(file)
        # save latent to disk
        save_to = (
            outdir / file.with_name(f"{file.stem}-latent.safetensors").name
        )
        print(f"Saving latent to {save_to}")
        save_model(
            {"latent": latent},
            save_to,
            metadata={
                **metadata,
                "file": file.name,
                "sha1_hash": sha1_hash,
                "note": "Latent of file",
            },
        )
    elif save_as == "numpy":
        save_to = outdir / file.with_name(f"{file.stem}-latent.npy").name
        print(f"Saving latent to {save_to}")
        numpy.save(save_to, latent.numpy())

    return latent


def process_embedding(
    image_processor,
    clip_model,
    file,
    outdir,
    device,
    save_as="tensor",
    metadata={},
):
    embedding = get_image_features(
        image_processor, clip_model, Image.open(file), device
    )

    if save_as == "tensor":
        sha1_hash = hash_file(file)
        # save latent to disk
        save_to = (
            outdir / file.with_name(f"{file.stem}-embedding.safetensors").name
        )
        print(f"Saving embedding to {save_to}")
        save_model(
            {"embedding": embedding},
            save_to,
            metadata={
                **metadata,
                "file": file.name,
                "sha1_hash": sha1_hash,
                "note": "CLIP Embedding of file",
            },
        )
    elif save_as == "numpy":
        save_to = outdir / file.with_name(f"{file.stem}-embedding.npy").name
        print(f"Saving embedding to {save_to}")
        numpy.save(save_to, embedding.numpy())

    return embedding


def main(args):
    if args.latents is False and args.embeddings is False:
        print("Nothing to process. Set --latents and/or --embeddings")
        return

    if args.compile is True and args.outdir is None:
        print("Set the directory to save the compiled output to")
        return

    device = (
        args.device
        if args.device is not None
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if args.latents:
        print("Loading VAE stabilityai/sd-vae-ft-mse...")
        vae = AutoencoderKL.from_pretrained(args.vae)
        vae.to(device)
        vae.enable_tiling()
        vae.enable_slicing()
        vae.eval()

    if args.embeddings:
        print(f"Loading {args.pretrained_clip} CLIP {args.clip_model}...")
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=args.pretrained_clip
        )
        clip_model.to(device)

    files = get_files(
        args.image_file_or_dir,
        file_ext=args.exts,
    )

    if args.filter:
        files = [file for file in files if (args.filter in str(file)) is False]

    metadata = {
        "clip_model": args.clip_model,
        "pretrained_clip_model": args.pretrained_clip,
        "vae": args.vae,
    }

    outdir = Path(args.outdir)

    latents = []
    embeddings = []

    for file in files:
        print(file)

        if args.latents:
            image_tensor = transform(file)
            latent = process_latents(
                vae,
                image_tensor,
                file,
                outdir,
                save_as=args.save_as,
                metadata=metadata,
            )

        if args.embeddings:
            embedding = process_embedding(
                preprocess,
                clip_model,
                file,
                outdir,
                device=device,
                save_as=args.save_as,
                metadata=metadata,
            )

        if args.compile:
            latents.append(latent)
            embeddings.append(embedding)

    if args.compile:
        state_dict = {}
        for i, file in enumerate(files):
            # print(f"Add to compiled {files[i].stem}")
            if isinstance(latents[i], torch.Tensor):
                state_dict[files[i].stem + "_latent"] = latents[i]

            if isinstance(embeddings[i], torch.Tensor):
                state_dict[files[i].stem + "_embedding"] = embeddings[i]

        save_compiled_as = outdir / "image_latents_embeddings.safetensors"
        print(f"Save compiled latents and embeddings as {save_compiled_as}")
        save_model(
            state_dict,
            save_compiled_as,
            metadata={
                **metadata,
                "notes": "Latent and embeddings of the image files",
                "files": json.dumps([str(file.name) for file in files]),
                "sha1_hashes": json.dumps([hash_file(file) for file in files]),
            },
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Cache VAE latents and CLIP Image embeddings to disk"
    )

    argparser.add_argument(
        "image_file_or_dir", help="Directory to get the images from"
    )
    argparser.add_argument(
        "--vae", default="stabilityai/sd-vae-ft-mse", help="VAE model to use"
    )
    argparser.add_argument(
        "--clip_model", default="ViT-L-14", help="CLIP model"
    )
    argparser.add_argument(
        "--pretrained_clip",
        choices=["openai", "openclip"],
        default="openai",
        help="Pretrained model producer",
    )
    argparser.add_argument(
        "--exts",
        nargs="+",
        default=IMAGE_EXTS,
        help=f"Extensions for images. Default [{', '.join(IMAGE_EXTS)}]",
    )

    argparser.add_argument(
        "--filter",
        help="Filter to use to exclude images. Checks if the filter is in the filename. Ex: --filter mask",
    )

    argparser.add_argument(
        "--save_as",
        choices=["tensor", "numpy"],
        default="tensor",
        help="Save the embedding and latents to disk as. tensor or numpy",
    )

    argparser.add_argument(
        "--outdir",
        default="tensor",
        help="Save the latents and/or embeddings to this location",
    )

    argparser.add_argument(
        "--compile",
        action="store_true",
        help="Compile all the latents and images together in a single file",
    )

    # argparser.add_argument(
    #     "--pair",
    #     action="store_true",
    #     help="Save latent and embedding together",
    # )
    argparser.add_argument(
        "--latents",
        action="store_true",
        default=False,
        help="Save the latents to disk",
    )
    argparser.add_argument(
        "--embeddings",
        action="store_true",
        default=False,
        help="Save the CLIP image embeddings to disk",
    )

    argparser.add_argument(
        "--device",
        default=None,
        help="Device to run the models on",
    )

    args = argparser.parse_args()
    main(args)
