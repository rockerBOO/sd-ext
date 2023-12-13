import argparse

from PIL import Image
import torch.nn.functional as F
import open_clip
from sd_ext.files import get_files
from sd_ext.torch import torch_args, get_device
from sd_ext.format import to_csv, to_json, format_args
from sd_ext.clip import VISION_TRANSFORMER_MODELS, get_image_features

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".avif"]


def main(args):
    device = get_device(args.device)

    print(device)

    print(f"Loading {args.pretrained_clip} CLIP {args.clip_model}...")
    clip_model, _, image_processor = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.pretrained_clip
    )
    clip_model.to(device)

    print("Getting files list...")

    files = get_files(
        args.image_file_or_dir,
        file_ext=args.exts,
        recursive=args.recursive is True,
    )

    print("Getting comparision files list...")

    other_files = files

    if args.other_file_or_dir:
        other_files = get_files(
            args.image_file_or_dir,
            file_ext=args.exts,
            recursive=args.recursive is True,
        )

    if args.filter:
        files = [file for file in files if (args.filter in str(file)) is False]
        other_files = [
            file for file in files if (args.filter in str(file)) is False
        ]

    print("Getting image embeddings from CLIP...")

    embeddings = []

    for file in files:
        embedding = get_image_features(
            image_processor, clip_model, Image.open(file), device
        )

        embeddings.append(embedding)

    print("Getting comparision image embeddings from CLIP...")

    other_embeddings = embeddings

    if args.other_file_or_dir:
        for file in other_files:
            other_embedding = get_image_features(
                image_processor, clip_model, Image.open(file), device
            )

            other_embeddings.append(other_embedding)

    print("Comparing image embeddings using cosine similarity...")

    # similarity tests

    similarities = []
    for file1, emb1 in zip(files, embeddings):
        for file2, emb2 in zip(other_files, other_embeddings):
            if file1 == file2:
                continue

            similarity = F.cosine_similarity(emb1, emb2)

            similarities.append(
                {
                    "file1": file1,
                    "file2": file2,
                    "similarity": similarity.cpu().item(),
                }
            )

    for similarity in similarities:
        print(
            similarity["file1"].name,
            similarity["file2"].name,
            f"{similarity['similarity']:4f}",
        )

    # conver over file names to  strings
    for i, similarity in enumerate(similarities):
        for key in similarity.keys():
            if key in ["file1", "file2"]:
                similarities[i][key] = str(similarity[key])

    if args.csv:
        to_csv(similarities, args.csv)

    if args.json:
        to_json(similarities, args.json)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Find the image similarities between an image or directory of images"
    )

    argparser.add_argument(
        "--image_file_or_dir", help="Images to use as inputs."
    )
    argparser.add_argument(
        "--other_file_or_dir", help="Image dataset to compare with."
    )
    argparser.add_argument(
        "--recursive", help="Recursively go through the directories."
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
        "--clip_model",
        choices=VISION_TRANSFORMER_MODELS,
        default="ViT-L-14",
        help="CLIP model",
    )
    argparser.add_argument(
        "--pretrained_clip",
        choices=["openai", "openclip"],
        default="openai",
        help="Pretrained model producer",
    )

    argparser = torch_args(argparser)
    argparser = format_args(argparser)

    args = argparser.parse_args()

    main(args)
