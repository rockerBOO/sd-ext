import argparse
from pathlib import Path

import open_clip
import torch
from PIL import Image
from platformdirs import user_cache_dir
from sd_ext.files import get_files
from sd_ext.cache import ensure_model
from sd_ext.format import to_csv, format_args
from sd_ext.aesthetic import load_model, AestheticScorer


APP_NAME = "ae-score"
APP_AUTHOR = "rockerBOO"

model_to_host = {
    "chadscorer": "https://github.com/grexzen/SD-Chad/raw/main/chadscorer.pth",
    "sac-logos": "https://github.com/christophschuhmann/"
    + "improved-aesthetic-predictor/blob/main/sac+logos+ava1-l14-linearMSE.pth"
    + "?raw=true",
}

clip_models = ["ViT-B/32", "ViT-B/32", "ViT-L/14", "ViT-L/14@336px"]

MODEL = "chadscorer"
CLIP_MODEL = "ViT-L/14"

assert CLIP_MODEL in clip_models
assert MODEL in model_to_host.keys()


def get_model(model) -> Path:
    # url = model_to_host[model]

    if model in model_to_host and (
        model_to_host[model].startswith("http")
        or model_to_host[model].startswith("https")
    ):
        model_file = Path(model_to_host[model]).name
        ensure_model(model_file, model_to_host[model])
        cache_dir = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
        model_file = cache_dir / model_file
    else:
        model_file = Path(model)

    assert model_file.exists()

    return model_file


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP {args.clip_model}...")
    # clip_model, image_processor = clip.load(args.clip_model)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model.to(device)

    print(f"Loading {args.model}...")
    predictor = load_model(get_model(args.model), args.clip_model)
    predictor.to(device)

    aesthetic_scorer = AestheticScorer(predictor, model, preprocess, device)

    files = get_files(
        args.image_file_or_dir,
        file_ext=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"],
    )

    print(f"Files to score: {len(files)}")

    scores = []
    for file in files:
        with Image.open(file) as image:
            scores.append(
                {"file": file, "score": aesthetic_scorer.score(image)}
            )

    scores = sorted(scores, key=lambda x: x["score"], reverse=True)

    for i, score in enumerate(scores):
        for key in score.keys():
            if key in ["file"]:
                scores[i][key] = str(score[key])

    if args.verbose:
        for score in scores:
            print(score["file"], score["score"])

    if args.csv:
        to_csv(scores, args.csv)
        print(f"Saved to: {args.csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_file_or_dir",
        type=str,
        help="Image file or directory containing the images.",
    )

    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Aesthetic predictor model: {MODEL}. "
        + f"Options: {', '.join(model_to_host.keys())}. Or pass a model path",
    )

    parser.add_argument(
        "--clip_model",
        default=CLIP_MODEL,
        help=f"CLIP model. Options: {', '.join(clip_models)}.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output the similarity between the images",
    )

    parser = format_args(parser)
    args = parser.parse_args()

    main(args)
