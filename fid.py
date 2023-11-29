# Copyright © 2023 Dave Lage
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset, Image, load_dataset
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from matplotlib import pyplot as plt

from torchmetrics.image.fid import FrechetInceptionDistance

# from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchvision import transforms


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_fake_images(sd_pipeline, sample_prompts, args):
    g = torch.Generator()
    g.manual_seed(args.seed)

    print("Sample prompts:")
    for sample_prompt in sample_prompts:
        print(f"\t{sample_prompt}")

    image_filenames = []
    batch_size = args.batch_size or 4

    for e, i in enumerate(range(len(sample_prompts) // batch_size)):
        images = sd_pipeline(
            sample_prompts[i * batch_size : i * batch_size + batch_size],
            # sample_prompts,
            num_images_per_prompt=1,
            num_inference_steps=args.num_inference_steps or 15,
            # output_type="np",
            generator=g,
        ).images

        for pi, (image, prompt) in enumerate(
            zip(
                images,
                sample_prompts[i * batch_size : i * batch_size + batch_size],
            )
        ):
            cleaned_prompt = prompt.replace(" ", "-")
            filename = f"./tmp/{cleaned_prompt}-{e+i+pi}.png"
            image.save(filename)
            image_filenames.append({"image": filename})

    # TODO we should save these images to the disk

    return image_filenames


def setup_sd_pipeline(args):
    model_ckpt = args.pretrained_model_name_or_path

    print("Using sampler scheduler: DPMSolverMultistepScheduler")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_ckpt, subfolder="scheduler"
    )

    print(f"Using SD model: {model_ckpt}")

    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        model_ckpt,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True,
        scheduler=scheduler,
    )

    if args.xformers:
        print("Using XFormers")
        sd_pipeline.enable_xformers_memory_efficient_attention()

    if args.ti_embedding_file is not None:
        ti_embedding_file = Path(args.ti_embedding_file)

        print(f"Using TI Embedding: {ti_embedding_file.name}")
        sd_pipeline.load_textual_inversion(
            args.ti_embedding_file, weight_name=ti_embedding_file.name
        )

    return sd_pipeline


TRANSFORMS = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(299, 299), antialias=True),
        transforms.PILToTensor(),
    ]
)


def collate_fn(data):
    return torch.stack([TRANSFORMS(d["image"]) for d in data])


def filter_invalid_images(d):
    # only supporting RGB images
    return d["image"].mode == "RGB"


def process_batch(batch, fid_model, real):
    fid_model.update(batch.to("cuda"), real=real)


def load_fake_dir(fake_data_dir, args):
    g = torch.Generator()
    g.manual_seed(args.seed)

    fake_ds = load_dataset(
        "imagefolder", data_dir=fake_data_dir, split="train"
    ).filter(filter_invalid_images)

    fake_dataloader = torch.utils.data.DataLoader(
        fake_ds["train"],
        batch_size=2,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )

    return fake_ds, fake_dataloader


def main(args):
    seed = args.seed
    torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    metric = FrechetInceptionDistance(feature=args.fid_feature)
    # metric = MemorizationInformedFrechetInceptionDistance(feature=args.fid_feature)
    metric.to(device)

    # features = Features({"image": Image()})
    real_ds = load_dataset(
        "imagefolder", data_dir=args.real_data_dir, split="train"
    ).filter(filter_invalid_images)

    real_dataloader = torch.utils.data.DataLoader(
        real_ds,
        batch_size=2,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )

    fakes = []
    if args.fake_data_dir:
        fake_ds, fake_dataloader = load_fake_dir(args.fake_data_dir)
        fakes.append((fake_ds, fake_dataloader, None))
    else:
        sd_pipeline = setup_sd_pipeline(args).to(device)

        for file in args.lora_files:
            lora_path = Path(file)

            if lora_path.is_dir():
                for lora_file in lora_path.iterdir():
                    if (
                        lora_file.is_file() is False
                        or lora_file.suffix != ".safetensors"
                    ):
                        continue

                    fake_ds, fake_dataloader = generate_fake_dataset(
                        sd_pipeline, lora_file, device, seed, len(real_ds), g
                    )
                    if fake_ds is not None:
                        fakes.append((fake_ds, fake_dataloader, lora_file))
            else:
                lora_file = lora_path
                fake_ds, fake_dataloader = generate_fake_dataset(
                    sd_pipeline, lora_file, device, seed, len(real_ds), g
                )
                fakes.append((fake_ds, fake_dataloader, lora_file))

    for fake_ds, fake_dataloader, lora_file in fakes:
        if fake_ds is None:
            print("Could not find fake dataset... continuing")
            continue

        print(f"Real: {len(real_ds)} batches: {len(real_dataloader)}")
        print(f"Fake: {len(fake_ds)} batches: {len(fake_dataloader)}")

        fid = calc_fid(real_dataloader, fake_dataloader, metric)

        lora_name = Path(lora_file).name
        if args.save_plot:
            metric.plot(fid)

        print(f"FID ({lora_name}): {fid.item()}")

    plt.savefig("fid.png")


def generate_fake_dataset(sd_pipeline, lora_file, device, seed, num_images, g):
    # lora_file = Path(lora_file)
    print(f"Using LoRA: {lora_file.name}")
    sd_pipeline.unload_lora_weights()
    try:
        sd_pipeline.load_lora_weights(lora_file, weight_name=lora_file.name)

        # AttributeError: 'ModuleList' object has no attribute 'time'
    except AttributeError as e:
        print(e)
        print(f"Could not load lora weights for {lora_file.name}")
        return None, None

    prompts = load_dataset("nateraw/parti-prompts", split="train")
    prompts = prompts.shuffle(seed=seed)
    sample_prompts = [prompts[i]["Prompt"] for i in range(num_images)]

    images = generate_fake_images(sd_pipeline, sample_prompts, args)

    fake_ds = Dataset.from_list(images)

    if args.save_fake_images:
        # Load the images and cast them as a Image
        fake_ds = fake_ds.cast_column("image", Image())

    fake_dataloader = torch.utils.data.DataLoader(
        fake_ds,
        batch_size=2,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )

    return fake_ds, fake_dataloader


def calc_fid(real_dataloader, fake_dataloader, metric):
    for i, images in enumerate(real_dataloader):
        process_batch(images, metric, real=True)

    for i, images in enumerate(fake_dataloader):
        process_batch(images, metric, real=False)

    metric.set_dtype(torch.float64)
    fid = metric.compute()

    metric.reset()

    return fid


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--seed", type=int, default=1234, help="Seed for random and torch"
    )
    argparser.add_argument(
        "--pretrained_model_name_or_path",
        default="runwayml/stable-diffusion-v1-5",
        help="Model to load",
    )

    argparser.add_argument(
        "--lora_files",
        default=None,
        nargs="+",
        help="Lora model file or files to load.",
    )

    argparser.add_argument(
        "--ti_embedding_file",
        default=None,
        help="Textual inversion file to load",
    )

    argparser.add_argument(
        "--fake_data_dir",
        default=None,
        help="Fake data dir with SD generated images",
    )
    argparser.add_argument(
        "--real_data_dir",
        required=True,
        help="Real images (non AI generated) data dir. "
        + "Probably your training or validation images",
    )

    argparser.add_argument(
        "--num_inference_steps",
        default=15,
        help="Number of inference steps for creating fake images",
    )

    argparser.add_argument(
        "--xformers", action="store_true", help="Use XFormers"
    )
    argparser.add_argument("--device", default=None, help="Set device to use")

    argparser.add_argument(
        "--save_fake_images",
        action="store_true",
        help="Should we save the fake image samples if we are creating them",
    )

    argparser.add_argument(
        "--save_fake_images_dir",
        default="./tmp",
        help="Where should we save the fake images to?",
    )

    argparser.add_argument(
        "--batch_size",
        default=None,
        help="Batch size for creating fake SD images",
    )
    argparser.add_argument(
        "--save_plot",
        default=None,
        help="Save an image with the plot for FID. Saves to fid.png",
    )

    argparser.add_argument(
        "--fid_feature",
        type=int,
        choices=[64, 192, 768, 2048],
        default=2048,
        help="an integer will indicate the inceptionv3 feature layer to choose."
        + " Can be one of the following: 64, 192, 768, 2048",
    )

    args = argparser.parse_args()

    main(args)
