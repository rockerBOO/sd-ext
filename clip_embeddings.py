# WIP get text embeddings from images? (i think we want text embeddings from captions)

import hashlib
from pathlib import Path
import open_clip
from sd_ext.aesthetic import get_image_features
import numpy
import torch


def hashFile(file: Path):
    sha1 = hashlib.sha1()

    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    with open(file, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)


def main():
    print("Loading CLIP model...")
    clip_model, _, image_processor = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )

    images_features = []
    hashes = []

    for image in images:
        with torch.no_grad():
            images_features.append(
                get_image_features(image_processor, clip_model, image)
            )
        hashes.append(hashFile(image))

    features = torch.stack(images_features)

    numpy.save("cached-latents", torch.cpu().numpy())
    numpy.save("hashes", hashes)


main()
# md5.update(data)
# # BUF_SIZE is totally arbitrary, change for your app!
#
# md5 = hashlib.md5()
#
# with open(sys.argv[1], "rb") as f:
#     while True:
#         data = f.read(BUF_SIZE)
#         if not data:
#             break
#         md5.update(data)
#         sha1.update(data)
#
# print("MD5: {0}".format(md5.hexdigest()))
# print("SHA1: {0}".format(sha1.hexdigest()))
