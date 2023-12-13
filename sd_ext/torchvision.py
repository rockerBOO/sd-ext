from torchvision import transforms
from PIL import Image

BASE_IMAGE_TO_TENSOR = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def image_file_to_3_channel_tensor(image_file_path):
    return BASE_IMAGE_TO_TENSOR(Image.open(image_file_path).convert("RGB"))


def image_to_tensor(image):
    return BASE_IMAGE_TO_TENSOR(image)


