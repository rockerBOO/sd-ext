import torch


def get_device(device=None):
    return (
        device
        if device is not None
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def torch_args(argparser):
    argparser.add_argument("--device", help="Device to run the model on")
    return argparser
