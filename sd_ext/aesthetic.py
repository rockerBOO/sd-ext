import torch
from torch import nn
import torchvision

from sd_ext.files import load_file


class Print(nn.Module):
    def __init__(self, name=""):
        super(Print, self).__init__()

        self.name = name

    def forward(self, x):
        print(self.name, x.shape, x.norm())
        return x


class AestheticPredictorSwish(nn.Module):
    def __init__(self, input_size, dropout1=0.2, dropout2=0.2, dropout3=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(dropout1),
            nn.GELU(),
            nn.LayerNorm(1024),
            Print(),
            nn.Linear(1024, 512),
            nn.Dropout(dropout2),
            nn.SiLU(),
            nn.LayerNorm(512),
            Print(),
            nn.Linear(512, 128),
            nn.Dropout(dropout2),
            nn.SiLU(),
            nn.LayerNorm(128),
            Print(),
            nn.Linear(128, 64),
            nn.Dropout(dropout3),
            nn.SiLU(),
            Print(),
            nn.Linear(64, 16),
            nn.SiLU(),
            Print(),
            # nn.Linear(16, 1),
            # nn.Sigmoid(),
            Print(),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticPredictorTransformer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Transformer(nhead=8, num_encoder_layers=6)

    def forward(self, x):
        return self.layers(x)


class AestheticPredictorRelu(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class LinearDown(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(LinearDown, self).__init__()
        print(in_channels, out_channels)
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(0.2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dropout(self.linear(x))
        x = self.act(x)

        return x


class AestheticPredictorSE(nn.Module):
    def __init__(self, in_channels=768, out_channels=1024, adaptive_pool=True):
        super(AestheticPredictorSE, self).__init__()

        self.in_channels = in_channels
        self.layers = nn.Sequential(
            LinearDown(self.in_channels, out_channels),
            LinearDown(out_channels, int(out_channels / 4)),
            LinearDown(int(out_channels / 4), int(out_channels / 8)),
            LinearDown(int(out_channels / 8), int(out_channels / 16)),
            LinearDown(int(out_channels / 16), int(out_channels / 64)),
            nn.AdaptiveAvgPool1d(1)
            if adaptive_pool
            else nn.Linear(int(out_channels / 64), 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            # nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScorer:
    def __init__(self, predictor, clip_model, image_processor, device):
        self.predictor = predictor
        self.clip = clip_model
        self.image_processor = image_processor
        self.device = device

    def score(self, image):
        with torch.no_grad():
            image = self.image_processor(image).unsqueeze(0).to(self.device)
            image_features = self.clip.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            score = (
                self.predictor(image_features.to(self.device)).detach().item()
            )

        return score


def load_model(model_file: str, clip_model_name: str) -> AestheticPredictor:
    # CLIP embedding dim is 768 for CLIP ViT L 14
    if "ViT-L" in clip_model_name:
        predictor = AestheticPredictor(768)
    elif "ViT-B" in clip_model_name:
        predictor = AestheticPredictor(512)
    else:
        predictor = AestheticPredictor(768)

    state_dict, metadata = load_file(model_file)

    predictor.load_state_dict(state_dict)
    predictor.eval()

    return predictor


