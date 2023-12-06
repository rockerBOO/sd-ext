import torch
from torch import nn

from sd_ext.files import load_file


class AestheticPredictor(nn.Module):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
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

    def training_step(self, batch):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = torch.nn.functional.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = torch.nn.functional.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


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

    predictor.load_state_dict(load_file(model_file))
    predictor.eval()

    return predictor
