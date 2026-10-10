import torch
from torch import nn

from config import Config


class PriceModel(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        target_mean: float,
        target_std: float,
        config: Config,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, width)
            for cardinality, width in zip(cardinalities, config.embedding_dims)
        )

        input_width = sum(config.embedding_dims) + len(config.continuous_features)
        self.mlp = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            nn.ReLU(),
            nn.Linear(config.hidden_width, 1),
        )

        self.target_mean = target_mean
        self.target_std = target_std

    def encode_features(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return torch.cat(embedded + [continuous], dim=1)

    def forward(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(self.encode_features(categorical, continuous)).squeeze(1)

    def predict_price(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        return self(categorical, continuous) * self.target_std + self.target_mean


class WideAndDeepModel(PriceModel):
    def __init__(
        self,
        cardinalities: list[int],
        target_mean: float,
        target_std: float,
        config: Config,
        lambda_value: float,
    ) -> None:
        super().__init__(cardinalities, target_mean, target_std, config)
        self.linear = nn.Linear(self.mlp[0].in_features, 1)
        self.lambda_value = lambda_value

    def forward(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        features = self.encode_features(categorical, continuous)
        prediction = self.mlp(features) + self.lambda_value * self.linear(features)
        return prediction.squeeze(1)
