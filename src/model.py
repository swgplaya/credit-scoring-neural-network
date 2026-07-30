import torch
from torch import nn


class CreditRiskModel(nn.Module):
    """Нейросеть для оценки вероятности кредитного дефолта."""

    def __init__(self, input_size: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.20),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),

            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)

        # Было: [batch_size, 1]
        # Станет: [batch_size]
        return logits.squeeze(dim=1)