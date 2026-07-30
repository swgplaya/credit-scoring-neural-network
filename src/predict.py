from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.model import CreditRiskModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pt"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"


class CreditRiskPredictor:
    """Загружает модель и выполняет прогноз для новых клиентов."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")

        self.preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=True,
        )

        self.default_threshold = checkpoint.get(
            "classification_threshold",
            0.50,
        )

        self.model = CreditRiskModel(
            input_size=checkpoint["input_size"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_probability(
        self,
        client_data: pd.DataFrame,
    ) -> float:
        """
        Возвращает вероятность дефолта для одной строки данных.
        """

        if len(client_data) != 1:
            raise ValueError(
                "Для единичного прогноза должна быть передана "
                "ровно одна строка."
            )

        processed_data = self.preprocessor.transform(
            client_data
        ).astype(np.float32)

        features_tensor = torch.from_numpy(
            processed_data
        ).to(self.device)

        logits = self.model(features_tensor)
        probability = torch.sigmoid(logits)

        return float(probability.item())

    def predict_class(
        self,
        client_data: pd.DataFrame,
        threshold: float | None = None,
    ) -> int:
        """Возвращает класс 0 или 1."""

        if threshold is None:
            threshold = self.default_threshold

        probability = self.predict_probability(
            client_data
        )

        return int(probability >= threshold)