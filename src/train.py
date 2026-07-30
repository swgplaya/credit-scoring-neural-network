import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    build_preprocessor,
    load_clean_data,
    split_data,
)
from src.model import CreditRiskModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
BATCH_SIZE = 256
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
CLASSIFICATION_THRESHOLD = 0.5


def set_random_seed(seed: int) -> None:
    """Фиксирует генераторы случайных чисел."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_tensor_dataset(
    features: np.ndarray,
    target,
) -> TensorDataset:
    """Преобразует NumPy-массивы в датасет PyTorch."""

    features_tensor = torch.from_numpy(
        features.astype(np.float32)
    )

    target_tensor = torch.from_numpy(
        target.to_numpy(dtype=np.float32)
    )

    return TensorDataset(
        features_tensor,
        target_tensor,
    )


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict[str, float]:
    """Вычисляет метрики бинарной классификации."""

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
        "accuracy": accuracy_score(
            targets,
            predictions,
        ),
        "precision": precision_score(
            targets,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            targets,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            targets,
            probabilities,
        ),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Оценивает модель без изменения её весов."""

    model.eval()

    total_loss = 0.0
    total_observations = 0

    all_targets = []
    all_probabilities = []

    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)

        loss = loss_function(
            logits,
            targets,
        )

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_observations += batch_size

        probabilities = torch.sigmoid(logits)

        all_targets.append(
            targets.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    targets_array = np.concatenate(all_targets)
    probabilities_array = np.concatenate(
        all_probabilities
    )

    metrics = calculate_metrics(
        targets_array,
        probabilities_array,
    )

    metrics["loss"] = (
        total_loss / total_observations
    )

    return metrics


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Выполняет одну эпоху обучения."""

    model.train()

    total_loss = 0.0
    total_observations = 0

    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)

        # Удаляем градиенты предыдущего шага.
        optimizer.zero_grad(set_to_none=True)

        # Прямой проход.
        logits = model(features)

        # Вычисляем ошибку.
        loss = loss_function(
            logits,
            targets,
        )

        # Вычисляем градиенты.
        loss.backward()

        # Обновляем веса нейросети.
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_observations += batch_size

    return total_loss / total_observations


def print_epoch_result(
    epoch: int,
    train_loss: float,
    validation_metrics: dict[str, float],
) -> None:
    print(
        f"Epoch {epoch:02d} | "
        f"train loss: {train_loss:.4f} | "
        f"val loss: {validation_metrics['loss']:.4f} | "
        f"val AUC: {validation_metrics['roc_auc']:.4f} | "
        f"precision: {validation_metrics['precision']:.4f} | "
        f"recall: {validation_metrics['recall']:.4f}"
    )


def save_artifacts(
    model: CreditRiskModel,
    preprocessor,
    feature_names: list[str],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    input_size: int,
) -> None:
    """Сохраняет модель и преобразователь данных."""

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = ARTIFACTS_DIR / "model.pt"
    preprocessor_path = (
        ARTIFACTS_DIR / "preprocessor.joblib"
    )
    feature_names_path = (
        ARTIFACTS_DIR / "feature_names.json"
    )
    metadata_path = (
        ARTIFACTS_DIR / "metadata.json"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "classification_threshold": (
                CLASSIFICATION_THRESHOLD
            ),
        },
        model_path,
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    with feature_names_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_names,
            file,
            ensure_ascii=False,
            indent=2,
        )

    metadata = {
        "input_size": input_size,
        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nАртефакты сохранены:")

    for path in [
        model_path,
        preprocessor_path,
        feature_names_path,
        metadata_path,
    ]:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    set_random_seed(RANDOM_STATE)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Устройство: {device}")

    data = load_clean_data()
    splits = split_data(
        data,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(
        splits.X_train
    )

    # Преобразователь обучается только на train.
    X_train = preprocessor.fit_transform(
        splits.X_train
    ).astype(np.float32)

    X_val = preprocessor.transform(
        splits.X_val
    ).astype(np.float32)

    X_test = preprocessor.transform(
        splits.X_test
    ).astype(np.float32)

    train_dataset = create_tensor_dataset(
        X_train,
        splits.y_train,
    )

    validation_dataset = create_tensor_dataset(
        X_val,
        splits.y_val,
    )

    test_dataset = create_tensor_dataset(
        X_test,
        splits.y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    input_size = X_train.shape[1]

    model = CreditRiskModel(
        input_size=input_size
    ).to(device)

    print(f"Количество входных признаков: {input_size}")
    print(model)

    negative_count = int(
        (splits.y_train == 0).sum()
    )

    positive_count = int(
        (splits.y_train == 1).sum()
    )

    positive_class_weight = (
        negative_count / positive_count
    )

    print(
        "Вес положительного класса:",
        f"{positive_class_weight:.4f}",
    )

    positive_weight_tensor = torch.tensor(
        positive_class_weight,
        dtype=torch.float32,
        device=device,
    )

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    best_validation_auc = float("-inf")
    best_model_state = None
    epochs_without_improvement = 0

    print("\nНачинаем обучение:\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        print_epoch_result(
            epoch=epoch,
            train_loss=train_loss,
            validation_metrics=validation_metrics,
        )

        validation_auc = validation_metrics[
            "roc_auc"
        ]

        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_model_state = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nОбучение остановлено досрочно: "
                "валидационный ROC-AUC не улучшается."
            )
            break

    if best_model_state is None:
        raise RuntimeError(
            "Не удалось сохранить состояние модели."
        )

    model.load_state_dict(best_model_state)

    validation_metrics = evaluate_model(
        model=model,
        data_loader=validation_loader,
        loss_function=loss_function,
        device=device,
    )

    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    print("\n" + "=" * 70)
    print("ЛУЧШАЯ МОДЕЛЬ")
    print("=" * 70)

    print("\nВалидационная выборка:")

    for name, value in validation_metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\nТестовая выборка:")

    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    save_artifacts(
        model=model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        input_size=input_size,
    )


if __name__ == "__main__":
    main()