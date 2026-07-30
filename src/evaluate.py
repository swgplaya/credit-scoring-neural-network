import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data import (
    TARGET_COLUMN,
    build_preprocessor,
    load_clean_data,
    split_data,
)
from src.model import CreditRiskModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = ARTIFACTS_DIR / "model.pt"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

RANDOM_STATE = 42


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
        "threshold": threshold,
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


def load_neural_network(
    device: torch.device,
) -> tuple[CreditRiskModel, float]:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = CreditRiskModel(
        input_size=checkpoint["input_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    threshold = checkpoint.get(
        "classification_threshold",
        0.5,
    )

    return model, threshold


@torch.no_grad()
def predict_neural_network(
    model: CreditRiskModel,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    features_tensor = torch.from_numpy(
        features.astype(np.float32)
    ).to(device)

    logits = model(features_tensor)
    probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


def find_threshold_for_recall(
    targets: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float = 0.70,
) -> dict[str, float]:
    """
    Среди порогов с recall не ниже заданного
    выбирает порог с максимальным precision.
    """

    candidates = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.01,
    ):
        metrics = calculate_metrics(
            targets,
            probabilities,
            threshold=float(threshold),
        )

        if metrics["recall"] >= minimum_recall:
            candidates.append(metrics)

    if not candidates:
        raise RuntimeError(
            "Не найден порог с требуемым recall."
        )

    return max(
        candidates,
        key=lambda item: item["precision"],
    )


def print_metrics(
    model_name: str,
    metrics: dict[str, float],
) -> None:
    print(f"\n{model_name}")

    print(
        f"  Порог:     {metrics['threshold']:.2f}"
    )
    print(
        f"  Accuracy:  {metrics['accuracy']:.4f}"
    )
    print(
        f"  Precision: {metrics['precision']:.4f}"
    )
    print(
        f"  Recall:    {metrics['recall']:.4f}"
    )
    print(
        f"  ROC-AUC:   {metrics['roc_auc']:.4f}"
    )


def save_confusion_matrix(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    filename: str,
    title: str,
) -> None:
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    matrix = confusion_matrix(
        targets,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No default",
            "Default",
        ],
    )

    display.plot(
        values_format="d",
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / filename,
        dpi=150,
    )
    plt.close()


def save_roc_curve(
    targets: np.ndarray,
    neural_probabilities: np.ndarray,
    logistic_probabilities: np.ndarray,
) -> None:
    RocCurveDisplay.from_predictions(
        targets,
        neural_probabilities,
        name="Neural network",
    )

    RocCurveDisplay.from_predictions(
        targets,
        logistic_probabilities,
        name="Logistic regression",
        ax=plt.gca(),
    )

    plt.title("ROC curve comparison")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "roc_curve.png",
        dpi=150,
    )
    plt.close()


def save_precision_recall_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    PrecisionRecallDisplay.from_predictions(
        targets,
        probabilities,
        name="Neural network",
    )

    plt.title("Precision-recall curve")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "precision_recall_curve.png",
        dpi=150,
    )
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_clean_data()

    splits = split_data(
        data,
        random_state=RANDOM_STATE,
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    X_val = preprocessor.transform(
        splits.X_val
    ).astype(np.float32)

    X_test = preprocessor.transform(
        splits.X_test
    ).astype(np.float32)

    y_val = splits.y_val.to_numpy()
    y_test = splits.y_test.to_numpy()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    neural_model, default_threshold = (
        load_neural_network(device)
    )

    neural_val_probabilities = (
        predict_neural_network(
            neural_model,
            X_val,
            device,
        )
    )

    neural_test_probabilities = (
        predict_neural_network(
            neural_model,
            X_test,
            device,
        )
    )

    default_metrics = calculate_metrics(
        y_test,
        neural_test_probabilities,
        threshold=default_threshold,
    )

    selected_threshold_metrics = (
        find_threshold_for_recall(
            y_val,
            neural_val_probabilities,
            minimum_recall=0.70,
        )
    )

    selected_threshold = (
        selected_threshold_metrics["threshold"]
    )

    tuned_test_metrics = calculate_metrics(
        y_test,
        neural_test_probabilities,
        threshold=selected_threshold,
    )

    logistic_model = Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                splits.X_train
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    logistic_model.fit(
        splits.X_train,
        splits.y_train,
    )

    logistic_test_probabilities = (
        logistic_model.predict_proba(
            splits.X_test
        )[:, 1]
    )

    logistic_metrics = calculate_metrics(
        y_test,
        logistic_test_probabilities,
        threshold=0.5,
    )

    print("=" * 70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)

    print_metrics(
        "Нейросеть, порог 0.50",
        default_metrics,
    )

    print_metrics(
        "Нейросеть, подобранный порог",
        tuned_test_metrics,
    )

    print_metrics(
        "Логистическая регрессия",
        logistic_metrics,
    )

    save_confusion_matrix(
        y_test,
        neural_test_probabilities,
        threshold=default_threshold,
        filename="confusion_matrix_threshold_050.png",
        title="Neural network, threshold 0.50",
    )

    save_confusion_matrix(
        y_test,
        neural_test_probabilities,
        threshold=selected_threshold,
        filename="confusion_matrix_tuned_threshold.png",
        title=(
            "Neural network, "
            f"threshold {selected_threshold:.2f}"
        ),
    )

    save_roc_curve(
        y_test,
        neural_test_probabilities,
        logistic_test_probabilities,
    )

    save_precision_recall_curve(
        y_test,
        neural_test_probabilities,
    )

    results = {
        "neural_network_default_threshold": (
            default_metrics
        ),
        "neural_network_tuned_threshold": (
            tuned_test_metrics
        ),
        "logistic_regression": logistic_metrics,
    }

    with (
        REPORTS_DIR / "evaluation_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    joblib.dump(
        logistic_model,
        ARTIFACTS_DIR
        / "logistic_regression.joblib",
    )

    print("\nГрафики и результаты сохранены:")
    print("  - reports/roc_curve.png")
    print("  - reports/precision_recall_curve.png")
    print(
        "  - reports/"
        "confusion_matrix_threshold_050.png"
    )
    print(
        "  - reports/"
        "confusion_matrix_tuned_threshold.png"
    )
    print("  - reports/evaluation_results.json")


if __name__ == "__main__":
    main()