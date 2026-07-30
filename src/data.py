from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SOURCE_TARGET_COLUMN = "default.payment.next.month"
TARGET_COLUMN = "default"

CATEGORICAL_COLUMNS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx",
}


@dataclass
class DataSplits:
    """Хранит обучающую, валидационную и тестовую выборки."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def find_dataset() -> Path:
    """Находит единственный датасет в папке data/raw."""

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Папка с исходными данными не найдена: {RAW_DATA_DIR}"
        )

    datasets = [
        path
        for path in RAW_DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not datasets:
        raise FileNotFoundError(
            "В папке data/raw не найден CSV, XLS или XLSX-файл."
        )

    if len(datasets) > 1:
        filenames = "\n".join(
            f"  - {path.name}"
            for path in datasets
        )

        raise RuntimeError(
            "В папке data/raw найдено несколько датасетов:\n"
            f"{filenames}\n"
            "Оставь только нужный файл."
        )

    return datasets[0]


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Загружает исходный датасет."""

    if path is None:
        path = find_dataset()

    extension = path.suffix.lower()

    if extension == ".csv":
        data = pd.read_csv(path)
    elif extension in {".xls", ".xlsx"}:
        data = pd.read_excel(path)
    else:
        raise ValueError(
            f"Формат {extension} не поддерживается."
        )

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    return data


def validate_raw_data(data: pd.DataFrame) -> None:
    """Проверяет наличие обязательных столбцов."""

    required_columns = {
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        SOURCE_TARGET_COLUMN,
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "В датасете отсутствуют обязательные столбцы: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Датасет пуст.")

    if data.isna().any().any():
        raise ValueError(
            "В датасете обнаружены пропущенные значения."
        )


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Выполняет базовую очистку датасета."""

    validate_raw_data(data)

    cleaned = data.copy()

    # ID является только техническим идентификатором клиента.
    cleaned = cleaned.drop(columns="ID")

    cleaned = cleaned.rename(
        columns={
            SOURCE_TARGET_COLUMN: TARGET_COLUMN,
        }
    )

    # 0, 5 и 6 — недокументированные значения образования.
    # Объединяем их с категорией 4 — Other.
    cleaned["EDUCATION"] = cleaned["EDUCATION"].replace({
        0: 4,
        5: 4,
        6: 4,
    })

    # 0 — недокументированное семейное положение.
    # Объединяем его с категорией 3 — Other.
    cleaned["MARRIAGE"] = cleaned["MARRIAGE"].replace({
        0: 3,
    })

    target_values = set(cleaned[TARGET_COLUMN].unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "Целевая переменная должна содержать только 0 и 1. "
            f"Найдены значения: {sorted(target_values)}"
        )

    return cleaned


def load_clean_data() -> pd.DataFrame:
    """Загружает и очищает исходный датасет."""

    raw_data = load_raw_data()
    return clean_data(raw_data)


def split_data(
    data: pd.DataFrame,
    random_state: int = 42,
) -> DataSplits:
    """
    Делит данные в соотношении:

    70% — обучение;
    15% — валидация;
    15% — тестирование.
    """

    X = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Создаёт преобразователь признаков.

    Числовые признаки стандартизируются.
    Категориальные признаки кодируются методом one-hot.
    """

    numerical_columns = [
        column
        for column in X_train.columns
        if column not in CATEGORICAL_COLUMNS
    ]

    numerical_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "one_hot_encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_COLUMNS,
        ),
    ])