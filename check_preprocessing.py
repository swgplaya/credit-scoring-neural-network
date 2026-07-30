import numpy as np

from src.data import (
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    load_clean_data,
    split_data,
)


def print_target_distribution(name, target) -> None:
    counts = target.value_counts().sort_index()
    percentages = (
        target.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(f"\n{name}:")

    for target_value in counts.index:
        print(
            f"  Класс {target_value}: "
            f"{counts[target_value]} наблюдений "
            f"({percentages[target_value]}%)"
        )


def main() -> None:
    data = load_clean_data()

    print("=" * 70)
    print("ОЧИЩЕННЫЕ ДАННЫЕ")
    print("=" * 70)

    print(f"Размер: {data.shape}")
    print(f"Целевая переменная: {TARGET_COLUMN}")

    print("\nКатегориальные значения после очистки:")

    for column in CATEGORICAL_COLUMNS:
        values = sorted(data[column].unique())
        print(f"{column}: {values}")

    splits = split_data(data)

    print("\n" + "=" * 70)
    print("РАЗДЕЛЕНИЕ ВЫБОРКИ")
    print("=" * 70)

    print(f"Обучающая:    {splits.X_train.shape}")
    print(f"Валидационная: {splits.X_val.shape}")
    print(f"Тестовая:      {splits.X_test.shape}")

    print_target_distribution(
        "Обучающая выборка",
        splits.y_train,
    )

    print_target_distribution(
        "Валидационная выборка",
        splits.y_val,
    )

    print_target_distribution(
        "Тестовая выборка",
        splits.y_test,
    )

    preprocessor = build_preprocessor(
        splits.X_train
    )

    # Обучаем преобразователь только на обучающей выборке.
    X_train_processed = preprocessor.fit_transform(
        splits.X_train
    )

    # Для остальных выборок используем уже обученный преобразователь.
    X_val_processed = preprocessor.transform(
        splits.X_val
    )

    X_test_processed = preprocessor.transform(
        splits.X_test
    )

    # PyTorch обычно работает с float32.
    X_train_processed = X_train_processed.astype(
        np.float32
    )
    X_val_processed = X_val_processed.astype(
        np.float32
    )
    X_test_processed = X_test_processed.astype(
        np.float32
    )

    print("\n" + "=" * 70)
    print("ПОСЛЕ ПРЕОБРАЗОВАНИЯ")
    print("=" * 70)

    print(
        "Обучающая матрица:",
        X_train_processed.shape,
    )
    print(
        "Валидационная матрица:",
        X_val_processed.shape,
    )
    print(
        "Тестовая матрица:",
        X_test_processed.shape,
    )

    print(
        "Тип данных:",
        X_train_processed.dtype,
    )

    feature_names = preprocessor.get_feature_names_out()

    print(
        "Количество признаков после преобразования:",
        len(feature_names),
    )

    print("\nНазвания преобразованных признаков:")

    for number, feature_name in enumerate(
        feature_names,
        start=1,
    ):
        print(f"{number:2}. {feature_name}")


if __name__ == "__main__":
    main()