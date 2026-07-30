from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def find_dataset() -> Path:
    """Находит единственный поддерживаемый датасет в data/raw."""

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Папка не найдена: {RAW_DATA_DIR}"
        )

    datasets = [
        path
        for path in RAW_DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not datasets:
        raise FileNotFoundError(
            "В папке data/raw не найден файл CSV, XLS или XLSX."
        )

    if len(datasets) > 1:
        files = "\n".join(f"  - {path.name}" for path in datasets)

        raise RuntimeError(
            "В папке data/raw найдено несколько датасетов:\n"
            f"{files}\n"
            "Оставь только нужный файл."
        )

    return datasets[0]


def read_excel_dataset(path: Path) -> pd.DataFrame:
    """
    Определяет строку с настоящими заголовками.

    В распространённой версии датасета первая строка Excel содержит
    служебные названия X1, X2 и т. д., а настоящие заголовки находятся
    во второй строке.
    """

    preview = pd.read_excel(
        path,
        header=None,
        nrows=10,
    )

    header_row = None

    for row_index, row in preview.iterrows():
        values = {
            str(value).strip()
            for value in row.dropna()
        }

        has_id = "ID" in values
        has_credit_limit = "LIMIT_BAL" in values

        if has_id and has_credit_limit:
            header_row = row_index
            break

    if header_row is None:
        raise ValueError(
            "Не удалось определить строку с заголовками Excel-файла."
        )

    print(f"Строка заголовков Excel: {header_row + 1}")

    return pd.read_excel(
        path,
        header=header_row,
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Загружает CSV или Excel-файл."""

    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension in {".xls", ".xlsx"}:
        return read_excel_dataset(path)

    raise ValueError(
        f"Неподдерживаемое расширение: {extension}"
    )


def find_target_column(data: pd.DataFrame) -> str | None:
    """Ищет целевую переменную в разных версиях датасета."""

    possible_names = [
        "default payment next month",
        "default.payment.next.month",
        "default",
        "Y",
    ]

    for column in possible_names:
        if column in data.columns:
            return column

    return None


def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    dataset_path = find_dataset()

    print(f"Датасет: {dataset_path.name}")

    data = load_dataset(dataset_path)

    # Убираем случайные пробелы из названий столбцов.
    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    print_separator("РАЗМЕР ТАБЛИЦЫ")
    print(f"Строк:    {data.shape[0]}")
    print(f"Столбцов: {data.shape[1]}")

    print_separator("НАЗВАНИЯ СТОЛБЦОВ")

    for number, column in enumerate(data.columns, start=1):
        print(f"{number:2}. {column}")

    print_separator("ПЕРВЫЕ ПЯТЬ СТРОК")
    print(data.head())

    print_separator("ТИПЫ ДАННЫХ")
    print(data.dtypes)

    print_separator("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")

    missing_values = data.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("Пропущенных значений не найдено.")
    else:
        print(missing_values)

    print_separator("ДУБЛИКАТЫ")

    print(f"Полностью одинаковых строк: {data.duplicated().sum()}")

    if "ID" in data.columns:
        print(f"Повторяющихся ID: {data['ID'].duplicated().sum()}")

    print_separator("КАТЕГОРИАЛЬНЫЕ ЗНАЧЕНИЯ")

    categorical_columns = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]

    for column in categorical_columns:
        if column in data.columns:
            values = sorted(data[column].dropna().unique())
            print(f"{column}: {values}")

    print_separator("ИСТОРИЯ ПЛАТЕЖЕЙ")

    payment_status_columns = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    for column in payment_status_columns:
        if column in data.columns:
            values = sorted(data[column].dropna().unique())
            print(f"{column}: {values}")

    print_separator("ЦЕЛЕВАЯ ПЕРЕМЕННАЯ")

    target_column = find_target_column(data)

    if target_column is None:
        print("Целевая переменная автоматически не найдена.")
    else:
        print(f"Название: {target_column}")

        target_counts = data[target_column].value_counts().sort_index()
        target_percentages = (
            data[target_column]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )

        target_summary = pd.DataFrame({
            "Количество": target_counts,
            "Доля, %": target_percentages,
        })

        print(target_summary)

    print_separator("ОБЩАЯ ИНФОРМАЦИЯ")
    data.info()


if __name__ == "__main__":
    main()