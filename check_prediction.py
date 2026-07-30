import pandas as pd

from src.predict import CreditRiskPredictor


def main() -> None:
    client = pd.DataFrame([
        {
            "LIMIT_BAL": 20000.0,
            "SEX": 2,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 24,
            "PAY_0": 2,
            "PAY_2": 2,
            "PAY_3": -1,
            "PAY_4": -1,
            "PAY_5": -2,
            "PAY_6": -2,
            "BILL_AMT1": 3913.0,
            "BILL_AMT2": 3102.0,
            "BILL_AMT3": 689.0,
            "BILL_AMT4": 0.0,
            "BILL_AMT5": 0.0,
            "BILL_AMT6": 0.0,
            "PAY_AMT1": 0.0,
            "PAY_AMT2": 689.0,
            "PAY_AMT3": 0.0,
            "PAY_AMT4": 0.0,
            "PAY_AMT5": 0.0,
            "PAY_AMT6": 0.0,
        }
    ])

    predictor = CreditRiskPredictor()

    probability = predictor.predict_probability(client)

    print(
        f"Вероятность дефолта: {probability:.2%}"
    )

    print(
        "Решение при пороге 0.50:",
        predictor.predict_class(
            client,
            threshold=0.50,
        ),
    )

    print(
        "Решение при пороге 0.44:",
        predictor.predict_class(
            client,
            threshold=0.44,
        ),
    )


if __name__ == "__main__":
    main()