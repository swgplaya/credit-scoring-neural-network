# Credit Default Risk Prediction

An end-to-end machine learning project for predicting the risk of credit card default using a neural network built with PyTorch.

The project covers the complete workflow: data inspection, preprocessing, model training, evaluation, comparison with a classical baseline, prediction, and deployment through a Streamlit web application.

## Live Demo

[Open the Streamlit application](https://credit-scoring-neural-network.streamlit.app/)

> This project is intended for educational purposes only. Its predictions must not be used for real lending or credit decisions.

## Project Overview

The model estimates whether an existing credit card client is likely to default in the following month.

It uses six months of behavioral and financial history, including:

- credit limit;
- client age and demographic categories;
- repayment status for the previous six months;
- monthly statement balances;
- monthly payment amounts.

This is a behavioral credit scoring task. The model evaluates existing credit card clients based on their payment history rather than assessing first-time credit applicants.

## Dataset

The project uses the **Default of Credit Card Clients** dataset.

The dataset contains:

- 30,000 client records;
- 23 input features after removing the technical `ID` column;
- one binary target variable;
- six months of repayment and billing history.

Target variable:

- `0` — no default in the following month;
- `1` — default in the following month.

Class distribution:

| Class | Observations | Share |
|---|---:|---:|
| No default | 23,364 | 77.88% |
| Default | 6,636 | 22.12% |

The dataset is not included in this repository. Download it from the [original dataset source](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) and place it in:

```text
data/raw/
```

## Data Preparation

The preprocessing pipeline performs the following steps:

1. Removes the technical `ID` column.
2. Renames the target variable to `default`.
3. Groups undocumented education categories `0`, `5`, and `6` into `Other`.
4. Groups undocumented marital status category `0` into `Other`.
5. Splits the data into training, validation, and test sets.
6. Standardizes numerical features using `StandardScaler`.
7. Encodes categorical features using one-hot encoding.

Dataset split:

| Subset | Observations | Share |
|---|---:|---:|
| Training | 21,000 | 70% |
| Validation | 4,500 | 15% |
| Test | 4,500 | 15% |

Stratified splitting preserves the original default rate in every subset.

The scaler and encoder are fitted only on the training data to prevent data leakage.

After preprocessing, the model receives **29 input features**.

## Neural Network

The model is implemented in PyTorch.

Architecture:

```text
29 input features
        ↓
Linear layer: 64 neurons
        ↓
ReLU
        ↓
Dropout: 20%
        ↓
Linear layer: 32 neurons
        ↓
ReLU
        ↓
Dropout: 10%
        ↓
1 output logit
```

Training configuration:

- loss function: `BCEWithLogitsLoss`;
- optimizer: `AdamW`;
- batch size: `256`;
- maximum epochs: `50`;
- early stopping patience: `7`;
- positive-class weighting to address class imbalance.

The best model is selected using validation ROC-AUC.

## Model Performance

### Neural network at threshold 0.50

| Metric | Test result |
|---|---:|
| Accuracy | 0.7613 |
| Precision | 0.4699 |
| Recall | 0.6114 |
| ROC-AUC | 0.7749 |

At the standard threshold, the model identifies approximately 61% of clients who later default.

### Neural network at threshold 0.44

| Metric | Test result |
|---|---:|
| Accuracy | 0.6993 |
| Precision | 0.3992 |
| Recall | 0.7098 |
| ROC-AUC | 0.7749 |

The lower threshold identifies approximately 71% of defaults but produces more false-positive risk classifications.

The appropriate threshold depends on the relative business cost of:

- failing to identify a future default;
- incorrectly classifying a reliable client as high-risk.

## Baseline Comparison

The neural network was compared with a class-weighted logistic regression baseline.

| Model | Threshold | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Neural network | 0.50 | 0.7613 | 0.4699 | 0.6114 | 0.7749 |
| Neural network | 0.44 | 0.6993 | 0.3992 | 0.7098 | 0.7749 |
| Logistic regression | 0.50 | 0.6831 | 0.3741 | 0.6416 | 0.7198 |

The neural network achieved a higher ROC-AUC, indicating better overall ranking of clients by default risk.

The logistic regression remains useful as an interpretable baseline and demonstrates that a more complex model should be compared against a simpler alternative.

## Evaluation Outputs

The evaluation pipeline generates:

- confusion matrices;
- ROC curve comparison;
- precision-recall curve;
- JSON file containing model metrics.

Generated files are stored in:

```text
reports/
```

## Web Application

The Streamlit application allows users to enter:

- credit limit;
- age;
- sex;
- education;
- marital status;
- repayment status for six months;
- monthly statement balances;
- monthly payment amounts;
- classification threshold.

The application displays:

- estimated model risk score;
- selected classification threshold;
- lower-risk or higher-risk classification.

The displayed score should be interpreted as a model risk score rather than a perfectly calibrated probability.

Because positive-class weighting was used during training, a displayed value such as `60%` does not necessarily mean that exactly 60 out of 100 similar clients will default. Probability calibration would be required for that interpretation.

## Project Structure

```text
credit_scoring_nn/
├── app.py
├── check_prediction.py
├── check_preprocessing.py
├── inspect_data.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── artifacts/
│   ├── model.pt
│   ├── preprocessor.joblib
│   ├── feature_names.json
│   └── metadata.json
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   ├── confusion_matrix_threshold_050.png
│   ├── confusion_matrix_tuned_threshold.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   └── evaluation_results.json
│
└── src/
    ├── __init__.py
    ├── data.py
    ├── evaluate.py
    ├── model.py
    ├── predict.py
    └── train.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/swgplaya/credit-scoring-neural-network
cd credit_scoring_nn
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Inspect the dataset

```bash
python inspect_data.py
```

### Check preprocessing

```bash
python check_preprocessing.py
```

### Train the neural network

Run the training script as a module from the project root:

```bash
python -m src.train
```

### Evaluate the model

```bash
python -m src.evaluate
```

### Test a single prediction

```bash
python check_prediction.py
```

### Run the web application locally

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Saved Artifacts

The training pipeline saves:

- `model.pt` — trained PyTorch model weights;
- `preprocessor.joblib` — fitted preprocessing pipeline;
- `feature_names.json` — transformed feature names;
- `metadata.json` — model configuration and evaluation metrics.

The web application loads these artifacts directly and does not retrain the model.

## Technologies

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn
- Streamlit
- Matplotlib
- Git and GitHub

## Limitations

This project has several important limitations:

- it uses a historical dataset from a specific country and period;
- performance may not generalize to other banks or populations;
- the output score is not fully probability-calibrated;
- no fairness or bias analysis has been performed;
- the neural network has not been compared with modern gradient-boosting models;
- no production monitoring or data-drift detection is implemented;
- demographic features would require careful legal and ethical review in a real financial system.

## Possible Improvements

Potential future extensions include:

- probability calibration;
- feature importance and explainability analysis;
- comparison with CatBoost, XGBoost, or LightGBM;
- automated hyperparameter tuning;
- model and data validation tests;
- Docker deployment;
- prediction logging;
- model monitoring and drift detection;
- fairness analysis across demographic groups.

## Disclaimer

This repository is an educational machine learning project.

It is not a validated banking model and must not be used to approve, reject, limit, or otherwise make real credit decisions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.