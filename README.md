# Fraud Detection System

## Major Project: Credit Card Fraud Detection using Machine Learning

This project detects potentially fraudulent credit-card transactions using:
- Logistic Regression
- Random Forest
- Isolation Forest (anomaly detection)

The project is designed for the Kaggle ULB Credit Card Fraud Detection dataset.

### Dataset
Download `creditcard.csv` from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains 284,807 transactions, including 492 fraud cases, and the fraud class is highly imbalanced.

### Folder structure

```text
fraud_detection_project/
│
├── data/
│   └── creditcard.csv        <- put the Kaggle file here
│
├── models/
│   ├── best_fraud_model.joblib
│   └── isolation_forest.joblib
│
├── outputs/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── confusion matrices
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── fraud_predictions.csv
│   └── metrics.json
│
├── fraud_detection.py
├── requirements.txt
└── README.md
```

### Installation

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

Run:
```bash
python fraud_detection.py
```

### Important project points

1. `Class = 0` means legitimate transaction.
2. `Class = 1` means fraudulent transaction.
3. The dataset is extremely imbalanced, so accuracy alone is not reliable.
4. The project therefore reports precision, recall, F1, ROC-AUC and PR-AUC.
5. `class_weight='balanced'` and `class_weight='balanced_subsample'` give more importance to the minority fraud class.
6. A validation set is used to choose the probability threshold rather than blindly using 0.50.
7. The test set is kept separate until final evaluation.
8. Isolation Forest provides an unsupervised anomaly-detection approach.
9. Risk levels are generated from fraud probabilities.

### For viva

Be ready to explain:
- What is class imbalance?
- Why is accuracy misleading here?
- Difference between precision and recall.
- Why PR-AUC is useful.
- What is threshold tuning?
- What is Random Forest?
- What is Isolation Forest?
- Why is data leakage dangerous?
- Why do we split train/validation/test?
