# Credit Card Fraud Detection

Binary classification of credit card transactions as fraudulent or legitimate, framed as a risk-scoring component for a bank's fraud monitoring system. Compares Logistic Regression, Random Forest, and K-Nearest Neighbours inside a single scikit-learn pipeline with grid search.

Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset) on Kaggle. 10,000 transactions, 9 predictors, no missing values and no duplicates.

## Features and target

Predictors cover transaction amount, hour of day, merchant category, foreign transaction flag, location mismatch flag, device trust score, transaction velocity in the last 24 hours, and cardholder age. The target `is_fraud` is heavily imbalanced: 151 fraud cases against 9,849 legitimate, or 1.51%.

Because of that imbalance, accuracy is close to meaningless (predicting "never fraud" scores 98.5%), so model selection uses F1 with recall as the secondary consideration.

## What the EDA shows

Two binary flags dominate. Foreign transactions carry an 8.4% fraud rate against 0.8% for domestic ones, and location mismatches show the same jump, 8.4% against 0.9%. Merchant category barely separates at all, ranging only from 1.2% (Clothing) to 2.0% (Grocery).

## Pipeline

Numeric features are median-imputed and standardised; categoricals are mode-imputed and one-hot encoded. Both branches sit inside a `ColumnTransformer` wrapped in a `Pipeline`, so preprocessing is refitted inside every cross-validation fold rather than leaking across splits. Each model is tuned with 5-fold `GridSearchCV` scoring on F1, on a stratified 80/20 split.

## Results

Test set of 2,000 transactions containing 30 fraud cases:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | 0.994 | 1.000 | 0.567 | **0.723** |
| KNN | 0.992 | 0.810 | 0.567 | 0.667 |
| Logistic Regression | 0.960 | 0.270 | 1.000 | 0.426 |

Random Forest wins on F1. Its confusion matrix is clean on precision and weak on recall: 17 of 30 frauds caught, 13 missed, zero false positives out of 1,970 legitimate transactions.

Logistic Regression with `class_weight="balanced"` sits at the opposite extreme, catching every fraud but flagging roughly 3.7 legitimate transactions for each real one. Which trade-off is right depends on the cost of a missed fraud against the cost of reviewing a false alarm, and F1 does not encode that asymmetry. In a real deployment the decision threshold would be tuned against those costs rather than left at 0.5.

Feature importances line up with the EDA: transaction hour (0.25) and device trust score (0.25) lead, followed by the foreign transaction flag (0.13), velocity (0.12), and location mismatch (0.12). Merchant category contributes almost nothing.

## Caveats

The dataset is synthetic, so the relationships are cleaner and more separable than real transaction data. With only 30 fraud cases in the test set, every metric is coarse: one additional catch moves recall by 3.3 points, so the gap between Random Forest and KNN is not meaningful at this sample size.

## Running it

```bash
pip install pandas numpy scikit-learn seaborn matplotlib plotly
```

Download the dataset from Kaggle and update the path in the ingestion cell.

## Possible extensions

Report PR-AUC rather than F1, since it summarises performance across all thresholds and is the standard choice for heavily imbalanced detection problems. Tune the decision threshold explicitly against an assumed cost ratio for missed fraud versus false alarms. Validate on a larger or real dataset before drawing conclusions about which model family wins.
