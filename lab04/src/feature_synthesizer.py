import logging
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger("FeatureSynthesizer")

def compute_derived_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exercise 5: Feature Engineering
    - capital_net = capital-gain - capital-loss
    - hours_per_age = hours-per-week / (age - 16)
    """
    augmented = df.copy()
    augmented["capital_net"] = augmented["capital-gain"] - augmented["capital-loss"]
    denom = (augmented["age"] - 16).clip(lower=1.0)
    augmented["hours_per_age"] = augmented["hours-per-week"] / denom
    return augmented

def benchmark_logistic_regression(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: pd.Series,
    test_y: pd.Series,
    numeric_vars: list,
    categorical_vars: list,
    seed: int = 101,
    artifact_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    Evaluates Logistic Regression performance:
    1. Base features only (without derived features)
    2. Augmented features (with capital_net and hours_per_age)
    """
    logger.info("Executing baseline modeling comparison...")

    # Phase 1: Without Derived Features
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler_base = StandardScaler()

    cat_train_encoded = ohe.fit_transform(train_x[categorical_vars])
    cat_test_encoded = ohe.transform(test_x[categorical_vars])

    num_train_scaled = scaler_base.fit_transform(train_x[numeric_vars])
    num_test_scaled = scaler_base.transform(test_x[numeric_vars])

    matrix_train_base = np.hstack([num_train_scaled, cat_train_encoded])
    matrix_test_base = np.hstack([num_test_scaled, cat_test_encoded])

    model_base = LogisticRegression(max_iter=1000, random_state=seed)
    model_base.fit(matrix_train_base, train_y)
    preds_base = model_base.predict(matrix_test_base)

    metrics_without = {
        "accuracy": round(float(accuracy_score(test_y, preds_base)), 4),
        "f1_score": round(float(f1_score(test_y, preds_base)), 4),
        "precision": round(float(precision_score(test_y, preds_base)), 4),
        "recall": round(float(recall_score(test_y, preds_base)), 4)
    }
    logger.info(f"Model WITHOUT derived features: Accuracy={metrics_without['accuracy']}, F1={metrics_without['f1_score']}")

    # Phase 2: With Derived Features
    logger.info("Injecting capital_net and hours_per_age...")
    train_x_feat = compute_derived_attributes(train_x)
    test_x_feat = compute_derived_attributes(test_x)

    augmented_numerics = numeric_vars + ["capital_net", "hours_per_age"]
    scaler_eng = StandardScaler()

    num_train_feat_scaled = scaler_eng.fit_transform(train_x_feat[augmented_numerics])
    num_test_feat_scaled = scaler_eng.transform(test_x_feat[augmented_numerics])

    matrix_train_eng = np.hstack([num_train_feat_scaled, cat_train_encoded])
    matrix_test_eng = np.hstack([num_test_feat_scaled, cat_test_encoded])

    model_eng = LogisticRegression(max_iter=1000, random_state=seed)
    model_eng.fit(matrix_train_eng, train_y)
    preds_eng = model_eng.predict(matrix_test_eng)

    metrics_with = {
        "accuracy": round(float(accuracy_score(test_y, preds_eng)), 4),
        "f1_score": round(float(f1_score(test_y, preds_eng)), 4),
        "precision": round(float(precision_score(test_y, preds_eng)), 4),
        "recall": round(float(recall_score(test_y, preds_eng)), 4)
    }
    logger.info(f"Model WITH derived features:    Accuracy={metrics_with['accuracy']}, F1={metrics_with['f1_score']}")

    delta_metrics = {
        "accuracy_gain": round(metrics_with["accuracy"] - metrics_without["accuracy"], 4),
        "f1_gain": round(metrics_with["f1_score"] - metrics_without["f1_score"], 4)
    }

    eval_report = {
        "baseline_without_engineered_features": metrics_without,
        "enhanced_with_engineered_features": metrics_with,
        "improvement_delta": delta_metrics
    }

    os.makedirs(artifact_dir, exist_ok=True)
    report_file = os.path.join(artifact_dir, "model_evaluation.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=4)
    logger.info(f"Stored evaluation summary to: {report_file}")

    return eval_report
