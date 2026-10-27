import logging
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

logger = logging.getLogger("DataCleaner")

NUMERIC_FEATURES = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_FEATURES = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

class LeakageSafePreprocessor:
    """
    Exercise 4: Leakage-Safe Preprocessor
    Learns all statistics (median, mode, clipping limits) strictly from training partition.
    Transforms evaluation sets using stored training parameters.
    """
    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.percentile_limits: Dict[str, Tuple[float, float]] = {}
        self._fitted = False

    def fit(self, train_features: pd.DataFrame):
        df = train_features.copy()

        # Learn categorical modes
        for c in self.categorical_cols:
            clean = df[c].astype(str).str.strip().replace("?", np.nan)
            mode_val = clean.dropna().mode()
            self.categorical_modes[c] = mode_val.iloc[0] if not mode_val.empty else "Missing"

        # Learn numerical medians and 1st/99th percentile outlier limits
        for n in self.numeric_cols:
            vals = pd.to_numeric(df[n], errors="coerce")
            self.numeric_medians[n] = float(vals.median())
            q01 = float(vals.quantile(0.01))
            q99 = float(vals.quantile(0.99))
            self.percentile_limits[n] = (q01, q99)

        self._fitted = True
        logger.info(f"Fitted LeakageSafePreprocessor on {len(df)} training records.")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("Cannot transform before calling fit()!")

        res = features.copy()

        # Apply mode imputation
        for c in self.categorical_cols:
            res[c] = res[c].astype(str).str.strip().replace("?", np.nan)
            res[c] = res[c].fillna(self.categorical_modes[c])

        # Apply median imputation and 1st/99th percentile clipping
        for n in self.numeric_cols:
            res[n] = pd.to_numeric(res[n], errors="coerce").fillna(self.numeric_medians[n])
            low, high = self.percentile_limits[n]
            res[n] = res[n].clip(lower=low, upper=high)

        return res

def clean_and_split_pipeline(
    raw_df: pd.DataFrame,
    seed: int = 101,
    artifact_path: str = "artifacts/preprocessor.joblib"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LeakageSafePreprocessor]:
    """
    Executes:
    1. Stratified 70/30 split before any imputation/cleaning.
    2. Drops exact duplicates on train partition only (test is untouched).
    3. Fits imputer & outlier bounds on train only.
    4. Serializes preprocessor via joblib.
    """
    logger.info("Executing clean_and_split_pipeline...")

    # Map target
    y = (raw_df["income"].astype(str).str.strip().isin([">50K", ">50K."])).astype(int)
    X = raw_df.drop(columns=["income"])

    # 1. Stratified 70/30 split before cleaning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    logger.info(f"Stratified partition created: Train={len(X_train)} (70%), Test={len(X_test)} (30%)")

    # 2. Drop exact duplicates from TRAIN ONLY
    train_full = pd.concat([X_train, y_train], axis=1)
    dup_count = train_full.duplicated().sum()
    train_clean = train_full.drop_duplicates()
    
    y_train = train_clean["income"]
    X_train = train_clean.drop(columns=["income"])
    logger.info(f"Removed {dup_count} duplicate rows from Train set. Test set remains untouched ({len(X_test)} rows).")

    # 3. Fit transformer on train partition only
    preprocessor = LeakageSafePreprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    preprocessor.fit(X_train)

    # 4. Transform train and test
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Serialize
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(preprocessor, artifact_path)
    logger.info(f"Persisted preprocessor artifact to: {artifact_path}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
