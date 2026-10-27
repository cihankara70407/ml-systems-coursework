import logging
import hashlib
import os
import yaml
import pandas as pd
from core.dataset_loader import ingest_adult_dataset
from core.schema_verifier import SchemaValidator
from core.data_cleaner import clean_and_split_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from core.feature_synthesizer import benchmark_logistic_regression
from core.label_consistency import audit_duplicate_conflicts

logger = logging.getLogger("PipelineEngine")

def calculate_sha256(df: pd.DataFrame) -> str:
    """Computes SHA-256 fingerprint from deterministic serialized bytes."""
    serialized = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()

def run_complete_pipeline(config_file: str = "configs/pipeline_config.yaml"):
    """
    Executes the comprehensive data engineering pipeline for Cihan Karayigit (70407).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("==================================================================")
    logger.info("  Data Pipeline & Quality Engineering - Cihan Karayigit (70407)   ")
    logger.info("==================================================================")

    seed = 101

    # 1. Ingestion
    raw_data = ingest_adult_dataset()

    # 2. Schema Validation
    validator = SchemaValidator()
    audit_results = validator.audit_dataset(raw_data)
    validator.write_markdown_log(audit_results, "docs/data_validation_log.md")

    # 3. Clean & Split (Zero-Leakage 70/30)
    X_train, X_test, y_train, y_test, preprocessor = clean_and_split_pipeline(
        raw_data, seed=seed, artifact_path="artifacts/preprocessor.joblib"
    )

    # 4. Feature Synthesis & Baseline Benchmark
    eval_metrics = benchmark_logistic_regression(
        X_train, X_test, y_train, y_test,
        numeric_vars=NUMERIC_FEATURES,
        categorical_vars=CATEGORICAL_FEATURES,
        seed=seed,
        artifact_dir="artifacts"
    )

    # 5. Label Integrity & Duplicate Conflict Audit
    audit_duplicate_conflicts(raw_data, sample_seed=seed, doc_path="docs/label_integrity_audit.md")

    # 6. Cryptographic Hashing
    train_combined = pd.concat([X_train, y_train], axis=1)
    test_combined = pd.concat([X_test, y_test], axis=1)

    train_hash = calculate_sha256(train_combined)
    test_hash = calculate_sha256(test_combined)

    logger.info(f"Generated SHA-256 Fingerprint (Train Set): {train_hash}")
    logger.info(f"Generated SHA-256 Fingerprint (Test Set):  {test_hash}")

    # 7. Store configuration YAML
    config_dict = {
        "pipeline_metadata": {
            "author": "Cihan Karayigit",
            "student_id": 70407,
            "version": "1.0.0",
            "seed": seed
        },
        "dataset": {
            "source": "OpenML adult (v2)",
            "instances": 48842,
            "features": 14,
            "target": "income"
        },
        "sha256_checksums": {
            "processed_train_data": train_hash,
            "processed_test_data": test_hash
        },
        "evaluation_summary": eval_metrics
    }

    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Configuration YAML persisted: {config_file}")

    logger.info("==================================================================")
    logger.info("  Pipeline Execution Complete. All Graded Evidence Generated!    ")
    logger.info("==================================================================")
    return config_dict
