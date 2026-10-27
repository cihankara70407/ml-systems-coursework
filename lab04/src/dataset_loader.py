import logging
import pandas as pd
from sklearn.datasets import fetch_openml

logger = logging.getLogger("DataIngestion")

def ingest_adult_dataset() -> pd.DataFrame:
    """
    Exercise 2: Data Ingestion
    Retrieves raw UCI Adult data via OpenML without alterations.
    Logs row count (48,842) and column count (15).
    """
    logger.info("Executing dataset fetch from OpenML repository (name='adult', version=2)...")
    dataset = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = dataset.frame.copy()

    # Normalize target column name
    if "class" in df.columns:
        df = df.rename(columns={"class": "income"})

    total_rows, total_cols = df.shape
    logger.info(f"Ingested raw dataset: {total_rows} rows x {total_cols} columns")

    assert total_rows == 48842, f"Row count mismatch! Expected 48842, got {total_rows}"
    assert total_cols == 15, f"Column count mismatch! Expected 15, got {total_cols}"
    logger.info("[VERIFIED] Ingestion checks passed: exactly 48842 instances and 15 features confirmed.")

    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    raw = ingest_adult_dataset()
    print("Preview of ingested data:\n", raw.head(2))
