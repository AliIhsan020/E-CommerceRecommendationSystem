"""
src/data_pipeline/transactions-train-data-pipeline.py

Reads raw transactions-train.csv from data/bronze/
Cleans and validates the data
Saves the result to data/silver/ as parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# This script lives in src/data_pipeline/ so we go up two levels to reach root
ROOT   = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze" / "transactions_train.csv"
SILVER = ROOT / "data" / "silver" / "transactions_train.parquet"


# ── Step 1: Load ──────────────────────────────────────────────────────────────
def load(path: Path) -> pd.DataFrame:
    log.info(f"Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ── Step 2: Basic inspection (logged, not blocking) ───────────────────────────
def inspect(df: pd.DataFrame) -> None:
    null_pct = (df.isnull().sum() / len(df) * 100).round(1)
    high_null = null_pct[null_pct > 20]
    if not high_null.empty:
        log.warning(f"  Columns with >20% missing values:\n{high_null.to_string()}")
    else:
        log.info("  No columns with >20% missing values")
    log.info(f"  Duplicate rows: {df.duplicated().sum():,}")


# ── Step 3: Clean ─────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    # --- Remove exact duplicate rows ---
    df = df.drop_duplicates()
    log.info(f"  Duplicates removed: {original_len - len(df):,}")

    # --- Strip whitespace from string columns ---
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # --- Normalize string columns to lowercase ---
    for col in str_cols:
        df[col] = df[col].str.lower()

    # --- Parse date/time columns automatically ---
    # Any column whose name contains "date", "time", or "at" is tried as datetime
    date_candidates = [
        c for c in df.columns
        if any(keyword in c.lower() for keyword in ["date", "time", "at", "created", "updated"])
    ]
    for col in date_candidates:
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            log.info(f"  Parsed as datetime: {col}")
        except Exception:
            pass

    # --- Drop columns that are entirely empty ---
    all_null_cols = [c for c in df.columns if df[c].isnull().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)
        log.info(f"  Dropped fully empty columns: {all_null_cols}")

    # --- Fix numeric columns stored as strings (e.g. "1,234.50" → 1234.50) ---
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].str.replace(",", "", regex=False)
        try:
            converted = pd.to_numeric(cleaned, errors="raise")
            df[col] = converted
            log.info(f"  Converted to numeric: {col}")
        except (ValueError, TypeError):
            pass  # not a numeric column, leave as is

    # --- Amount / price columns: clip negative values to 0 ---
    amount_cols = [
        c for c in df.select_dtypes(include="number").columns
        if any(k in c.lower() for k in ["amount", "price", "total", "value", "cost"])
    ]
    for col in amount_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            log.warning(f"  {neg_count:,} negative values in '{col}' → clipped to 0")
            df[col] = df[col].clip(lower=0)

    log.info(f"  Clean shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ── Step 4: Add basic derived columns ─────────────────────────────────────────
def feature_hints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a few lightweight columns that are almost always useful.
    Feel free to delete or extend this in notebooks/02_features/.
    """

    # If there's a datetime column, extract year/month/day_of_week
    dt_cols = df.select_dtypes(include="datetime").columns.tolist()
    if dt_cols:
        col = dt_cols[0]  # use the first datetime column found
        df["_year"]        = df[col].dt.year
        df["_month"]       = df[col].dt.month
        df["_day_of_week"] = df[col].dt.dayofweek   # 0=Monday, 6=Sunday
        df["_is_weekend"]  = df["_day_of_week"].isin([5, 6]).astype(int)
        log.info(f"  Added time features from: {col}")

    return df


# ── Step 5: Save ──────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    size_mb = path.stat().st_size / 1e6
    log.info(f"  Saved → {path}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    log.info("=" * 55)
    log.info("TRANSACTIONS TRAIN — DATA PIPELINE")
    log.info("=" * 55)

    df = load(BRONZE)

    log.info("Step 2 | Inspect")
    inspect(df)

    log.info("Step 3 | Clean")
    df = clean(df)

    log.info("Step 4 | Feature hints")
    df = feature_hints(df)

    log.info("Step 5 | Save to silver")
    save(df, SILVER)

    log.info("Done ✓")
    return df   # return so notebooks can call run() and get the df back


if __name__ == "__main__":
    run()
