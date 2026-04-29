"""
H&M Basket Recommendation — Preprocessing Pipeline
=====================================================
Steps:
  1. Load transactions_train.csv
  2. Group duplicates → 'quantity' feature
  3. Aggregate articles per (t_dat, customer_id) → basket
  4. Drop single-item baskets
  5. Save result
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_PATH  = Path("data/bronze/transactions_train.csv")
OUTPUT_PATH = Path("data/gold/baskets.parquet")   # parquet önerilir; csv istersen değiştir
MIN_BASKET_SIZE = 2                           # bu sayının altındaki sepetler düşülür


# ── Pipeline ──────────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    log.info(f"Loading {path} ...")
    df = pd.read_csv(path, dtype={"article_id": str, "customer_id": str})
    log.info(f"  Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


def group_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Aynı (tarih, müşteri, ürün, fiyat, kanal) satırlarını quantity'ye çevir."""
    original_len = len(df)

    group_cols = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    existing_cols = [c for c in group_cols if c in df.columns]

    df = df.groupby(existing_cols).size().reset_index(name="quantity")
    log.info(f"  Duplicates → quantity. Rows: {original_len:,} → {len(df):,}")
    return df


def build_baskets(df: pd.DataFrame) -> pd.DataFrame:
    """
    (t_dat, customer_id) bazında article_id'leri listeye topla.
    apply() yerine hızlı agg + size kullanılıyor (~28M satırda saniyeler içinde biter).
    """
    log.info("Building baskets ...")

    grp = ["t_dat", "customer_id"]

    articles   = df.groupby(grp)["article_id"].agg(list).rename("articles")
    basket_size = df.groupby(grp).size().rename("basket_size")

    baskets = pd.concat([articles, basket_size], axis=1).reset_index()

    log.info(f"  Total baskets: {len(baskets):,}")
    return baskets


def filter_single_items(df: pd.DataFrame, min_size: int = MIN_BASKET_SIZE) -> pd.DataFrame:
    """Sepet boyutu min_size'dan küçük olan satırları çıkar."""
    before = len(df)
    df = df[df["basket_size"] >= min_size].reset_index(drop=True)
    dropped = before - len(df)
    log.info(
        f"  Dropped {dropped:,} single-item baskets. "
        f"Remaining: {len(df):,} baskets"
    )
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    log.info(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH):
    df = load(input_path)
    df = group_duplicates(df)
    df = build_baskets(df)
    df = filter_single_items(df)
    save(df, output_path)
    log.info("Done.")
    return df


if __name__ == "__main__":
    run_pipeline()