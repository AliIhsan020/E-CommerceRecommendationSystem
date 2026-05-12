from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.recommenders.cross_sell.cooccurrence_recommender import (
        CrossSellCooccurrenceRecommender,
    )
except ModuleNotFoundError:
    from cooccurrence_recommender import CrossSellCooccurrenceRecommender


DEFAULT_INPUT_PATH = Path("data/gold/baskets.parquet")
DEFAULT_MODEL_PATH = Path("artifacts/models/cross_sell_cooccurrence.pkl")
DEFAULT_METRICS_PATH = Path("artifacts/metrics/cross_sell_training_summary.json")
DEFAULT_SAMPLE_OUTPUT_PATH = Path(
    "artifacts/recommendation_outputs/cross_sell_sample_recommendations.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a basket-based cross-sell co-occurrence recommender."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--basket-column", default="articles")
    parser.add_argument("--date-column", default="t_dat")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE_OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-pair-count", type=int, default=5)
    parser.add_argument("--top-k-per-item", type=int, default=50)
    parser.add_argument("--max-basket-size", type=int, default=50)
    parser.add_argument("--confidence-shrinkage", type=float, default=20.0)
    parser.add_argument("--lift-weight", type=float, default=0.35)
    parser.add_argument("--count-weight", type=float, default=0.35)
    parser.add_argument("--popularity-penalty-weight", type=float, default=0.15)
    parser.add_argument("--test-days", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--max-eval-baskets", type=int, default=50_000)
    parser.add_argument("--eval-k-values", default="5,10,20")
    parser.add_argument("--refit-full", action="store_true")
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--sample-items", type=int, default=20)
    parser.add_argument("--recommendations-per-item", type=int, default=10)
    return parser.parse_args()


def parse_k_values(raw_value: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw_value.split(",") if value.strip()})
    if not values or values[0] < 1:
        raise ValueError("eval-k-values must contain positive integers")
    return values


def load_basket_frame(
    path: Path,
    basket_column: str,
    date_column: str,
    include_date_column: bool,
    sample_size: int | None,
    random_state: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    columns = [basket_column]
    if include_date_column:
        columns.append(date_column)

    df = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
    return df.reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame,
    date_column: str,
    test_days: int,
    test_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if test_days > 0:
        if date_column not in df.columns:
            raise ValueError(f"date column is required for test-days: {date_column}")

        split_df = df.copy()
        split_df[date_column] = pd.to_datetime(split_df[date_column])
        max_date = split_df[date_column].max()
        cutoff_date = max_date - pd.Timedelta(days=test_days)
        train_df = split_df[split_df[date_column] < cutoff_date]
        test_df = split_df[split_df[date_column] >= cutoff_date]

        return train_df, test_df, {
            "strategy": "time_based",
            "date_column": date_column,
            "test_days": test_days,
            "cutoff_date": str(cutoff_date.date()),
            "max_date": str(max_date.date()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        }

    if test_fraction > 0:
        if not 0 < test_fraction < 1:
            raise ValueError("test-fraction must be between 0 and 1")

        shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        split_index = int(len(shuffled) * (1 - test_fraction))
        train_df = shuffled.iloc[:split_index]
        test_df = shuffled.iloc[split_index:]

        return train_df, test_df, {
            "strategy": "random_fraction",
            "test_fraction": test_fraction,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        }

    return df, df.iloc[0:0], {
        "strategy": "none",
        "train_rows": int(len(df)),
        "test_rows": 0,
    }


def evaluate_model(
    model: CrossSellCooccurrenceRecommender,
    test_baskets: pd.Series,
    k_values: list[int],
    random_state: int,
    max_eval_baskets: int,
    use_fallback: bool,
) -> dict[str, Any]:
    if test_baskets.empty:
        return {"enabled": False}

    if max_eval_baskets > 0 and len(test_baskets) > max_eval_baskets:
        test_baskets = test_baskets.sample(
            n=max_eval_baskets,
            random_state=random_state,
        )

    rng = random.Random(random_state)
    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    reciprocal_ranks = {k: 0.0 for k in k_values}
    precision_sums = {k: 0.0 for k in k_values}
    evaluated_baskets = 0
    skipped_baskets = 0
    recommendation_counts = []
    unique_recommended_items: set[str] = set()

    for raw_basket in test_baskets:
        items = model.normalize_basket(raw_basket)
        if len(items) < 2:
            skipped_baskets += 1
            continue

        holdout_index = rng.randrange(len(items))
        target_item = items[holdout_index]
        context_items = items[:holdout_index] + items[holdout_index + 1 :]
        recommendations = model.recommend_for_basket(
            context_items,
            n=max_k,
            use_fallback=use_fallback,
        )
        recommended_item_ids = [rec["item_id"] for rec in recommendations]
        unique_recommended_items.update(recommended_item_ids)
        recommendation_counts.append(len(recommended_item_ids))
        evaluated_baskets += 1

        for k in k_values:
            top_k = recommended_item_ids[:k]
            if target_item in top_k:
                rank = top_k.index(target_item) + 1
                hits[k] += 1
                reciprocal_ranks[k] += 1 / rank
                precision_sums[k] += 1 / k

    if evaluated_baskets == 0:
        return {
            "enabled": True,
            "evaluated_baskets": 0,
            "skipped_baskets": skipped_baskets,
        }

    metrics: dict[str, Any] = {
        "enabled": True,
        "evaluated_baskets": evaluated_baskets,
        "skipped_baskets": skipped_baskets,
        "catalog_coverage": (
            len(unique_recommended_items) / len(model.item_counts)
            if model.item_counts
            else 0.0
        ),
        "unique_recommended_items": len(unique_recommended_items),
        "mean_recommendation_count": (
            sum(recommendation_counts) / evaluated_baskets
            if recommendation_counts
            else 0.0
        ),
    }

    for k in k_values:
        metrics[f"hit_rate_at_{k}"] = hits[k] / evaluated_baskets
        metrics[f"recall_at_{k}"] = hits[k] / evaluated_baskets
        metrics[f"precision_at_{k}"] = precision_sums[k] / evaluated_baskets
        metrics[f"mrr_at_{k}"] = reciprocal_ranks[k] / evaluated_baskets

    return metrics


def build_sample_recommendations(
    model: CrossSellCooccurrenceRecommender,
    sample_items: int,
    recommendations_per_item: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for source in model.get_popular_items(sample_items):
        source_item_id = source["item_id"]
        recommendations = model.recommend(
            item_id=source_item_id,
            n=recommendations_per_item,
        )

        for rank, rec in enumerate(recommendations, start=1):
            rows.append(
                {
                    "source_item_id": source_item_id,
                    "recommended_item_id": rec["item_id"],
                    "rank": rank,
                    "score": rec["score"],
                    "confidence": rec["confidence"],
                    "shrunk_confidence": rec["shrunk_confidence"],
                    "lift": rec["lift"],
                    "cosine": rec["cosine"],
                    "jaccard": rec["jaccard"],
                    "support": rec["support"],
                    "cooccurrence_count": rec["cooccurrence_count"],
                    "source_item_count": rec["source_item_count"],
                    "target_item_count": rec["target_item_count"],
                    "reason": rec["reason"],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.eval_k_values)
    include_date_column = args.test_days > 0

    df = load_basket_frame(
        path=args.input,
        basket_column=args.basket_column,
        date_column=args.date_column,
        include_date_column=include_date_column,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    train_df, test_df, split_summary = split_train_test(
        df=df,
        date_column=args.date_column,
        test_days=args.test_days,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )

    model = CrossSellCooccurrenceRecommender(
        min_pair_count=args.min_pair_count,
        top_k_per_item=args.top_k_per_item,
        max_basket_size=args.max_basket_size,
        confidence_shrinkage=args.confidence_shrinkage,
        lift_weight=args.lift_weight,
        count_weight=args.count_weight,
        popularity_penalty_weight=args.popularity_penalty_weight,
    )
    model.fit(train_df[args.basket_column])

    evaluation_summary = evaluate_model(
        model=model,
        test_baskets=test_df[args.basket_column],
        k_values=k_values,
        random_state=args.random_state,
        max_eval_baskets=args.max_eval_baskets,
        use_fallback=not args.disable_fallback,
    )

    if args.refit_full and not test_df.empty:
        model.fit(df[args.basket_column])
        model.training_summary["refit_full_after_evaluation"] = True

    model.save(args.model_output)

    report = {
        "training_summary": model.training_summary,
        "split_summary": split_summary,
        "evaluation_summary": evaluation_summary,
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    sample_recommendations = build_sample_recommendations(
        model=model,
        sample_items=args.sample_items,
        recommendations_per_item=args.recommendations_per_item,
    )
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    sample_recommendations.to_csv(args.sample_output, index=False)

    print("Training completed.")
    print(f"Model: {args.model_output}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Sample recommendations: {args.sample_output}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
