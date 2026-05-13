from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommenders.cross_sell.cooccurrence_recommender import (
    CrossSellCooccurrenceRecommender,
)
from src.recommenders.cross_sell.train_cross_sell import (
    evaluate_model,
    load_basket_frame,
    parse_k_values,
    split_train_test,
)


DEFAULT_INPUT_PATH = Path("data/gold/baskets.parquet")
DEFAULT_RESULTS_PATH = Path("artifacts/metrics/cross_sell_tuning_results.csv")
DEFAULT_BEST_PARAMS_PATH = Path("artifacts/metrics/cross_sell_best_params.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune cross-sell co-occurrence recommender hyperparameters."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--basket-column", default="articles")
    parser.add_argument("--date-column", default="t_dat")
    parser.add_argument("--sample-size", type=int, default=150_000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--test-days", type=int, default=0)
    parser.add_argument("--max-eval-baskets", type=int, default=10_000)
    parser.add_argument("--eval-k-values", default="5,10,20")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--best-params-output", type=Path, default=DEFAULT_BEST_PARAMS_PATH)
    parser.add_argument("--disable-fallback", action="store_true")
    return parser.parse_args()


def build_search_space() -> list[dict[str, Any]]:
    fixed_baseline = {
        "min_pair_count": 5,
        "top_k_per_item": 100,
        "max_basket_size": 50,
        "confidence_shrinkage": 20.0,
        "lift_weight": 0.35,
        "count_weight": 0.35,
        "popularity_penalty_weight": 0.15,
    }

    grid = {
        "min_pair_count": [2, 3, 5, 10],
        "top_k_per_item": [50, 100],
        "max_basket_size": [30, 50],
        "confidence_shrinkage": [5.0, 20.0, 50.0, 100.0],
        "lift_weight": [0.1, 0.35, 0.6],
        "count_weight": [0.1, 0.35, 0.6],
        "popularity_penalty_weight": [0.0, 0.15, 0.3],
    }

    combinations = [
        dict(zip(grid.keys(), values))
        for values in itertools.product(*grid.values())
    ]
    combinations = [params for params in combinations if params != fixed_baseline]
    return [fixed_baseline, *combinations]


def sample_configs(
    search_space: list[dict[str, Any]],
    trials: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if trials >= len(search_space):
        return search_space

    baseline = search_space[0]
    remaining = search_space[1:]
    rng = random.Random(random_state)
    return [baseline, *rng.sample(remaining, k=trials - 1)]


def objective_score(metrics: dict[str, Any]) -> float:
    return (
        metrics.get("hit_rate_at_10", 0.0)
        + 0.25 * metrics.get("mrr_at_10", 0.0)
        + 0.05 * metrics.get("catalog_coverage", 0.0)
    )


def flatten_result(
    trial_id: int,
    params: dict[str, Any],
    training_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    result = {
        "trial_id": trial_id,
        "objective_score": objective_score(evaluation_summary),
        "elapsed_seconds": round(elapsed_seconds, 3),
        **params,
        "total_baskets": training_summary.get("total_baskets"),
        "unique_items": training_summary.get("unique_items"),
        "unique_pairs": training_summary.get("unique_pairs"),
        "retained_items": training_summary.get("retained_items"),
    }
    for key, value in evaluation_summary.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.eval_k_values)
    include_date_column = args.test_days > 0

    print("Loading basket data...", flush=True)
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
    print(json.dumps(split_summary, indent=2, sort_keys=True), flush=True)

    configs = sample_configs(
        search_space=build_search_space(),
        trials=args.trials,
        random_state=args.random_state,
    )
    results = []

    for trial_id, params in enumerate(configs, start=1):
        started_at = time.perf_counter()
        print(
            f"\nTrial {trial_id}/{len(configs)}: {json.dumps(params, sort_keys=True)}",
            flush=True,
        )

        model = CrossSellCooccurrenceRecommender(**params)
        model.fit(train_df[args.basket_column])
        evaluation_summary = evaluate_model(
            model=model,
            test_baskets=test_df[args.basket_column],
            k_values=k_values,
            random_state=args.random_state,
            max_eval_baskets=args.max_eval_baskets,
            use_fallback=not args.disable_fallback,
        )
        elapsed_seconds = time.perf_counter() - started_at
        result = flatten_result(
            trial_id=trial_id,
            params=params,
            training_summary=model.training_summary,
            evaluation_summary=evaluation_summary,
            elapsed_seconds=elapsed_seconds,
        )
        results.append(result)
        args.results_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).sort_values(
            by=["objective_score", "hit_rate_at_10", "mrr_at_10"],
            ascending=False,
        ).to_csv(args.results_output, index=False)
        print(
            "score={score:.5f} hit@10={hit:.5f} mrr@10={mrr:.5f} coverage={coverage:.5f}".format(
                score=result["objective_score"],
                hit=result.get("hit_rate_at_10", 0.0),
                mrr=result.get("mrr_at_10", 0.0),
                coverage=result.get("catalog_coverage", 0.0),
            ),
            flush=True,
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["objective_score", "hit_rate_at_10", "mrr_at_10"],
        ascending=False,
    )
    best_result = results_df.iloc[0].to_dict()
    best_params = {
        key: best_result[key]
        for key in [
            "min_pair_count",
            "top_k_per_item",
            "max_basket_size",
            "confidence_shrinkage",
            "lift_weight",
            "count_weight",
            "popularity_penalty_weight",
        ]
    }

    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results_output, index=False)
    args.best_params_output.write_text(
        json.dumps(
            {
                "best_params": best_params,
                "best_result": best_result,
                "split_summary": split_summary,
                "sample_size": args.sample_size,
                "trials": len(configs),
                "objective": "hit_rate_at_10 + 0.25*mrr_at_10 + 0.05*catalog_coverage",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\nBest parameters:", flush=True)
    print(json.dumps(best_params, indent=2, sort_keys=True), flush=True)
    print(f"Results: {args.results_output}", flush=True)
    print(f"Best params: {args.best_params_output}", flush=True)


if __name__ == "__main__":
    main()
