from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd

try:
    from cross_sell_recommender import CrossSellAssociationRulesRecommender
except ModuleNotFoundError:
    from src.recommenders.cross_sell.cross_sell_recommender import (
        CrossSellAssociationRulesRecommender,
    )


PARAM_GRID = [
    {"min_item_support": 10, "min_pair_support": 3, "score_formula": "confidence_lift"},
    {"min_item_support": 20, "min_pair_support": 5, "score_formula": "confidence_lift"},
    {"min_item_support": 50, "min_pair_support": 10, "score_formula": "confidence_lift"},
    {"min_item_support": 20, "min_pair_support": 5, "score_formula": "confidence_support"},
    {"min_item_support": 20, "min_pair_support": 5, "score_formula": "jaccard_lift"},
]


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "requirements.txt").exists() and (path / "data").exists():
            return path
    raise FileNotFoundError("Project root could not be found.")


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_train_test(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = find_project_root(Path.cwd().resolve())
    train_path = project_path(project_root, args.train_path)
    test_path = project_path(project_root, args.test_path)
    return pd.read_parquet(train_path), pd.read_parquet(test_path)


def sample_eval_baskets(test_baskets: pd.DataFrame, args) -> pd.DataFrame:
    if args.eval_sample_size > 0 and len(test_baskets) > args.eval_sample_size:
        return test_baskets.sample(args.eval_sample_size, random_state=args.random_state)
    return test_baskets


def train_baseline(args) -> None:
    project_root = find_project_root(Path.cwd().resolve())
    train_baskets, test_baskets = load_train_test(args)
    eval_baskets = sample_eval_baskets(test_baskets, args)

    model = CrossSellAssociationRulesRecommender(
        min_item_support=args.min_item_support,
        min_pair_support=args.min_pair_support,
        score_formula=args.score_formula,
        max_basket_size_for_pairs=args.max_basket_size_for_pairs,
        top_n_rules_per_item=args.top_n_rules_per_item,
        recommend_k=args.recommend_k,
    )
    model.fit(train_baskets)
    metrics = model.evaluate(eval_baskets, top_k=args.recommend_k)

    model_path = project_path(project_root, args.baseline_model_path)
    rules_path = project_path(project_root, args.baseline_rules_path)
    metrics_path = project_path(project_root, args.baseline_metrics_path)

    model.save_rules(rules_path)
    model.save_model(model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved baseline rules: {rules_path}")
    print(f"Saved baseline model: {model_path}")
    print(f"Saved baseline metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


def tune_model(args) -> None:
    project_root = find_project_root(Path.cwd().resolve())
    train_baskets, test_baskets = load_train_test(args)
    eval_baskets = sample_eval_baskets(test_baskets, args)

    articles = train_baskets["articles"].apply(
        CrossSellAssociationRulesRecommender.normalize_articles
    )
    item_support = articles.explode().value_counts().astype("int64")
    popular_items = item_support.index.tolist()

    base_min_item_support = min(params["min_item_support"] for params in PARAM_GRID)
    pair_counts, used_basket_count = CrossSellAssociationRulesRecommender.count_pair_counts(
        articles=articles,
        item_support=item_support,
        min_item_support=base_min_item_support,
        max_basket_size_for_pairs=args.max_basket_size_for_pairs,
    )

    results = []
    best_model = None
    best_metrics = None
    best_selection_score = None

    for params in PARAM_GRID:
        model = CrossSellAssociationRulesRecommender(
            min_item_support=params["min_item_support"],
            min_pair_support=params["min_pair_support"],
            score_formula=params["score_formula"],
            max_basket_size_for_pairs=args.max_basket_size_for_pairs,
            top_n_rules_per_item=args.top_n_rules_per_item,
            recommend_k=args.recommend_k,
        )
        model.fit_from_counts(
            pair_counts=pair_counts,
            item_support=item_support,
            n_baskets=len(train_baskets),
            popular_items=popular_items,
            used_basket_count=used_basket_count,
        )
        metrics = model.evaluate(eval_baskets, top_k=args.recommend_k)

        row = {
            **params,
            "rule_count": int(len(model.rules_)) if model.rules_ is not None else 0,
            "antecedent_count": (
                int(model.rules_["antecedent"].nunique())
                if model.rules_ is not None and not model.rules_.empty
                else 0
            ),
            **metrics,
        }
        results.append(row)

        selection_score = metrics[f"recall_at_{args.recommend_k}"]
        if best_selection_score is None or selection_score > best_selection_score:
            best_model = model
            best_metrics = metrics
            best_selection_score = selection_score

    tuning_results = pd.DataFrame(results).sort_values(
        [
            f"recall_at_{args.recommend_k}",
            f"hit_rate_at_{args.recommend_k}",
            f"precision_at_{args.recommend_k}",
        ],
        ascending=False,
    )

    if best_model is None or best_metrics is None:
        raise RuntimeError("No tuning result was produced.")

    best_model_path = project_path(project_root, args.best_model_path)
    best_rules_path = project_path(project_root, args.best_rules_path)
    tuning_results_path = project_path(project_root, args.tuning_results_path)
    best_metrics_path = project_path(project_root, args.best_metrics_path)

    tuning_results_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(tuning_results_path, index=False)
    best_model.save_rules(best_rules_path)
    best_model.save_model(best_model_path)

    best_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with best_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": {
                    "min_item_support": best_model.min_item_support,
                    "min_pair_support": best_model.min_pair_support,
                    "score_formula": best_model.score_formula,
                },
                **best_metrics,
            },
            f,
            indent=2,
        )

    print(f"Saved tuning results: {tuning_results_path}")
    print(f"Saved best rules: {best_rules_path}")
    print(f"Saved best model: {best_model_path}")
    print(f"Saved best metrics: {best_metrics_path}")
    print(tuning_results.head().to_string(index=False))


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train or tune the cross-sell recommender.")
    parser.add_argument("--mode", choices=["baseline", "tuning"], default="tuning")
    parser.add_argument("--train-path", default="data/gold/cross_sell_train_baskets.parquet")
    parser.add_argument("--test-path", default="data/gold/cross_sell_test_baskets.parquet")

    parser.add_argument(
        "--baseline-model-path",
        default="artifacts/models/cross_sell_association_rules_baseline.pkl",
    )
    parser.add_argument(
        "--baseline-rules-path",
        default="artifacts/recommendation_outputs/cross_sell_association_rules_baseline.parquet",
    )
    parser.add_argument(
        "--baseline-metrics-path",
        default="artifacts/metrics/cross_sell_association_rules_baseline_metrics.json",
    )

    parser.add_argument(
        "--best-model-path",
        default="artifacts/models/cross_sell_association_rules_best.pkl",
    )
    parser.add_argument(
        "--best-rules-path",
        default="artifacts/recommendation_outputs/cross_sell_association_rules_best.parquet",
    )
    parser.add_argument(
        "--tuning-results-path",
        default="artifacts/metrics/cross_sell_association_rules_tuning_results.csv",
    )
    parser.add_argument(
        "--best-metrics-path",
        default="artifacts/metrics/cross_sell_association_rules_best_metrics.json",
    )

    parser.add_argument("--min-item-support", type=int, default=20)
    parser.add_argument("--min-pair-support", type=int, default=5)
    parser.add_argument("--score-formula", default="confidence_lift")
    parser.add_argument("--max-basket-size-for-pairs", type=int, default=30)
    parser.add_argument("--top-n-rules-per-item", type=int, default=200)
    parser.add_argument("--recommend-k", type=int, default=12)
    parser.add_argument("--eval-sample-size", type=int, default=100_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    if args.mode == "baseline":
        train_baseline(args)
    else:
        tune_model(args)


if __name__ == "__main__":
    main()
