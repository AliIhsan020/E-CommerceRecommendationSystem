from __future__ import annotations

from collections import defaultdict
import json
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

ARTICLES_PATH = ROOT / "data" / "bronze" / "articles.csv"
RULES_PATH = (
    ROOT
    / "artifacts"
    / "recommendation_outputs"
    / "cross_sell_association_rules_best_with_product_names.parquet"
)
MODEL_PATH = ROOT / "artifacts" / "models" / "cross_sell_association_rules_best.pkl"
TEST_BASKETS_PATH = ROOT / "data" / "gold" / "cross_sell_test_baskets.parquet"
METRICS_PATH = ROOT / "artifacts" / "metrics" / "cross_sell_association_rules_best_metrics.json"
SPLIT_PATH = ROOT / "artifacts" / "metrics" / "cross_sell_split_summary.json"
OUTPUT_PATH = REPORTS_DIR / "cross_sell_presentation_data.json"


def normalize_articles(value) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(article) for article in value if pd.notna(article)]
    return [str(value)]


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def recommend_for_basket(articles: list[str], model: dict, k: int = 12) -> list[str]:
    context = unique_preserve_order(normalize_articles(articles))
    seen = set(context)
    candidate_scores = defaultdict(float)

    for article in context:
        for rule in model["rules_by_item"].get(article, []):
            candidate = rule["consequent"]
            if candidate not in seen:
                candidate_scores[candidate] += rule["score"]

    ranked = [
        article
        for article, _ in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    ]

    for article in model["popular_items"]:
        if len(ranked) >= k:
            break
        if article not in seen and article not in candidate_scores:
            ranked.append(article)

    return ranked[:k]


def label_items(article_ids: list[str], names: dict[str, str], max_items: int | None = None) -> str:
    selected = article_ids if max_items is None else article_ids[:max_items]
    labels = [f"{article_id} - {names.get(article_id, '')}" for article_id in selected]
    if max_items is not None and len(article_ids) > max_items:
        labels.append(f"... +{len(article_ids) - max_items} more")
    return "\n".join(labels)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    split_summary = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))

    article_names = (
        pd.read_csv(ARTICLES_PATH, dtype={"article_id": str}, usecols=["article_id", "prod_name"])
        .drop_duplicates("article_id")
        .assign(prod_name=lambda df: df["prod_name"].fillna("").astype(str))
    )
    names = dict(zip(article_names["article_id"], article_names["prod_name"]))

    rules = pd.read_parquet(RULES_PATH)
    top_rules = (
        rules.nlargest(300, "score")[
            [
                "antecedent_id",
                "antecedent_name",
                "consequent_id",
                "consequent_name",
                "pair_count",
                "confidence",
                "lift",
                "jaccard",
                "score",
            ]
        ]
        .copy()
    )
    top_rules["confidence_pct"] = top_rules["confidence"] * 100
    top_rules = top_rules[
        [
            "antecedent_id",
            "antecedent_name",
            "consequent_id",
            "consequent_name",
            "pair_count",
            "confidence_pct",
            "lift",
            "jaccard",
            "score",
        ]
    ]

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    test_baskets = pd.read_parquet(TEST_BASKETS_PATH)
    sample_rows = []
    for row in test_baskets.itertuples(index=False):
        articles = unique_preserve_order(normalize_articles(row.articles))
        if len(articles) < 2:
            continue

        split_idx = max(1, len(articles) // 2)
        context = articles[:split_idx]
        target = articles[split_idx:]
        recs = recommend_for_basket(context, model, k=12)
        hits = sorted(set(recs) & set(target))

        if not hits:
            continue

        sample_rows.append(
            {
                "customer_id": str(row.customer_id)[:16] + "...",
                "basket_size": len(articles),
                "model_input_context": label_items(context, names, max_items=5),
                "hidden_target_for_eval": label_items(target, names, max_items=5),
                "recommendations_top12": label_items(recs, names, max_items=12),
                "hits": label_items(hits, names, max_items=5),
                "hit_count": len(hits),
            }
        )

        if len(sample_rows) >= 12:
            break

    model_params = metrics.get("best_params", {})
    summary_rows = [
        ["Model", "cross_sell_association_rules"],
        ["Yaklaşım", "Association rules / item-to-item co-occurrence"],
        ["Best params", ", ".join(f"{key}={value}" for key, value in model_params.items())],
        ["Train tarih aralığı", f"{split_summary['train_min_date']} - {split_summary['train_max_date']}"],
        ["Test tarih aralığı", f"{split_summary['test_min_date']} - {split_summary['test_max_date']}"],
        ["Train sepet sayısı", split_summary["train_rows"]],
        ["Test sepet sayısı", split_summary["test_rows"]],
        ["Toplam rule sayısı", int(len(rules))],
        ["Ürün adı eşleşmeyen rule", int(rules["antecedent_name"].isna().sum() + rules["consequent_name"].isna().sum())],
        ["Evaluated baskets", metrics["evaluated_baskets"]],
        ["precision@12", metrics["precision_at_12"]],
        ["recall@12", metrics["recall_at_12"]],
        ["hit_rate@12", metrics["hit_rate_at_12"]],
    ]

    chart_rows = [
        ["Metric", "Value"],
        ["precision@12", metrics["precision_at_12"]],
        ["recall@12", metrics["recall_at_12"]],
        ["hit_rate@12", metrics["hit_rate_at_12"]],
    ]

    payload = {
        "summary_rows": summary_rows,
        "chart_rows": chart_rows,
        "top_rules": top_rules.to_dict(orient="records"),
        "sample_recommendations": sample_rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "top_rules": len(top_rules), "samples": len(sample_rows)}, indent=2))


if __name__ == "__main__":
    main()
