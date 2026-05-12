from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommenders.cross_sell.cooccurrence_recommender import (  # noqa: E402
    CrossSellCooccurrenceRecommender,
)


DEFAULT_MODEL_PATH = Path("artifacts/models/cross_sell_cooccurrence.pkl")
DEFAULT_ARTICLES_PATH = Path("data/bronze/articles.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect cross-sell recommendations with article metadata."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--item", default="0706016001")
    parser.add_argument("--basket", nargs="*", default=["0706016001", "0372860001"])
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def load_article_metadata(path: Path) -> pd.DataFrame:
    articles = pd.read_csv(path, dtype={"article_id": str})
    articles["article_id"] = articles["article_id"].str.zfill(10)
    return articles.set_index("article_id")[
        [
            "prod_name",
            "product_type_name",
            "product_group_name",
            "colour_group_name",
            "index_name",
        ]
    ]


def get_product_label(metadata: pd.DataFrame, item_id: str) -> dict[str, str]:
    normalized_item_id = str(item_id).zfill(10)
    if normalized_item_id not in metadata.index:
        return {
            "item_id": normalized_item_id,
            "prod_name": "UNKNOWN",
            "product_type_name": "",
            "product_group_name": "",
            "colour_group_name": "",
            "index_name": "",
        }

    row = metadata.loc[normalized_item_id]
    return {
        "item_id": normalized_item_id,
        "prod_name": row["prod_name"],
        "product_type_name": row["product_type_name"],
        "product_group_name": row["product_group_name"],
        "colour_group_name": row["colour_group_name"],
        "index_name": row["index_name"],
    }


def recommendation_rows(
    metadata: pd.DataFrame,
    recommendations: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for rank, rec in enumerate(recommendations, start=1):
        product = get_product_label(metadata, rec["item_id"])
        rows.append(
            {
                "rank": rank,
                "recommended_item_id": product["item_id"],
                "prod_name": product["prod_name"],
                "type": product["product_type_name"],
                "color": product["colour_group_name"],
                "score": round(rec["score"], 4),
                "confidence_%": round(rec["confidence"] * 100, 2),
                "lift": round(rec["lift"], 2),
                "together_count": rec["cooccurrence_count"],
                "supporting_items": ",".join(rec.get("supporting_items", [])),
                "reason": rec["reason"],
            }
        )
    return pd.DataFrame(rows)


def print_source_item(metadata: pd.DataFrame, item_id: str) -> None:
    product = get_product_label(metadata, item_id)
    print(
        f"{product['item_id']} | {product['prod_name']} | "
        f"{product['product_type_name']} | {product['colour_group_name']} | "
        f"{product['index_name']}"
    )


def main() -> None:
    args = parse_args()
    model = CrossSellCooccurrenceRecommender.load(args.model)
    metadata = load_article_metadata(args.articles)

    print("\n=== Single Item Recommendation ===")
    print("Source:")
    print_source_item(metadata, args.item)
    item_recommendations = model.recommend(args.item, n=args.top_n)
    print(recommendation_rows(metadata, item_recommendations).to_string(index=False))

    print("\n=== Basket Recommendation ===")
    print("Basket:")
    for item_id in args.basket:
        print_source_item(metadata, item_id)
    basket_recommendations = model.recommend_for_basket(args.basket, n=args.top_n)
    print(recommendation_rows(metadata, basket_recommendations).to_string(index=False))


if __name__ == "__main__":
    main()
