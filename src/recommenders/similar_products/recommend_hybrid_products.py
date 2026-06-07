import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.recommenders.similar_products.hybrid_recommender import (
    HybridSimilarProductsRecommender,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get similar product recommendations using the trained hybrid recommender."
    )

    parser.add_argument(
        "--article-id",
        type=int,
        required=True,
        help="Article ID to get recommendations for.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to return.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/similarproducts/hybrid",
        help="Path to the saved hybrid recommender artifacts.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = PROJECT_ROOT / args.model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}\n"
            "Please train the model first by running:\n"
            "python src/recommenders/similar_products/train_hybrid_recommender.py"
        )

    print("Loading hybrid recommender...")
    recommender = HybridSimilarProductsRecommender.load(model_path)

    print(f"Getting recommendations for article_id={args.article_id}...")
    query_product, recommendations = recommender.recommend(
        article_id=args.article_id,
        top_k=args.top_k,
    )

    print("\nQuery Product:")
    print(query_product.to_string(index=False))

    print("\nRecommendations:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()