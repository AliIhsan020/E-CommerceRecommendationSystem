from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.recommenders.similar_products.hybrid_recommender import (
    HybridSimilarProductsRecommender,
)


def main():
    articles_path = PROJECT_ROOT / "data/gold/articles.parquet"
    model_path = PROJECT_ROOT / "artifacts/models/similarproducts/hybrid"

    print("Loading articles data...")
    articles = pd.read_parquet(articles_path)
    print(f"Articles shape: {articles.shape}")

    recommender = HybridSimilarProductsRecommender(
        categorical_weight=1.5,
        text_weight=1.0,
        max_features=50000,
        candidate_pool=200,
    )

    print("Training hybrid recommender...")
    recommender.fit(articles)

    print("Saving hybrid recommender...")
    recommender.save(model_path)

    print(f"Hybrid recommender saved to: {model_path}")


if __name__ == "__main__":
    main()