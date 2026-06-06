import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, normalize


class HybridSimilarProductsRecommender:
    def __init__(
        self,
        categorical_cols=None,
        categorical_weight=1.5,
        text_weight=1.0,
        max_features=50000,
        candidate_pool=200,
    ):
        self.categorical_cols = categorical_cols or [
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
            "department_name",
            "index_name",
            "index_group_name",
            "section_name",
            "garment_group_name",
        ]

        self.categorical_weight = categorical_weight
        self.text_weight = text_weight
        self.max_features = max_features
        self.candidate_pool = candidate_pool

        self.categorical_encoder = None
        self.tfidf_vectorizer = None
        self.nn_model = None
        self.hybrid_matrix = None
        self.articles_index = None
        self.article_id_to_idx = None

    def _create_one_hot_encoder(self):
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,
            )

    def _validate_articles(self, articles):
        required_cols = [
            "article_id",
            "product_code",
            "prod_name",
            "detail_desc",
        ] + self.categorical_cols

        missing_cols = [col for col in required_cols if col not in articles.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if articles["article_id"].duplicated().any():
            raise ValueError("article_id column contains duplicate values.")

    def fit(self, articles):
        self._validate_articles(articles)

        articles = articles.copy().reset_index(drop=True)

        combined_text = (
            articles["prod_name"].astype(str)
            + " "
            + articles["detail_desc"].astype(str)
        )

        self.categorical_encoder = self._create_one_hot_encoder()
        categorical_matrix = self.categorical_encoder.fit_transform(
            articles[self.categorical_cols]
        )

        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=self.max_features,
        )

        text_matrix = self.tfidf_vectorizer.fit_transform(combined_text)

        categorical_matrix_norm = normalize(
            categorical_matrix,
            norm="l2",
            axis=1,
        )

        text_matrix_norm = normalize(
            text_matrix,
            norm="l2",
            axis=1,
        )

        self.hybrid_matrix = hstack(
            [
                categorical_matrix_norm * self.categorical_weight,
                text_matrix_norm * self.text_weight,
            ],
            format="csr",
        )

        self.nn_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
        )

        self.nn_model.fit(self.hybrid_matrix)

        index_cols = [
            "article_id",
            "product_code",
            "prod_name",
            "product_type_name",
            "colour_group_name",
            "section_name",
            "garment_group_name",
        ]

        self.articles_index = articles[index_cols].copy()
        self.article_id_to_idx = {
            article_id: idx
            for idx, article_id in enumerate(self.articles_index["article_id"])
        }

        return self

    def recommend(self, article_id, top_k=10, candidate_pool=None):
        if self.nn_model is None or self.hybrid_matrix is None:
            raise ValueError("Model is not fitted. Call fit() or load() first.")

        if article_id not in self.article_id_to_idx:
            raise ValueError(f"article_id not found: {article_id}")

        query_idx = self.article_id_to_idx[article_id]
        query_product_code = self.articles_index.loc[query_idx, "product_code"]

        candidate_pool = candidate_pool or self.candidate_pool
        n_neighbors = min(candidate_pool, self.hybrid_matrix.shape[0])

        distances, indices = self.nn_model.kneighbors(
            self.hybrid_matrix[query_idx],
            n_neighbors=n_neighbors,
        )

        filtered_neighbors = []
        seen_product_codes = set()

        for idx, dist in zip(indices[0], distances[0]):
            candidate_product_code = self.articles_index.loc[idx, "product_code"]

            if idx == query_idx:
                continue

            if candidate_product_code == query_product_code:
                continue

            if candidate_product_code in seen_product_codes:
                continue

            filtered_neighbors.append((idx, dist))
            seen_product_codes.add(candidate_product_code)

            if len(filtered_neighbors) == top_k:
                break

        neighbor_indices = [idx for idx, _ in filtered_neighbors]
        neighbor_distances = [dist for _, dist in filtered_neighbors]

        query_product = self.articles_index.iloc[[query_idx]].copy()

        recommendations = self.articles_index.iloc[neighbor_indices].copy()
        recommendations["rank"] = range(1, len(recommendations) + 1)
        recommendations["cosine_similarity"] = [
            1 - dist for dist in neighbor_distances
        ]

        return query_product, recommendations

    def save(self, path):
        if self.nn_model is None or self.hybrid_matrix is None:
            raise ValueError("Model is not fitted. Call fit() before save().")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.categorical_encoder,
            path / "hybrid_categorical_encoder.joblib",
        )

        joblib.dump(
            self.tfidf_vectorizer,
            path / "hybrid_tfidf_vectorizer.joblib",
        )

        joblib.dump(
            self.nn_model,
            path / "hybrid_nn_model.joblib",
        )

        save_npz(
            path / "hybrid_matrix.npz",
            self.hybrid_matrix,
        )

        self.articles_index.to_parquet(
            path / "hybrid_articles_index.parquet",
            index=False,
        )

        config = {
            "categorical_cols": self.categorical_cols,
            "categorical_weight": self.categorical_weight,
            "text_weight": self.text_weight,
            "max_features": self.max_features,
            "candidate_pool": self.candidate_pool,
        }

        with open(path / "hybrid_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, path):
        path = Path(path)

        with open(path / "hybrid_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        recommender = cls(**config)

        recommender.categorical_encoder = joblib.load(
            path / "hybrid_categorical_encoder.joblib"
        )

        recommender.tfidf_vectorizer = joblib.load(
            path / "hybrid_tfidf_vectorizer.joblib"
        )

        recommender.nn_model = joblib.load(
            path / "hybrid_nn_model.joblib"
        )

        recommender.hybrid_matrix = load_npz(
            path / "hybrid_matrix.npz"
        ).tocsr()

        recommender.articles_index = pd.read_parquet(
            path / "hybrid_articles_index.parquet"
        )

        recommender.article_id_to_idx = {
            article_id: idx
            for idx, article_id in enumerate(
                recommender.articles_index["article_id"]
            )
        }

        return recommender