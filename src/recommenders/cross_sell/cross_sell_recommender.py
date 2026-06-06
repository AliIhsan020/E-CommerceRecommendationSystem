from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable
import pickle

import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = Path("artifacts/models/cross_sell_association_rules_best.pkl")
DEFAULT_ARTICLES_PATH = Path("data/bronze/articles.csv")


class CrossSellAssociationRulesRecommender:
    MODEL_NAME = "cross_sell_association_rules"

    def __init__(
        self,
        min_item_support: int = 20,
        min_pair_support: int = 5,
        score_formula: str = "confidence_lift",
        max_basket_size_for_pairs: int = 30,
        top_n_rules_per_item: int = 200,
        recommend_k: int = 12,
    ) -> None:
        self.min_item_support = min_item_support
        self.min_pair_support = min_pair_support
        self.score_formula = score_formula
        self.max_basket_size_for_pairs = max_basket_size_for_pairs
        self.top_n_rules_per_item = top_n_rules_per_item
        self.recommend_k = recommend_k

        self.item_support_: pd.Series | None = None
        self.rules_: pd.DataFrame | None = None
        self.rules_by_item_: dict[str, list[dict[str, Any]]] | None = None
        self.popular_items_: list[str] | None = None
        self.metrics_: dict[str, Any] | None = None
        self.n_baskets_: int | None = None
        self.used_basket_count_: int | None = None

    @staticmethod
    def normalize_articles(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            return [str(article) for article in value if pd.notna(article)]
        return [str(value)]

    @staticmethod
    def unique_preserve_order(values: Iterable[str]) -> list[str]:
        seen = set()
        output = []
        for value in values:
            if value not in seen:
                seen.add(value)
                output.append(value)
        return output

    @staticmethod
    def context_target_split(articles: Any) -> tuple[list[str] | None, set[str] | None]:
        articles = CrossSellAssociationRulesRecommender.unique_preserve_order(
            CrossSellAssociationRulesRecommender.normalize_articles(articles)
        )
        if len(articles) < 2:
            return None, None
        split_idx = max(1, len(articles) // 2)
        return articles[:split_idx], set(articles[split_idx:])

    @staticmethod
    def _articles_series(frame: pd.DataFrame, articles_col: str) -> pd.Series:
        if articles_col in frame.columns:
            return frame[articles_col]
        if "element" in frame.columns:
            return frame["element"]
        raise KeyError(f"Articles column not found: {articles_col}")

    @staticmethod
    def count_item_support(baskets: pd.DataFrame, articles_col: str = "articles") -> pd.Series:
        articles = CrossSellAssociationRulesRecommender._articles_series(
            baskets, articles_col
        ).apply(CrossSellAssociationRulesRecommender.normalize_articles)
        return articles.explode().value_counts().astype("int64")

    @classmethod
    def count_pair_counts(
        cls,
        articles: Iterable[Any],
        item_support: pd.Series,
        min_item_support: int,
        max_basket_size_for_pairs: int,
    ) -> tuple[Counter[tuple[str, str]], int]:
        frequent_items = set(item_support[item_support >= min_item_support].index)
        pair_counts: Counter[tuple[str, str]] = Counter()
        used_basket_count = 0

        for value in articles:
            basket = sorted(
                set(article for article in cls.normalize_articles(value) if article in frequent_items)
            )
            if len(basket) < 2 or len(basket) > max_basket_size_for_pairs:
                continue
            used_basket_count += 1
            pair_counts.update(combinations(basket, 2))

        return pair_counts, used_basket_count

    def fit(self, baskets: pd.DataFrame, articles_col: str = "articles") -> "CrossSellAssociationRulesRecommender":
        articles = self._articles_series(baskets, articles_col).apply(self.normalize_articles)
        item_support = articles.explode().value_counts().astype("int64")
        popular_items = item_support.index.tolist()
        pair_counts, used_basket_count = self.count_pair_counts(
            articles=articles,
            item_support=item_support,
            min_item_support=self.min_item_support,
            max_basket_size_for_pairs=self.max_basket_size_for_pairs,
        )
        return self.fit_from_counts(
            pair_counts=pair_counts,
            item_support=item_support,
            n_baskets=len(articles),
            popular_items=popular_items,
            used_basket_count=used_basket_count,
        )

    def fit_from_counts(
        self,
        pair_counts: Counter[tuple[str, str]],
        item_support: pd.Series,
        n_baskets: int,
        popular_items: list[str] | None = None,
        used_basket_count: int | None = None,
    ) -> "CrossSellAssociationRulesRecommender":
        self.item_support_ = item_support.astype("int64")
        self.popular_items_ = popular_items or self.item_support_.index.tolist()
        self.n_baskets_ = int(n_baskets)
        self.used_basket_count_ = used_basket_count
        self.rules_ = self._build_rules_dataframe(pair_counts)
        self.rules_by_item_ = self._rules_to_lookup(self.rules_)
        return self

    def _calculate_score(
        self,
        confidence: float,
        lift: float,
        jaccard: float,
        pair_count: int,
    ) -> float:
        if self.score_formula == "confidence_support":
            return confidence * np.log1p(pair_count)
        if self.score_formula == "jaccard_lift":
            return jaccard * np.log1p(lift) * np.log1p(pair_count)
        return confidence * np.log1p(lift) * np.log1p(pair_count)

    def _build_rules_dataframe(
        self,
        pair_counts: Counter[tuple[str, str]],
    ) -> pd.DataFrame:
        if self.item_support_ is None or self.n_baskets_ is None:
            raise ValueError("item_support_ and n_baskets_ must be set before building rules.")

        rows = []
        for (left, right), pair_count in pair_counts.items():
            if pair_count < self.min_pair_support:
                continue

            left_count = int(self.item_support_.get(left, 0))
            right_count = int(self.item_support_.get(right, 0))
            if left_count < self.min_item_support or right_count < self.min_item_support:
                continue

            for antecedent, consequent, antecedent_count, consequent_count in [
                (left, right, left_count, right_count),
                (right, left, right_count, left_count),
            ]:
                support = pair_count / self.n_baskets_
                confidence = pair_count / antecedent_count
                consequent_support = consequent_count / self.n_baskets_
                lift = confidence / consequent_support if consequent_support > 0 else 0
                jaccard = pair_count / (antecedent_count + consequent_count - pair_count)
                score = self._calculate_score(confidence, lift, jaccard, pair_count)

                rows.append(
                    {
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "pair_count": int(pair_count),
                        "antecedent_count": antecedent_count,
                        "consequent_count": consequent_count,
                        "support": support,
                        "confidence": confidence,
                        "lift": lift,
                        "jaccard": jaccard,
                        "score": score,
                    }
                )

        rules = pd.DataFrame(rows)
        if rules.empty:
            return rules

        rules = rules.sort_values(
            ["antecedent", "score", "confidence", "lift", "pair_count"],
            ascending=[True, False, False, False, False],
        )
        rules = rules.groupby("antecedent", as_index=False).head(
            self.top_n_rules_per_item
        )
        return rules.reset_index(drop=True)

    @staticmethod
    def _rules_to_lookup(rules: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        lookup: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        if rules.empty:
            return dict(lookup)

        for row in rules.itertuples(index=False):
            lookup[row.antecedent].append(
                {
                    "consequent": row.consequent,
                    "score": float(row.score),
                    "confidence": float(row.confidence),
                    "lift": float(row.lift),
                    "support": float(row.support),
                    "pair_count": int(row.pair_count),
                }
            )
        return dict(lookup)

    def recommend(self, articles: Any, top_k: int | None = None) -> list[str]:
        if self.rules_by_item_ is None or self.popular_items_ is None:
            raise ValueError("Model is not fitted. Call fit() or load_model() first.")

        k = top_k or self.recommend_k
        context = self.unique_preserve_order(self.normalize_articles(articles))
        seen = set(context)
        candidate_scores: defaultdict[str, float] = defaultdict(float)

        for article in context:
            for rule in self.rules_by_item_.get(article, []):
                candidate = rule["consequent"]
                if candidate not in seen:
                    candidate_scores[candidate] += rule["score"]

        ranked = [
            article
            for article, _ in sorted(
                candidate_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]

        for article in self.popular_items_:
            if len(ranked) >= k:
                break
            if article not in seen and article not in candidate_scores:
                ranked.append(article)

        return ranked[:k]

    def evaluate(
        self,
        test_baskets: pd.DataFrame,
        top_k: int | None = None,
        articles_col: str = "articles",
    ) -> dict[str, float | int]:
        k = top_k or self.recommend_k
        articles_series = self._articles_series(test_baskets, articles_col)
        precisions = []
        recalls = []
        hit_rates = []

        for articles in articles_series:
            context, target = self.context_target_split(articles)
            if not context or not target:
                continue

            recommendations = self.recommend(context, top_k=k)
            hits = len(set(recommendations) & target)
            precisions.append(hits / k)
            recalls.append(hits / len(target))
            hit_rates.append(1.0 if hits > 0 else 0.0)

        metrics = {
            "evaluated_baskets": int(len(precisions)),
            f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
            f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
            f"hit_rate_at_{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        }
        self.metrics_ = metrics
        return metrics

    def get_params(self) -> dict[str, Any]:
        return {
            "min_item_support": self.min_item_support,
            "min_pair_support": self.min_pair_support,
            "score_formula": self.score_formula,
            "max_basket_size_for_pairs": self.max_basket_size_for_pairs,
            "top_n_rules_per_item": self.top_n_rules_per_item,
            "recommend_k": self.recommend_k,
        }

    def to_artifact(self) -> dict[str, Any]:
        if self.rules_by_item_ is None or self.popular_items_ is None:
            raise ValueError("Model is not fitted. Call fit() before saving.")

        artifact = {
            "model_name": self.MODEL_NAME,
            "params": self.get_params(),
            "rules_by_item": self.rules_by_item_,
            "popular_items": self.popular_items_,
        }
        if self.metrics_ is not None:
            artifact["metrics"] = self.metrics_
        return artifact

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.to_artifact(), f)

    @classmethod
    def load_model(cls, path: str | Path) -> "CrossSellAssociationRulesRecommender":
        with Path(path).open("rb") as f:
            artifact = pickle.load(f)

        if isinstance(artifact, cls):
            return artifact

        params = artifact.get("params", {})
        model = cls(
            min_item_support=params.get("min_item_support", 20),
            min_pair_support=params.get("min_pair_support", 5),
            score_formula=params.get("score_formula", "confidence_lift"),
            max_basket_size_for_pairs=params.get("max_basket_size_for_pairs", 30),
            top_n_rules_per_item=params.get("top_n_rules_per_item", 200),
            recommend_k=params.get("recommend_k", 12),
        )
        model.rules_by_item_ = artifact.get("rules_by_item", {})
        model.popular_items_ = artifact.get("popular_items", [])
        model.metrics_ = artifact.get("metrics")
        return model

    def save_rules(self, path: str | Path) -> None:
        if self.rules_ is None:
            raise ValueError("Rules are not available. Fit the model before saving rules.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rules_.to_parquet(path, index=False)


CrossSellRecommender = CrossSellAssociationRulesRecommender


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "requirements.txt").exists() and (path / "data").exists():
            return path
    raise FileNotFoundError("Project root could not be found.")


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def normalize_input_articles(values: Iterable[str]) -> list[str]:
    articles = []
    for value in values:
        for article in str(value).split(","):
            article = article.strip()
            if article:
                articles.append(article)
    return articles


def load_article_name_lookup(articles_path: Path) -> dict[str, str]:
    articles = pd.read_csv(
        articles_path,
        dtype={"article_id": str},
        usecols=["article_id", "prod_name"],
    )
    articles["prod_name"] = articles["prod_name"].fillna("").astype(str)
    return dict(zip(articles["article_id"], articles["prod_name"]))


def recommend_cross_sell(
    basket_articles: list[str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    articles_path: str | Path = DEFAULT_ARTICLES_PATH,
    top_k: int = 12,
    include_names: bool = True,
) -> pd.DataFrame:
    project_root = find_project_root(Path.cwd().resolve())
    model_path = project_path(project_root, model_path)
    articles_path = project_path(project_root, articles_path)

    model = CrossSellAssociationRulesRecommender.load_model(model_path)
    recommended_ids = model.recommend(basket_articles, top_k=top_k)

    result = pd.DataFrame(
        {
            "rank": range(1, len(recommended_ids) + 1),
            "recommended_article_id": recommended_ids,
        }
    )

    if include_names:
        name_lookup = load_article_name_lookup(articles_path)
        result["recommended_prod_name"] = result["recommended_article_id"].map(
            name_lookup
        )

    return result


def print_recommendations(
    basket_articles: list[str],
    model_path: str | Path,
    articles_path: str | Path,
    top_k: int,
    include_names: bool,
) -> None:
    recommendations = recommend_cross_sell(
        basket_articles=basket_articles,
        model_path=model_path,
        articles_path=articles_path,
        top_k=top_k,
        include_names=include_names,
    )

    print()
    print("Basket article ids:")
    print(", ".join(basket_articles))
    print()
    print(recommendations.to_string(index=False))
    print()


def interactive_terminal(
    model_path: str | Path,
    articles_path: str | Path,
    top_k: int,
    include_names: bool,
) -> None:
    print("Cross-sell recommendation terminal")
    print("Article id gir. Birden fazla id icin bosluk veya virgul kullan.")
    print("Cikmak icin q, quit veya exit yaz.")
    print()

    while True:
        raw_value = input("Sepetteki article_id: ").strip()
        if raw_value.lower() in {"q", "quit", "exit"}:
            print("Cikis yapildi.")
            return

        basket_articles = normalize_input_articles([raw_value])
        if not basket_articles:
            print("En az bir article_id girmelisin.")
            continue

        print_recommendations(
            basket_articles=basket_articles,
            model_path=model_path,
            articles_path=articles_path,
            top_k=top_k,
            include_names=include_names,
        )


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Recommend cross-sell products for product ids in a basket."
    )
    parser.add_argument(
        "articles",
        nargs="*",
        help="Product ids in the basket. Example: 0108775015 0538699001",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--articles-path", default=str(DEFAULT_ARTICLES_PATH))
    parser.add_argument(
        "--no-names",
        action="store_true",
        help="Print only recommended ids, without product names.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    basket_articles = normalize_input_articles(args.articles)

    if not basket_articles:
        interactive_terminal(
            model_path=args.model_path,
            articles_path=args.articles_path,
            top_k=args.top_k,
            include_names=not args.no_names,
        )
        return

    print_recommendations(
        basket_articles=basket_articles,
        model_path=args.model_path,
        articles_path=args.articles_path,
        top_k=args.top_k,
        include_names=not args.no_names,
    )


if __name__ == "__main__":
    main()
