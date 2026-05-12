from __future__ import annotations

import ast
import math
import pickle
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


class CrossSellCooccurrenceRecommender:
    """
    Item-to-item cross-sell recommender trained from basket data only.

    The model counts item pairs that appear in the same basket and ranks target
    items by directional confidence, lift, and co-occurrence strength.
    """

    def __init__(
        self,
        min_pair_count: int = 5,
        top_k_per_item: int = 50,
        max_basket_size: int | None = 50,
        confidence_shrinkage: float = 20.0,
        lift_weight: float = 0.35,
        count_weight: float = 0.35,
        popularity_penalty_weight: float = 0.15,
    ) -> None:
        if min_pair_count < 1:
            raise ValueError("min_pair_count must be at least 1")
        if top_k_per_item < 1:
            raise ValueError("top_k_per_item must be at least 1")
        if max_basket_size is not None and max_basket_size < 2:
            raise ValueError("max_basket_size must be None or at least 2")
        if confidence_shrinkage < 0:
            raise ValueError("confidence_shrinkage must be non-negative")
        if lift_weight < 0 or count_weight < 0 or popularity_penalty_weight < 0:
            raise ValueError("score weights must be non-negative")

        self.min_pair_count = min_pair_count
        self.top_k_per_item = top_k_per_item
        self.max_basket_size = max_basket_size
        self.confidence_shrinkage = confidence_shrinkage
        self.lift_weight = lift_weight
        self.count_weight = count_weight
        self.popularity_penalty_weight = popularity_penalty_weight
        self.total_baskets = 0
        self.item_counts: Counter[str] = Counter()
        self.recommendations_by_item: dict[str, list[dict[str, Any]]] = {}
        self.training_summary: dict[str, Any] = {}

    def fit(self, baskets: Iterable[Any]) -> "CrossSellCooccurrenceRecommender":
        pair_counts: Counter[tuple[str, str]] = Counter()
        total_baskets = 0
        skipped_small_baskets = 0
        skipped_large_baskets = 0

        for raw_basket in baskets:
            items = self.normalize_basket(raw_basket)

            if len(items) < 2:
                skipped_small_baskets += 1
                continue
            if self.max_basket_size is not None and len(items) > self.max_basket_size:
                skipped_large_baskets += 1
                continue

            total_baskets += 1
            self.item_counts.update(items)

            for item_a, item_b in combinations(sorted(items), 2):
                pair_counts[(item_a, item_b)] += 1

        self.total_baskets = total_baskets
        self.recommendations_by_item = self._build_recommendations(
            pair_counts=pair_counts,
            total_baskets=total_baskets,
        )
        self.training_summary = {
            "total_baskets": total_baskets,
            "unique_items": len(self.item_counts),
            "unique_pairs": len(pair_counts),
            "retained_items": len(self.recommendations_by_item),
            "min_pair_count": self.min_pair_count,
            "top_k_per_item": self.top_k_per_item,
            "max_basket_size": self.max_basket_size,
            "confidence_shrinkage": self.confidence_shrinkage,
            "lift_weight": self.lift_weight,
            "count_weight": self.count_weight,
            "popularity_penalty_weight": self.popularity_penalty_weight,
            "skipped_small_baskets": skipped_small_baskets,
            "skipped_large_baskets": skipped_large_baskets,
        }
        return self

    def recommend(
        self,
        item_id: str,
        n: int = 10,
        exclude_items: Iterable[str] | None = None,
        use_fallback: bool = True,
    ) -> list[dict[str, Any]]:
        source_item = str(item_id)
        excluded = {str(item) for item in exclude_items or []}
        excluded.add(source_item)

        recommendations = [
            rec
            for rec in self.recommendations_by_item.get(source_item, [])
            if rec["item_id"] not in excluded
        ]
        if use_fallback and len(recommendations) < n:
            excluded.update(rec["item_id"] for rec in recommendations)
            recommendations.extend(
                self._popular_fallback_recommendations(
                    excluded_items=excluded,
                    n=n - len(recommendations),
                )
            )
        return recommendations[:n]

    def recommend_for_basket(
        self,
        item_ids: Iterable[str],
        n: int = 10,
        use_fallback: bool = True,
    ) -> list[dict[str, Any]]:
        source_items = {str(item_id) for item_id in item_ids}
        aggregated: dict[str, dict[str, Any]] = {}

        for source_item in source_items:
            for rec in self.recommendations_by_item.get(source_item, []):
                target_item = rec["item_id"]
                if target_item in source_items:
                    continue

                current = aggregated.setdefault(
                    target_item,
                    {
                        "item_id": target_item,
                        "score": 0.0,
                        "confidence": 0.0,
                        "lift": 0.0,
                        "cooccurrence_count": 0,
                        "supporting_items": [],
                        "reason": "basket_cooccurrence",
                    },
                )
                current["score"] += rec["score"]
                current["confidence"] = max(current["confidence"], rec["confidence"])
                current["lift"] = max(current["lift"], rec["lift"])
                current["cooccurrence_count"] += rec["cooccurrence_count"]
                current["supporting_items"].append(source_item)

        ranked = sorted(
            aggregated.values(),
            key=lambda rec: (
                rec["score"],
                rec["confidence"],
                rec["cooccurrence_count"],
            ),
            reverse=True,
        )
        if use_fallback and len(ranked) < n:
            excluded = source_items | {rec["item_id"] for rec in ranked}
            ranked.extend(
                self._popular_fallback_recommendations(
                    excluded_items=excluded,
                    n=n - len(ranked),
                )
            )
        return ranked[:n]

    def get_popular_items(self, n: int = 20) -> list[dict[str, Any]]:
        return [
            {"item_id": item_id, "basket_count": basket_count}
            for item_id, basket_count in self.item_counts.most_common(n)
        ]

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "CrossSellCooccurrenceRecommender":
        with Path(path).open("rb") as file:
            try:
                model = pickle.load(file)
            except ModuleNotFoundError as exc:
                if exc.name != "cooccurrence_recommender":
                    raise

                module_dir = Path(__file__).resolve().parent
                sys.path.insert(0, str(module_dir))
                try:
                    file.seek(0)
                    model = pickle.load(file)
                finally:
                    sys.path.remove(str(module_dir))

        if not isinstance(model, cls):
            if type(model).__name__ != cls.__name__:
                raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")

            converted_model = cls(
                min_pair_count=getattr(model, "min_pair_count", 5),
                top_k_per_item=getattr(model, "top_k_per_item", 50),
                max_basket_size=getattr(model, "max_basket_size", 50),
                confidence_shrinkage=getattr(model, "confidence_shrinkage", 20.0),
                lift_weight=getattr(model, "lift_weight", 0.35),
                count_weight=getattr(model, "count_weight", 0.35),
                popularity_penalty_weight=getattr(
                    model,
                    "popularity_penalty_weight",
                    0.15,
                ),
            )
            converted_model.__dict__.update(model.__dict__)
            model = converted_model
        return model

    def _build_recommendations(
        self,
        pair_counts: Counter[tuple[str, str]],
        total_baskets: int,
    ) -> dict[str, list[dict[str, Any]]]:
        if total_baskets == 0:
            return {}

        recommendations: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for (item_a, item_b), pair_count in pair_counts.items():
            if pair_count < self.min_pair_count:
                continue

            recommendations[item_a].append(
                self._make_directional_recommendation(
                    source_item=item_a,
                    target_item=item_b,
                    pair_count=pair_count,
                    total_baskets=total_baskets,
                )
            )
            recommendations[item_b].append(
                self._make_directional_recommendation(
                    source_item=item_b,
                    target_item=item_a,
                    pair_count=pair_count,
                    total_baskets=total_baskets,
                )
            )

        return {
            item_id: sorted(
                item_recommendations,
                key=lambda rec: (
                    rec["score"],
                    rec["confidence"],
                    rec["cooccurrence_count"],
                ),
                reverse=True,
            )[: self.top_k_per_item]
            for item_id, item_recommendations in recommendations.items()
        }

    def _make_directional_recommendation(
        self,
        source_item: str,
        target_item: str,
        pair_count: int,
        total_baskets: int,
    ) -> dict[str, Any]:
        source_count = self.item_counts[source_item]
        target_count = self.item_counts[target_item]

        support = pair_count / total_baskets
        confidence = pair_count / source_count if source_count else 0.0
        target_support = target_count / total_baskets if total_baskets else 0.0
        lift = confidence / target_support if target_support else 0.0
        cosine = pair_count / math.sqrt(source_count * target_count)
        jaccard = pair_count / (source_count + target_count - pair_count)
        shrunk_confidence = pair_count / (
            source_count + self.confidence_shrinkage
        )
        score = (
            shrunk_confidence
            * (math.log1p(lift) ** self.lift_weight)
            * (math.log1p(pair_count) ** self.count_weight)
        )
        if self.popularity_penalty_weight > 0:
            score /= math.log1p(target_count) ** self.popularity_penalty_weight

        return {
            "item_id": target_item,
            "score": score,
            "support": support,
            "confidence": confidence,
            "shrunk_confidence": shrunk_confidence,
            "lift": lift,
            "cosine": cosine,
            "jaccard": jaccard,
            "cooccurrence_count": pair_count,
            "source_item_count": source_count,
            "target_item_count": target_count,
            "reason": "cooccurrence",
        }

    def _popular_fallback_recommendations(
        self,
        excluded_items: set[str],
        n: int,
    ) -> list[dict[str, Any]]:
        recommendations = []
        for item_id, basket_count in self.item_counts.most_common():
            if item_id in excluded_items:
                continue
            recommendations.append(
                {
                    "item_id": item_id,
                    "score": 0.0,
                    "support": (
                        basket_count / self.total_baskets
                        if self.total_baskets
                        else 0.0
                    ),
                    "confidence": 0.0,
                    "shrunk_confidence": 0.0,
                    "lift": 0.0,
                    "cosine": 0.0,
                    "jaccard": 0.0,
                    "cooccurrence_count": 0,
                    "source_item_count": 0,
                    "target_item_count": basket_count,
                    "reason": "popular_fallback",
                }
            )
            if len(recommendations) == n:
                break
        return recommendations

    @staticmethod
    def normalize_basket(raw_basket: Any) -> list[str]:
        if raw_basket is None:
            return []

        if isinstance(raw_basket, str):
            stripped = raw_basket.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    raw_basket = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    raw_basket = [stripped]
            else:
                raw_basket = [stripped]

        try:
            iterator = iter(raw_basket)
        except TypeError:
            iterator = iter([raw_basket])

        seen = set()
        items = []
        for item in iterator:
            if item is None:
                continue
            item_id = str(item).strip()
            if not item_id or item_id.lower() == "nan" or item_id in seen:
                continue
            seen.add(item_id)
            items.append(item_id)

        return items

    _normalize_basket = normalize_basket
