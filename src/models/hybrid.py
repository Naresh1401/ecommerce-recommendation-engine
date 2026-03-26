"""
Hybrid Recommender
Combines collaborative filtering and content-based scores with popularity re-ranking
and cold-start handling.
"""

import numpy as np
from scipy import sparse


class HybridRecommender:
    """
    Weighted hybrid recommender that blends collaborative and content-based signals,
    with re-ranking for popularity/recency and cold-start fallback strategies.
    """

    def __init__(self, cf_weight: float = 0.6, cb_weight: float = 0.3, pop_weight: float = 0.1,
                 recency_decay: float = 0.01):
        assert abs(cf_weight + cb_weight + pop_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.pop_weight = pop_weight
        self.recency_decay = recency_decay

        self.cf_model = None
        self.cb_model = None
        self.user_item_matrix = None
        self.item_popularity = None
        self.item_recency = None
        self.n_items = 0
        self.name = "Hybrid"

    def fit(self, cf_model, cb_model, user_item_matrix: sparse.csr_matrix,
            item_popularity: np.ndarray = None, item_recency: np.ndarray = None):
        """
        Fit the hybrid model.

        Parameters
        ----------
        cf_model : fitted collaborative filtering model with recommend(user_id, n)
        cb_model : fitted content-based model with recommend(user_id, n)
        user_item_matrix : sparse user-item interaction matrix
        item_popularity : array of popularity scores per item (0-indexed)
        item_recency : array of recency scores per item (higher = more recent)
        """
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.user_item_matrix = user_item_matrix
        self.n_items = user_item_matrix.shape[1]

        # Compute popularity if not provided
        if item_popularity is not None:
            self.item_popularity = item_popularity
        else:
            pop = np.asarray(user_item_matrix.sum(axis=0)).flatten()
            pop_max = pop.max() if pop.max() > 0 else 1.0
            self.item_popularity = pop / pop_max

        # Compute recency if not provided
        if item_recency is not None:
            self.item_recency = item_recency
        else:
            self.item_recency = np.ones(self.n_items)

        return self

    def _is_cold_start_user(self, user_id: int, threshold: int = 3) -> bool:
        """Check if user has fewer than `threshold` interactions."""
        n_interactions = self.user_item_matrix[user_id].nnz
        return n_interactions < threshold

    def _popularity_recommendations(self, n: int, exclude: set = None) -> list:
        """Fallback: recommend most popular items."""
        scores = self.item_popularity.copy()
        if exclude:
            for item in exclude:
                scores[item] = -1
        top_items = np.argsort(scores)[-n:][::-1]
        return [(int(item), float(scores[item])) for item in top_items]

    def _normalize_scores(self, recs: list) -> dict:
        """Normalize recommendation scores to [0, 1]."""
        if not recs:
            return {}
        scores = {item_id: score for item_id, score in recs}
        max_score = max(scores.values()) if scores else 1.0
        min_score = min(scores.values()) if scores else 0.0
        range_score = max_score - min_score if max_score != min_score else 1.0
        return {k: (v - min_score) / range_score for k, v in scores.items()}

    def recommend(self, user_id: int, n: int = 10, exclude_known: bool = True) -> list:
        """
        Generate hybrid recommendations for a user.
        Returns list of (item_id, score) tuples.
        """
        known_items = set()
        if exclude_known:
            known_items = set(self.user_item_matrix[user_id].nonzero()[1])

        # Cold-start user: fallback to content-based or popularity
        if self._is_cold_start_user(user_id):
            cb_recs = self.cb_model.recommend(user_id, n=n * 3, exclude_known=exclude_known)
            if len(cb_recs) >= n:
                return cb_recs[:n]
            # Ultra-cold: no content signal either
            return self._popularity_recommendations(n, exclude=known_items)

        # Get recommendations from each model (fetch more than needed for blending)
        fetch_n = min(n * 5, self.n_items)

        cf_recs = self.cf_model.recommend(user_id, n=fetch_n, exclude_known=exclude_known)
        cb_recs = self.cb_model.recommend(user_id, n=fetch_n, exclude_known=exclude_known)

        # Normalize scores
        cf_scores = self._normalize_scores(cf_recs)
        cb_scores = self._normalize_scores(cb_recs)

        # Combine all candidate items
        all_candidates = set(cf_scores.keys()) | set(cb_scores.keys())

        # Compute blended scores
        blended = {}
        for item_id in all_candidates:
            if item_id in known_items:
                continue

            cf_s = cf_scores.get(item_id, 0.0)
            cb_s = cb_scores.get(item_id, 0.0)
            pop_s = self.item_popularity[item_id] if item_id < len(self.item_popularity) else 0.0
            rec_s = self.item_recency[item_id] if item_id < len(self.item_recency) else 1.0

            score = (
                self.cf_weight * cf_s +
                self.cb_weight * cb_s +
                self.pop_weight * pop_s
            )
            # Recency re-ranking boost
            score *= (1 + self.recency_decay * rec_s)

            blended[item_id] = score

        # Sort and return top-n
        sorted_items = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        return [(int(item_id), float(score)) for item_id, score in sorted_items[:n]]
