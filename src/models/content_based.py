"""
Content-Based Filtering Recommender
Uses TF-IDF vectors from product attributes and cosine similarity
to recommend items similar to a user's historical preferences.
"""

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Content-based recommender using TF-IDF product representations."""

    def __init__(self, price_weight: float = 0.2):
        self.price_weight = price_weight
        self.tfidf_matrix = None
        self.item_similarity = None
        self.user_item_matrix = None
        self.item_prices = None
        self.name = "Content-Based"

    def fit(self, user_item_matrix: sparse.csr_matrix, tfidf_matrix: sparse.csr_matrix,
            item_prices: np.ndarray = None):
        """
        Fit the content-based model.

        Parameters
        ----------
        user_item_matrix : sparse matrix (n_users, n_items)
        tfidf_matrix : sparse matrix (n_items, n_features) from product content
        item_prices : optional array of item prices for price-aware filtering
        """
        self.user_item_matrix = user_item_matrix
        self.tfidf_matrix = tfidf_matrix
        self.item_prices = item_prices

        # Precompute item-item content similarity
        self.item_similarity = cosine_similarity(tfidf_matrix, dense_output=False)
        return self

    def _get_user_profile(self, user_id: int) -> np.ndarray:
        """
        Build a user content profile as a weighted average of TF-IDF vectors
        of items the user has interacted with.
        """
        user_interactions = self.user_item_matrix[user_id]
        interacted_items = user_interactions.nonzero()[1]
        weights = np.asarray(user_interactions[:, interacted_items].todense()).flatten()

        if len(interacted_items) == 0:
            return None

        # Weighted average of item TF-IDF vectors
        item_vectors = self.tfidf_matrix[interacted_items]
        weighted_profile = np.asarray(item_vectors.T.dot(weights)).flatten()
        norm = np.linalg.norm(weighted_profile)
        if norm > 0:
            weighted_profile /= norm

        return weighted_profile

    def _price_penalty(self, user_id: int, item_indices: np.ndarray) -> np.ndarray:
        """
        Compute a price-range penalty: items whose price is far from the user's
        average purchase price get penalized.
        """
        if self.item_prices is None:
            return np.ones(len(item_indices))

        user_interactions = self.user_item_matrix[user_id]
        interacted_items = user_interactions.nonzero()[1]

        if len(interacted_items) == 0:
            return np.ones(len(item_indices))

        avg_price = np.mean(self.item_prices[interacted_items])
        std_price = max(np.std(self.item_prices[interacted_items]), 1.0)

        candidate_prices = self.item_prices[item_indices]
        # Gaussian kernel around the user's average price
        price_scores = np.exp(-0.5 * ((candidate_prices - avg_price) / std_price) ** 2)
        return price_scores

    def recommend(self, user_id: int, n: int = 10, exclude_known: bool = True) -> list:
        """
        Recommend top-n items for a user based on content similarity.
        Returns list of (item_id, score) tuples.
        """
        if self.item_similarity is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_profile = self._get_user_profile(user_id)

        # Cold-start: no interactions
        if user_profile is None:
            return []

        # Compute similarity between user profile and all items
        scores = np.asarray(self.tfidf_matrix.dot(user_profile)).flatten()

        # Apply price-aware penalty
        all_indices = np.arange(len(scores))
        price_scores = self._price_penalty(user_id, all_indices)
        scores = (1 - self.price_weight) * scores + self.price_weight * price_scores

        # Exclude already-interacted items
        if exclude_known:
            known_items = set(self.user_item_matrix[user_id].nonzero()[1])
            for item in known_items:
                scores[item] = -1

        top_items = np.argsort(scores)[-n:][::-1]
        return [(int(item), float(scores[item])) for item in top_items if scores[item] > 0]

    def similar_items(self, product_id: int, n: int = 10) -> list:
        """Return top-n similar items to a given product (0-indexed)."""
        if self.item_similarity is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        sims = np.asarray(self.item_similarity[product_id].todense()).flatten()
        sims[product_id] = -1  # exclude self
        top_items = np.argsort(sims)[-n:][::-1]
        return [(int(item), float(sims[item])) for item in top_items]
