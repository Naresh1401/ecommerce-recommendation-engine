"""
Collaborative Filtering Recommenders
- User-Based CF (cosine similarity)
- Item-Based CF (cosine similarity)
- ALS Matrix Factorization (implicit library)
- SVD (surprise library)
"""

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class UserBasedCF:
    """User-based collaborative filtering using cosine similarity."""

    def __init__(self, k_neighbors: int = 50):
        self.k = k_neighbors
        self.user_item_matrix = None
        self.user_similarity = None
        self.name = "User-Based CF"

    def fit(self, user_item_matrix: sparse.csr_matrix):
        self.user_item_matrix = user_item_matrix
        # Compute pairwise user similarities
        self.user_similarity = cosine_similarity(user_item_matrix, dense_output=False)
        return self

    def recommend(self, user_id: int, n: int = 10, exclude_known: bool = True) -> list:
        """
        Recommend top-n items for a user.
        user_id is 0-indexed.
        Returns list of (item_id, score) tuples.
        """
        if self.user_similarity is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_idx = user_id
        sim_scores = np.asarray(self.user_similarity[user_idx].todense()).flatten()

        # Get top-K similar users (excluding self)
        sim_scores[user_idx] = 0
        top_neighbors = np.argsort(sim_scores)[-self.k:]

        # Aggregate neighbor interactions weighted by similarity
        neighbor_sims = sim_scores[top_neighbors]
        neighbor_matrix = self.user_item_matrix[top_neighbors]

        scores = np.asarray(neighbor_matrix.T.dot(neighbor_sims)).flatten()

        # Exclude already-interacted items
        if exclude_known:
            known_items = set(self.user_item_matrix[user_idx].nonzero()[1])
            for item in known_items:
                scores[item] = -1

        top_items = np.argsort(scores)[-n:][::-1]
        return [(int(item), float(scores[item])) for item in top_items if scores[item] > 0]


class ItemBasedCF:
    """Item-based collaborative filtering using cosine similarity."""

    def __init__(self, k_neighbors: int = 50):
        self.k = k_neighbors
        self.user_item_matrix = None
        self.item_similarity = None
        self.name = "Item-Based CF"

    def fit(self, user_item_matrix: sparse.csr_matrix):
        self.user_item_matrix = user_item_matrix
        # Item-item similarity (transpose so items are rows)
        self.item_similarity = cosine_similarity(user_item_matrix.T, dense_output=False)
        return self

    def recommend(self, user_id: int, n: int = 10, exclude_known: bool = True) -> list:
        """Recommend top-n items based on similarity to user's past items."""
        if self.item_similarity is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_idx = user_id
        user_interactions = self.user_item_matrix[user_idx]
        interacted_items = user_interactions.nonzero()[1]
        interaction_weights = np.asarray(user_interactions[:, interacted_items].todense()).flatten()

        if len(interacted_items) == 0:
            return []

        # Score each candidate item
        candidate_scores = np.zeros(self.user_item_matrix.shape[1])
        for item_idx, weight in zip(interacted_items, interaction_weights):
            sims = np.asarray(self.item_similarity[item_idx].todense()).flatten()
            candidate_scores += sims * weight

        if exclude_known:
            for item in interacted_items:
                candidate_scores[item] = -1

        top_items = np.argsort(candidate_scores)[-n:][::-1]
        return [(int(item), float(candidate_scores[item])) for item in top_items if candidate_scores[item] > 0]


class ALSRecommender:
    """Alternating Least Squares matrix factorization using the implicit library."""

    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.model = None
        self.user_item_matrix = None
        self.name = "ALS (Matrix Factorization)"

    def fit(self, user_item_matrix: sparse.csr_matrix):
        import implicit

        self.user_item_matrix = user_item_matrix
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
        )
        # implicit expects item-user matrix (CSR) for fitting
        self.model.fit(user_item_matrix.T.tocsr())
        return self

    def recommend(self, user_id: int, n: int = 10, exclude_known: bool = True) -> list:
        """Recommend top-n items for a user via ALS."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Filter items already seen
        filter_items = None
        if not exclude_known:
            filter_items = []

        ids, scores = self.model.recommend(
            user_id,
            self.user_item_matrix[user_id],
            N=n,
            filter_already_liked_items=exclude_known,
        )
        return [(int(i), float(s)) for i, s in zip(ids, scores)]


class SVDRecommender:
    """SVD-based recommender using the surprise library for explicit ratings."""

    def __init__(self, n_factors: int = 50, n_epochs: int = 20, lr_all: float = 0.005, reg_all: float = 0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.model = None
        self.trainset = None
        self.user_item_matrix = None
        self.name = "SVD"

    def fit(self, user_item_matrix: sparse.csr_matrix, rating_matrix: sparse.csr_matrix = None):
        from surprise import SVD, Dataset, Reader

        self.user_item_matrix = user_item_matrix

        # Use rating matrix if available, otherwise convert interaction matrix
        matrix = rating_matrix if rating_matrix is not None else user_item_matrix

        # Convert sparse matrix to surprise dataset
        coo = matrix.tocoo()
        rows, cols, vals = coo.row, coo.col, coo.data

        import pandas as pd
        df = pd.DataFrame({"user": rows, "item": cols, "rating": vals})

        reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
        data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
        self.trainset = data.build_full_trainset()

        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42,
        )
        self.model.fit(self.trainset)
        return self

    def recommend(self, user_id: int, n: int = 10, exclude_known: bool = True) -> list:
        """Predict ratings for all items and return top-n."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_items = self.user_item_matrix.shape[1]
        known_items = set(self.user_item_matrix[user_id].nonzero()[1]) if exclude_known else set()

        predictions = []
        for item_id in range(n_items):
            if item_id in known_items:
                continue
            try:
                inner_uid = self.trainset.to_inner_uid(user_id)
                inner_iid = self.trainset.to_inner_iid(item_id)
                pred = self.model.predict(user_id, item_id).est
                predictions.append((item_id, pred))
            except ValueError:
                continue

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
