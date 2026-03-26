"""
Training Pipeline
- Temporal train/test split
- Train all recommender models
- Evaluate with ranking metrics
- Save best model and metrics
"""

import os
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
import joblib

from src.features.build_matrices import (
    load_raw_data,
    build_user_item_matrix,
    build_rating_matrix,
    build_tfidf_vectors,
    INTERACTION_SCORES,
)
from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF, ALSRecommender, SVDRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender

np.random.seed(42)


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Precision@K: fraction of recommended items that are relevant."""
    if k == 0:
        return 0.0
    rec_k = [item for item, _ in recommended[:k]]
    hits = len(set(rec_k) & relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Recall@K: fraction of relevant items that are recommended."""
    if not relevant:
        return 0.0
    rec_k = [item for item, _ in recommended[:k]]
    hits = len(set(rec_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, (item, _) in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    # Ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(recommended: list, relevant: set, k: int) -> float:
    """Average Precision at K."""
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, (item, _) in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """Hit Rate@K: 1 if at least one relevant item is in top-K, else 0."""
    rec_k = [item for item, _ in recommended[:k]]
    return 1.0 if len(set(rec_k) & relevant) > 0 else 0.0


def coverage(all_recommendations: list, n_items: int) -> float:
    """Catalog coverage: fraction of total items ever recommended."""
    unique_items = set()
    for recs in all_recommendations:
        for item, _ in recs:
            unique_items.add(item)
    return len(unique_items) / n_items if n_items > 0 else 0.0


def diversity(all_recommendations: list, item_categories: dict) -> float:
    """Intra-list diversity: average number of unique categories per recommendation list."""
    if not all_recommendations:
        return 0.0
    diversities = []
    for recs in all_recommendations:
        cats = set()
        for item, _ in recs:
            if item in item_categories:
                cats.add(item_categories[item])
        if len(recs) > 0:
            diversities.append(len(cats) / len(recs))
    return np.mean(diversities) if diversities else 0.0


# ---------------------------------------------------------------------------
# Evaluation Harness
# ---------------------------------------------------------------------------

def evaluate_model(model, test_user_items: dict, n_items: int, item_categories: dict,
                   k: int = 10, sample_users: int = 1000) -> dict:
    """Evaluate a model across multiple ranking metrics."""
    users = list(test_user_items.keys())
    if len(users) > sample_users:
        users = np.random.choice(users, size=sample_users, replace=False)

    metrics = {"precision": [], "recall": [], "ndcg": [], "map": [], "hit_rate": []}
    all_recs = []

    for user_id in users:
        relevant = test_user_items[user_id]
        if not relevant:
            continue

        try:
            recs = model.recommend(user_id, n=k, exclude_known=True)
        except Exception:
            continue

        if not recs:
            continue

        metrics["precision"].append(precision_at_k(recs, relevant, k))
        metrics["recall"].append(recall_at_k(recs, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(recs, relevant, k))
        metrics["map"].append(average_precision(recs, relevant, k))
        metrics["hit_rate"].append(hit_rate_at_k(recs, relevant, k))
        all_recs.append(recs)

    result = {
        f"Precision@{k}": float(np.mean(metrics["precision"])) if metrics["precision"] else 0.0,
        f"Recall@{k}": float(np.mean(metrics["recall"])) if metrics["recall"] else 0.0,
        f"NDCG@{k}": float(np.mean(metrics["ndcg"])) if metrics["ndcg"] else 0.0,
        f"MAP@{k}": float(np.mean(metrics["map"])) if metrics["map"] else 0.0,
        f"HitRate@{k}": float(np.mean(metrics["hit_rate"])) if metrics["hit_rate"] else 0.0,
        "Coverage": float(coverage(all_recs, n_items)),
        "Diversity": float(diversity(all_recs, item_categories)),
        "n_evaluated_users": len(metrics["precision"]),
    }
    return result


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("  E-Commerce Recommendation Engine — Training Pipeline")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading raw data...")
    users, products, interactions = load_raw_data(project_root)

    n_users = users["user_id"].max()
    n_products = products["product_id"].max()

    # Item categories mapping (0-indexed)
    item_categories = dict(zip(products["product_id"] - 1, products["category"]))
    item_prices = products.sort_values("product_id")["price"].values.astype(np.float32)

    # Temporal train/test split: last 20% of interactions by time
    print("[2/6] Creating temporal train/test split...")
    interactions_sorted = interactions.sort_values("timestamp")
    split_idx = int(len(interactions_sorted) * 0.8)
    train_interactions = interactions_sorted.iloc[:split_idx]
    test_interactions = interactions_sorted.iloc[split_idx:]

    print(f"  Train: {len(train_interactions):,} interactions")
    print(f"  Test:  {len(test_interactions):,} interactions")

    # Build matrices from training data
    train_ui_matrix = build_user_item_matrix(train_interactions, n_users, n_products)

    # Build rating matrix from training data
    train_rating_matrix = build_rating_matrix(train_interactions, n_users, n_products)

    # Build TF-IDF
    tfidf_matrix, tfidf_vectorizer = build_tfidf_vectors(products)

    # Test ground truth: items each user interacted with in test set (purchase or add_to_cart)
    test_relevant = test_interactions[
        test_interactions["interaction_type"].isin(["purchase", "add_to_cart"])
    ]
    test_user_items = {}
    for uid, group in test_relevant.groupby("user_id"):
        test_user_items[uid - 1] = set(group["product_id"].values - 1)  # 0-indexed

    print(f"  Test users with relevant items: {len(test_user_items):,}")

    # Train models
    print("\n[3/6] Training models...")
    k = 10
    results = {}

    # --- User-Based CF ---
    print("  Training User-Based CF...")
    t0 = time.time()
    user_cf = UserBasedCF(k_neighbors=50)
    user_cf.fit(train_ui_matrix)
    train_time = time.time() - t0
    print(f"    Trained in {train_time:.1f}s")

    # --- Item-Based CF ---
    print("  Training Item-Based CF...")
    t0 = time.time()
    item_cf = ItemBasedCF(k_neighbors=50)
    item_cf.fit(train_ui_matrix)
    train_time = time.time() - t0
    print(f"    Trained in {train_time:.1f}s")

    # --- ALS ---
    print("  Training ALS Matrix Factorization...")
    t0 = time.time()
    als = ALSRecommender(factors=64, regularization=0.01, iterations=15)
    als.fit(train_ui_matrix)
    train_time = time.time() - t0
    print(f"    Trained in {train_time:.1f}s")

    # --- SVD ---
    print("  Training SVD...")
    t0 = time.time()
    svd = SVDRecommender(n_factors=50, n_epochs=20)
    svd.fit(train_ui_matrix, rating_matrix=train_rating_matrix)
    train_time = time.time() - t0
    print(f"    Trained in {train_time:.1f}s")

    # --- Content-Based ---
    print("  Training Content-Based...")
    t0 = time.time()
    cb = ContentBasedRecommender(price_weight=0.2)
    cb.fit(train_ui_matrix, tfidf_matrix, item_prices=item_prices)
    train_time = time.time() - t0
    print(f"    Trained in {train_time:.1f}s")

    # --- Hybrid ---
    print("  Training Hybrid...")
    t0 = time.time()
    # Compute item popularity from training data
    item_pop = np.asarray(train_ui_matrix.sum(axis=0)).flatten()
    item_pop = item_pop / (item_pop.max() if item_pop.max() > 0 else 1.0)

    hybrid = HybridRecommender(cf_weight=0.6, cb_weight=0.3, pop_weight=0.1)
    hybrid.fit(als, cb, train_ui_matrix, item_popularity=item_pop)
    train_time = time.time() - t0
    print(f"    Trained in {train_time:.1f}s")

    # Evaluate all models
    print("\n[4/6] Evaluating models...")
    models = {
        "User-Based CF": user_cf,
        "Item-Based CF": item_cf,
        "ALS": als,
        "SVD": svd,
        "Content-Based": cb,
        "Hybrid": hybrid,
    }

    for name, model in models.items():
        print(f"  Evaluating {name}...")
        results[name] = evaluate_model(model, test_user_items, n_products, item_categories, k=k)

    # Print comparison table
    print("\n[5/6] Results Comparison (K={})".format(k))
    print("-" * 95)
    header = f"{'Model':<20} {'Prec@K':>8} {'Rec@K':>8} {'NDCG@K':>8} {'MAP@K':>8} {'HitRate':>8} {'Cover':>8} {'Divers':>8}"
    print(header)
    print("-" * 95)

    best_model_name = None
    best_ndcg = -1

    for name, metrics in results.items():
        p = metrics[f"Precision@{k}"]
        r = metrics[f"Recall@{k}"]
        ndcg = metrics[f"NDCG@{k}"]
        m = metrics[f"MAP@{k}"]
        hr = metrics[f"HitRate@{k}"]
        cov = metrics["Coverage"]
        div = metrics["Diversity"]

        marker = ""
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_model_name = name

        print(f"{name:<20} {p:>8.4f} {r:>8.4f} {ndcg:>8.4f} {m:>8.4f} {hr:>8.4f} {cov:>7.1%} {div:>8.3f}")

    print("-" * 95)
    print(f"Best model by NDCG@{k}: {best_model_name}")

    # Save artifacts
    print("\n[6/6] Saving artifacts...")
    best_model = models[best_model_name]
    joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(hybrid, os.path.join(models_dir, "hybrid_model.joblib"))
    joblib.dump(cb, os.path.join(models_dir, "content_based_model.joblib"))
    joblib.dump(als, os.path.join(models_dir, "als_model.joblib"))

    # Save full user-item matrix for serving
    full_ui_matrix = build_user_item_matrix(interactions, n_users, n_products)
    sparse.save_npz(os.path.join(models_dir, "user_item_matrix.npz"), full_ui_matrix)
    sparse.save_npz(os.path.join(models_dir, "tfidf_matrix.npz"), tfidf_matrix)
    joblib.dump(item_prices, os.path.join(models_dir, "item_prices.joblib"))

    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Models saved to {models_dir}/")
    print(f"  Metrics saved to {results_dir}/metrics.json")
    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()
