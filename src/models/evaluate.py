"""
Detailed Evaluation & Visualization
Generates comparison charts, coverage analysis, popularity bias, and cold-start analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

from src.features.build_matrices import load_raw_data, build_user_item_matrix, build_tfidf_vectors, INTERACTION_SCORES
from src.models.train import evaluate_model

import joblib

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="husl")


def load_artifacts(project_root: str):
    """Load saved models and data."""
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")

    with open(os.path.join(results_dir, "metrics.json")) as f:
        metrics = json.load(f)

    hybrid = joblib.load(os.path.join(models_dir, "hybrid_model.joblib"))
    cb = joblib.load(os.path.join(models_dir, "content_based_model.joblib"))
    als = joblib.load(os.path.join(models_dir, "als_model.joblib"))

    return metrics, hybrid, cb, als


def plot_ndcg_comparison(metrics: dict, figures_dir: str):
    """Bar chart comparing NDCG@10 across models."""
    models = list(metrics.keys())
    ndcg_values = [metrics[m].get("NDCG@10", 0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(models))
    bars = ax.barh(models, ndcg_values, color=colors, edgecolor="white", linewidth=0.5)

    # Highlight best
    best_idx = np.argmax(ndcg_values)
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

    ax.set_xlabel("NDCG@10", fontsize=12)
    ax.set_title("Model Comparison — NDCG@10", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, ndcg_values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ndcg_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved ndcg_comparison.png")


def plot_precision_recall_at_k(metrics: dict, test_user_items: dict, models_dict: dict,
                                n_items: int, item_categories: dict, figures_dir: str):
    """Precision and Recall at different K values."""
    k_values = [1, 3, 5, 10, 20]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, model in models_dict.items():
        precisions = []
        recalls = []
        for k in k_values:
            result = evaluate_model(model, test_user_items, n_items, item_categories, k=k, sample_users=500)
            precisions.append(result.get(f"Precision@{k}", 0))
            recalls.append(result.get(f"Recall@{k}", 0))

        axes[0].plot(k_values, precisions, marker="o", label=name, linewidth=2)
        axes[1].plot(k_values, recalls, marker="s", label=name, linewidth=2)

    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Precision@K")
    axes[0].set_title("Precision@K Across Models", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Recall@K")
    axes[1].set_title("Recall@K Across Models", fontweight="bold")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "precision_recall_at_k.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved precision_recall_at_k.png")


def plot_coverage_analysis(metrics: dict, figures_dir: str):
    """Coverage and Diversity comparison."""
    models = list(metrics.keys())
    coverage_vals = [metrics[m].get("Coverage", 0) * 100 for m in models]
    diversity_vals = [metrics[m].get("Diversity", 0) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = sns.color_palette("husl", len(models))
    axes[0].barh(models, coverage_vals, color=colors)
    axes[0].set_xlabel("Coverage (%)")
    axes[0].set_title("Catalog Coverage", fontweight="bold")
    for i, v in enumerate(coverage_vals):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=10)

    axes[1].barh(models, diversity_vals, color=colors)
    axes[1].set_xlabel("Diversity Score")
    axes[1].set_title("Intra-List Diversity", fontweight="bold")
    for i, v in enumerate(diversity_vals):
        axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "coverage_diversity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved coverage_diversity.png")


def plot_popularity_bias(train_ui_matrix: sparse.csr_matrix, model, model_name: str,
                          test_user_items: dict, figures_dir: str):
    """Analyze how recommendations skew toward popular items vs. long-tail."""
    item_pop = np.asarray(train_ui_matrix.sum(axis=0)).flatten()

    # Get recommendations for sample users
    sample_users = list(test_user_items.keys())[:500]
    rec_items = []
    for uid in sample_users:
        try:
            recs = model.recommend(uid, n=10, exclude_known=True)
            rec_items.extend([item for item, _ in recs])
        except Exception:
            continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distribution of item popularity in catalog vs recommendations
    axes[0].hist(item_pop, bins=50, alpha=0.6, label="All Items", color="steelblue", density=True)
    if rec_items:
        rec_pop = item_pop[rec_items]
        axes[0].hist(rec_pop, bins=50, alpha=0.6, label="Recommended", color="coral", density=True)
    axes[0].set_xlabel("Item Popularity (interaction count)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Popularity Bias — {model_name}", fontweight="bold")
    axes[0].legend()

    # Long-tail analysis: cumulative % of recommendations by item popularity rank
    if rec_items:
        item_counts = pd.Series(rec_items).value_counts()
        popular_ranks = np.argsort(-item_pop)
        rank_map = {item: rank for rank, item in enumerate(popular_ranks)}

        rec_ranks = [rank_map.get(item, len(item_pop)) for item in item_counts.index]
        sorted_by_rank = sorted(zip(rec_ranks, item_counts.values))
        cumulative = np.cumsum([v for _, v in sorted_by_rank])
        cumulative = cumulative / cumulative[-1]
        percentiles = np.arange(len(cumulative)) / len(cumulative) * 100

        axes[1].plot(percentiles, cumulative * 100, linewidth=2, color="coral")
        axes[1].plot([0, 100], [0, 100], "k--", alpha=0.3, label="Perfect equality")
        axes[1].set_xlabel("Item Popularity Percentile")
        axes[1].set_ylabel("Cumulative % of Recommendations")
        axes[1].set_title("Long-Tail Distribution", fontweight="bold")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "popularity_bias.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved popularity_bias.png")


def plot_cold_start_analysis(train_ui_matrix: sparse.csr_matrix, model,
                              test_user_items: dict, item_categories: dict,
                              n_items: int, figures_dir: str):
    """Evaluate model performance by user activity level (cold vs warm)."""
    user_activity = np.asarray(train_ui_matrix.sum(axis=1)).flatten()

    # Bucket users by activity level
    buckets = {
        "1-5 interactions": (1, 5),
        "6-20 interactions": (6, 20),
        "21-50 interactions": (21, 50),
        "51-100 interactions": (51, 100),
        "100+ interactions": (101, 99999),
    }

    bucket_metrics = {}
    for bucket_name, (lo, hi) in buckets.items():
        bucket_users = {}
        for uid in test_user_items:
            activity = user_activity[uid] if uid < len(user_activity) else 0
            if lo <= activity <= hi:
                bucket_users[uid] = test_user_items[uid]

        if len(bucket_users) > 10:
            result = evaluate_model(model, bucket_users, n_items, item_categories, k=10, sample_users=300)
            bucket_metrics[bucket_name] = result
        else:
            bucket_metrics[bucket_name] = {"NDCG@10": 0, "HitRate@10": 0}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = list(bucket_metrics.keys())
    ndcg_vals = [bucket_metrics[n].get("NDCG@10", 0) for n in names]
    hr_vals = [bucket_metrics[n].get("HitRate@10", 0) for n in names]

    x = np.arange(len(names))
    axes[0].bar(x, ndcg_vals, color=sns.color_palette("Blues_d", len(names)))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("NDCG@10")
    axes[0].set_title("Cold-Start Analysis — NDCG@10", fontweight="bold")

    axes[1].bar(x, hr_vals, color=sns.color_palette("Oranges_d", len(names)))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Hit Rate@10")
    axes[1].set_title("Cold-Start Analysis — Hit Rate@10", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cold_start_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved cold_start_analysis.png")


def plot_metrics_heatmap(metrics: dict, figures_dir: str):
    """Heatmap of all metrics across all models."""
    models = list(metrics.keys())
    metric_names = ["Precision@10", "Recall@10", "NDCG@10", "MAP@10", "HitRate@10", "Coverage", "Diversity"]

    data = []
    for m in models:
        row = [metrics[m].get(mn, 0) for mn in metric_names]
        data.append(row)

    df = pd.DataFrame(data, index=models, columns=metric_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Score"})
    ax.set_title("Model Performance Heatmap", fontsize=14, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "metrics_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved metrics_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    figures_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("  Evaluation & Visualization Pipeline")
    print("=" * 60)

    # Load data and artifacts
    print("\nLoading data and models...")
    users, products, interactions = load_raw_data(project_root)
    metrics, hybrid, cb, als = load_artifacts(project_root)

    n_users = users["user_id"].max()
    n_products = products["product_id"].max()
    item_categories = dict(zip(products["product_id"] - 1, products["category"]))

    # Temporal split for evaluation
    interactions_sorted = interactions.sort_values("timestamp")
    split_idx = int(len(interactions_sorted) * 0.8)
    train_interactions = interactions_sorted.iloc[:split_idx]
    test_interactions = interactions_sorted.iloc[split_idx:]

    train_ui_matrix = build_user_item_matrix(train_interactions, n_users, n_products)

    test_relevant = test_interactions[
        test_interactions["interaction_type"].isin(["purchase", "add_to_cart"])
    ]
    test_user_items = {}
    for uid, group in test_relevant.groupby("user_id"):
        test_user_items[uid - 1] = set(group["product_id"].values - 1)

    models_dict = {"ALS": als, "Content-Based": cb, "Hybrid": hybrid}

    # Generate all plots
    print("\nGenerating evaluation plots...")
    plot_ndcg_comparison(metrics, figures_dir)
    plot_metrics_heatmap(metrics, figures_dir)
    plot_coverage_analysis(metrics, figures_dir)
    plot_precision_recall_at_k(metrics, test_user_items, models_dict, n_products, item_categories, figures_dir)
    plot_popularity_bias(train_ui_matrix, hybrid, "Hybrid", test_user_items, figures_dir)
    plot_cold_start_analysis(train_ui_matrix, hybrid, test_user_items, item_categories, n_products, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
