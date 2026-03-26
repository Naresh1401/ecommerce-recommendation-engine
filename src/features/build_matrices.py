"""
Feature Engineering — Build recommendation matrices from raw data.
Produces sparse user-item matrices, user/item feature matrices, and TF-IDF vectors.
"""

import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib


# Interaction weights for implicit feedback signal
INTERACTION_SCORES = {
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
}


def load_raw_data(project_root: str):
    """Load raw CSVs."""
    raw_dir = os.path.join(project_root, "data", "raw")
    users = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    products = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    interactions = pd.read_csv(os.path.join(raw_dir, "interactions.csv"), parse_dates=["timestamp"])
    return users, products, interactions


def build_user_item_matrix(interactions: pd.DataFrame, n_users: int, n_products: int) -> sparse.csr_matrix:
    """
    Build a sparse user-item interaction matrix with weighted implicit feedback.
    Rows = users (0-indexed), Columns = products (0-indexed).
    """
    interactions = interactions.copy()
    interactions["score"] = interactions["interaction_type"].map(INTERACTION_SCORES)

    # Aggregate scores per user-product pair
    agg = interactions.groupby(["user_id", "product_id"])["score"].sum().reset_index()

    rows = agg["user_id"].values - 1  # 0-indexed
    cols = agg["product_id"].values - 1
    data = agg["score"].values.astype(np.float32)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_products))
    return matrix


def build_rating_matrix(interactions: pd.DataFrame, n_users: int, n_products: int) -> sparse.csr_matrix:
    """Build explicit rating matrix (purchases with ratings only)."""
    purchases = interactions[interactions["rating"].notna()].copy()
    if purchases.empty:
        return sparse.csr_matrix((n_users, n_products))

    agg = purchases.groupby(["user_id", "product_id"])["rating"].mean().reset_index()
    rows = agg["user_id"].values - 1
    cols = agg["product_id"].values - 1
    data = agg["rating"].values.astype(np.float32)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_products))
    return matrix


def build_user_features(users: pd.DataFrame) -> np.ndarray:
    """Encode user demographic features into a numeric matrix."""
    df = users.copy()

    le_age = LabelEncoder()
    le_gender = LabelEncoder()
    le_region = LabelEncoder()

    df["age_enc"] = le_age.fit_transform(df["age_group"])
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["region_enc"] = le_region.fit_transform(df["region"])

    features = df[["age_enc", "gender_enc", "region_enc"]].values.astype(np.float32)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features


def build_item_features(products: pd.DataFrame, interactions: pd.DataFrame) -> np.ndarray:
    """Encode item features: category, price bin, brand popularity, avg_rating."""
    df = products.copy()

    le_cat = LabelEncoder()
    le_subcat = LabelEncoder()
    le_brand = LabelEncoder()

    df["cat_enc"] = le_cat.fit_transform(df["category"])
    df["subcat_enc"] = le_subcat.fit_transform(df["subcategory"])
    df["brand_enc"] = le_brand.fit_transform(df["brand"])

    # Price bins (quintiles)
    df["price_bin"] = pd.qcut(df["price"], q=10, labels=False, duplicates="drop")

    # Brand popularity: number of interactions per brand
    brand_counts = interactions.groupby("product_id").size().reset_index(name="interaction_count")
    df = df.merge(brand_counts, on="product_id", how="left")
    df["interaction_count"] = df["interaction_count"].fillna(0)

    feature_cols = ["cat_enc", "subcat_enc", "brand_enc", "price_bin", "avg_rating", "interaction_count"]
    features = df[feature_cols].values.astype(np.float32)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features


def build_tfidf_vectors(products: pd.DataFrame) -> sparse.csr_matrix:
    """Build TF-IDF content vectors from product category + subcategory + brand."""
    products = products.copy()
    products["content"] = (
        products["category"] + " " +
        products["subcategory"] + " " +
        products["brand"] + " " +
        products["category"] + "_" + products["subcategory"]
    )
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(products["content"])
    return tfidf_matrix, vectorizer


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading raw data...")
    users, products, interactions = load_raw_data(project_root)

    n_users = users["user_id"].max()
    n_products = products["product_id"].max()

    print("Building user-item interaction matrix...")
    ui_matrix = build_user_item_matrix(interactions, n_users, n_products)
    sparse.save_npz(os.path.join(processed_dir, "user_item_matrix.npz"), ui_matrix)
    print(f"  -> Shape: {ui_matrix.shape}, NNZ: {ui_matrix.nnz}")

    print("Building rating matrix...")
    rating_matrix = build_rating_matrix(interactions, n_users, n_products)
    sparse.save_npz(os.path.join(processed_dir, "rating_matrix.npz"), rating_matrix)
    print(f"  -> Shape: {rating_matrix.shape}, NNZ: {rating_matrix.nnz}")

    print("Building user features...")
    user_features = build_user_features(users)
    joblib.dump(user_features, os.path.join(processed_dir, "user_features.joblib"))
    print(f"  -> Shape: {user_features.shape}")

    print("Building item features...")
    item_features = build_item_features(products, interactions)
    joblib.dump(item_features, os.path.join(processed_dir, "item_features.joblib"))
    print(f"  -> Shape: {item_features.shape}")

    print("Building TF-IDF content vectors...")
    tfidf_matrix, vectorizer = build_tfidf_vectors(products)
    sparse.save_npz(os.path.join(processed_dir, "tfidf_matrix.npz"), tfidf_matrix)
    joblib.dump(vectorizer, os.path.join(processed_dir, "tfidf_vectorizer.joblib"))
    print(f"  -> Shape: {tfidf_matrix.shape}")

    print("\nAll feature matrices saved to data/processed/")


if __name__ == "__main__":
    main()
