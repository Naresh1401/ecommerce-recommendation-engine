"""
Synthetic E-Commerce Data Generator
Generates realistic users, products, and interaction data with behavioral patterns.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_USERS = 10_000
NUM_PRODUCTS = 5_000
TARGET_INTERACTIONS = 500_000

CATEGORIES = {
    "Electronics": ["Smartphones", "Laptops", "Headphones", "Cameras", "Tablets", "Speakers"],
    "Clothing": ["T-Shirts", "Jeans", "Dresses", "Jackets", "Shoes", "Accessories"],
    "Books": ["Fiction", "Non-Fiction", "Science", "History", "Self-Help", "Comics"],
    "Home": ["Furniture", "Kitchen", "Decor", "Bedding", "Lighting", "Storage"],
    "Sports": ["Fitness", "Outdoor", "Team Sports", "Cycling", "Swimming", "Yoga"],
    "Beauty": ["Skincare", "Makeup", "Haircare", "Fragrance", "Nails", "Tools"],
}

BRANDS = {
    "Electronics": ["TechPro", "NovaTech", "DigiMax", "ElectraX", "SmartWave"],
    "Clothing": ["UrbanFit", "StyleCraft", "ModaVista", "ThreadLine", "AeroWear"],
    "Books": ["PageTurner", "BookHaven", "LitPress", "ReadMore", "InkWell"],
    "Home": ["HomeNest", "CozyLiving", "CasaLux", "DomStyle", "NestCraft"],
    "Sports": ["ActiveEdge", "FitGear", "SportMax", "PeakForm", "AthletiX"],
    "Beauty": ["GlowUp", "PureSkin", "LuxeBeauty", "RadiantCo", "VelvetTouch"],
}

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
GENDERS = ["M", "F", "Non-Binary"]
REGIONS = ["North America", "Europe", "Asia", "South America", "Oceania", "Africa"]

INTERACTION_TYPES = ["view", "click", "add_to_cart", "purchase"]
INTERACTION_WEIGHTS = [0.50, 0.25, 0.15, 0.10]

# Category preferences by age group (probability weights)
AGE_CATEGORY_AFFINITY = {
    "18-24": {"Electronics": 0.30, "Clothing": 0.25, "Books": 0.05, "Home": 0.05, "Sports": 0.20, "Beauty": 0.15},
    "25-34": {"Electronics": 0.25, "Clothing": 0.20, "Books": 0.10, "Home": 0.15, "Sports": 0.15, "Beauty": 0.15},
    "35-44": {"Electronics": 0.20, "Clothing": 0.15, "Books": 0.15, "Home": 0.25, "Sports": 0.10, "Beauty": 0.15},
    "45-54": {"Electronics": 0.15, "Clothing": 0.15, "Books": 0.20, "Home": 0.25, "Sports": 0.10, "Beauty": 0.15},
    "55-64": {"Electronics": 0.10, "Clothing": 0.10, "Books": 0.30, "Home": 0.25, "Sports": 0.10, "Beauty": 0.15},
    "65+":   {"Electronics": 0.10, "Clothing": 0.10, "Books": 0.30, "Home": 0.30, "Sports": 0.05, "Beauty": 0.15},
}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_users(n: int) -> pd.DataFrame:
    """Generate user demographic data."""
    signup_start = datetime(2021, 1, 1)
    signup_end = datetime(2025, 12, 31)
    total_days = (signup_end - signup_start).days

    users = pd.DataFrame({
        "user_id": range(1, n + 1),
        "age_group": np.random.choice(AGE_GROUPS, size=n, p=[0.18, 0.28, 0.22, 0.16, 0.10, 0.06]),
        "gender": np.random.choice(GENDERS, size=n, p=[0.47, 0.47, 0.06]),
        "signup_date": [signup_start + timedelta(days=int(d)) for d in np.random.randint(0, total_days, size=n)],
        "region": np.random.choice(REGIONS, size=n, p=[0.30, 0.25, 0.25, 0.10, 0.05, 0.05]),
    })
    return users


def generate_products(n: int) -> pd.DataFrame:
    """Generate product catalog data."""
    categories_list = list(CATEGORIES.keys())
    cat_weights = [0.22, 0.22, 0.14, 0.18, 0.12, 0.12]

    records = []
    for i in range(1, n + 1):
        cat = np.random.choice(categories_list, p=cat_weights)
        subcat = np.random.choice(CATEGORIES[cat])
        brand = np.random.choice(BRANDS[cat])

        price_ranges = {
            "Electronics": (29.99, 1499.99),
            "Clothing": (9.99, 299.99),
            "Books": (4.99, 49.99),
            "Home": (14.99, 799.99),
            "Sports": (9.99, 499.99),
            "Beauty": (4.99, 149.99),
        }
        low, high = price_ranges[cat]
        price = round(np.random.uniform(low, high), 2)
        avg_rating = round(np.clip(np.random.normal(3.8, 0.7), 1.0, 5.0), 1)

        records.append({
            "product_id": i,
            "category": cat,
            "subcategory": subcat,
            "price": price,
            "brand": brand,
            "avg_rating": avg_rating,
        })

    return pd.DataFrame(records)


def generate_interactions(users: pd.DataFrame, products: pd.DataFrame, target: int) -> pd.DataFrame:
    """Generate realistic user-product interactions with behavioral patterns."""
    n_users = len(users)
    n_products = len(products)

    # Assign popularity scores to products (power-law)
    popularity = np.random.power(0.5, size=n_products)
    popularity /= popularity.sum()

    # Map user age group to preferred categories
    user_age_map = users.set_index("user_id")["age_group"].to_dict()
    product_cats = products.set_index("product_id")["category"].to_dict()

    # Pre-compute product indices per category
    cat_product_ids = {}
    for cat in CATEGORIES:
        cat_product_ids[cat] = products[products["category"] == cat]["product_id"].values

    # Seasonal weights (month -> category boost)
    seasonal_boosts = {
        12: {"Electronics": 1.5, "Clothing": 1.3, "Books": 1.4, "Home": 1.2, "Sports": 0.8, "Beauty": 1.3},
        11: {"Electronics": 1.3, "Clothing": 1.2, "Books": 1.1, "Home": 1.1, "Sports": 0.9, "Beauty": 1.1},
        6:  {"Sports": 1.5, "Clothing": 1.3, "Beauty": 1.2, "Electronics": 0.9, "Books": 0.8, "Home": 1.0},
        7:  {"Sports": 1.4, "Clothing": 1.3, "Beauty": 1.2, "Electronics": 0.9, "Books": 0.8, "Home": 1.0},
    }

    interaction_start = datetime(2023, 1, 1)
    interaction_end = datetime(2025, 12, 31)
    total_seconds = int((interaction_end - interaction_start).total_seconds())

    records = []
    session_counter = 0

    # Each user gets a variable number of interactions
    interactions_per_user = np.random.negative_binomial(5, 0.09, size=n_users)
    interactions_per_user = np.clip(interactions_per_user, 5, 300)

    # Scale to hit target
    scale_factor = target / interactions_per_user.sum()
    interactions_per_user = np.round(interactions_per_user * scale_factor).astype(int)
    interactions_per_user = np.clip(interactions_per_user, 3, 400)

    categories_list = list(CATEGORIES.keys())

    for idx, user_row in users.iterrows():
        uid = user_row["user_id"]
        age_group = user_row["age_group"]
        n_interactions = interactions_per_user[idx]

        # User-specific category preferences
        cat_prefs = AGE_CATEGORY_AFFINITY[age_group].copy()
        # Add personal variation
        noise = np.random.dirichlet(np.ones(len(categories_list)) * 5)
        for i, cat in enumerate(categories_list):
            cat_prefs[cat] = 0.7 * cat_prefs[cat] + 0.3 * noise[i]

        cat_weights_arr = np.array([cat_prefs[c] for c in categories_list])
        cat_weights_arr /= cat_weights_arr.sum()

        # Generate sessions
        session_counter += 1
        items_in_session = 0

        for _ in range(n_interactions):
            # New session every ~5 interactions
            if items_in_session >= np.random.poisson(5):
                session_counter += 1
                items_in_session = 0

            # Pick category based on user preference
            chosen_cat = np.random.choice(categories_list, p=cat_weights_arr)
            cat_pids = cat_product_ids[chosen_cat]

            # Pick product with popularity bias within category
            cat_pop = popularity[cat_pids - 1]
            cat_pop /= cat_pop.sum()
            pid = np.random.choice(cat_pids, p=cat_pop)

            # Timestamp with some temporal clustering
            ts_offset = np.random.randint(0, total_seconds)
            timestamp = interaction_start + timedelta(seconds=ts_offset)

            # Interaction type — funnel: view > click > cart > purchase
            itype = np.random.choice(INTERACTION_TYPES, p=INTERACTION_WEIGHTS)

            # Rating only for purchases
            rating = None
            if itype == "purchase":
                rating = int(np.clip(np.random.normal(3.8, 1.0), 1, 5))

            # Seasonal boost
            month = timestamp.month
            if month in seasonal_boosts:
                boost = seasonal_boosts[month].get(chosen_cat, 1.0)
                if np.random.random() > (1.0 / boost):
                    pass  # keep the interaction (boosted)

            records.append({
                "user_id": uid,
                "product_id": pid,
                "interaction_type": itype,
                "timestamp": timestamp,
                "rating": rating,
                "session_id": f"sess_{session_counter:08d}",
            })
            items_in_session += 1

    interactions = pd.DataFrame(records)
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)
    return interactions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("Generating users...")
    users = generate_users(NUM_USERS)
    users.to_csv(os.path.join(raw_dir, "users.csv"), index=False)
    print(f"  -> {len(users)} users saved")

    print("Generating products...")
    products = generate_products(NUM_PRODUCTS)
    products.to_csv(os.path.join(raw_dir, "products.csv"), index=False)
    print(f"  -> {len(products)} products saved")

    print("Generating interactions...")
    interactions = generate_interactions(users, products, TARGET_INTERACTIONS)
    interactions.to_csv(os.path.join(raw_dir, "interactions.csv"), index=False)
    print(f"  -> {len(interactions)} interactions saved")

    # Summary statistics
    print("\n--- Data Summary ---")
    print(f"Users:        {len(users):,}")
    print(f"Products:     {len(products):,}")
    print(f"Interactions: {len(interactions):,}")
    print(f"Sparsity:     {1 - len(interactions) / (len(users) * len(products)):.4%}")
    print(f"Purchases:    {(interactions['interaction_type'] == 'purchase').sum():,}")
    print(f"Unique sessions: {interactions['session_id'].nunique():,}")
    print(f"Date range:   {interactions['timestamp'].min()} to {interactions['timestamp'].max()}")


if __name__ == "__main__":
    main()
