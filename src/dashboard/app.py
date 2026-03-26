"""
Streamlit Dashboard for the E-Commerce Recommendation Engine
- Personalized recommendations viewer
- Product similarity explorer
- Model performance comparison
- User segment analysis
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from scipy import sparse


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Recommendation Engine Dashboard",
    page_icon="🛒",
    layout="wide",
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    users = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    products = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    interactions = pd.read_csv(os.path.join(raw_dir, "interactions.csv"), parse_dates=["timestamp"])
    return users, products, interactions


@st.cache_data
def load_metrics():
    metrics_path = os.path.join(PROJECT_ROOT, "results", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_models():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    hybrid = joblib.load(os.path.join(models_dir, "hybrid_model.joblib"))
    cb = joblib.load(os.path.join(models_dir, "content_based_model.joblib"))
    return hybrid, cb


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🛒 RecEngine Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Personalized Recommendations", "Product Similarity", "Model Performance", "User Segments", "Data Explorer"],
)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_recommendations():
    st.title("🎯 Personalized Recommendations")
    st.markdown("Select a user to see their personalized product recommendations from the Hybrid model.")

    users, products, interactions = load_data()
    hybrid, _ = load_models()

    col1, col2 = st.columns([1, 3])

    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=len(users), value=1, step=1)
        n_recs = st.slider("Number of Recommendations", 5, 30, 10)

        # Show user info
        user = users[users["user_id"] == user_id].iloc[0]
        st.markdown("### User Profile")
        st.write(f"**Age Group:** {user['age_group']}")
        st.write(f"**Gender:** {user['gender']}")
        st.write(f"**Region:** {user['region']}")

        # User's recent interactions
        user_ints = interactions[interactions["user_id"] == user_id].sort_values("timestamp", ascending=False).head(10)
        st.markdown("### Recent Activity")
        if not user_ints.empty:
            for _, row in user_ints.iterrows():
                prod = products[products["product_id"] == row["product_id"]]
                if not prod.empty:
                    p = prod.iloc[0]
                    st.write(f"• {row['interaction_type'].title()}: {p['category']} — {p['subcategory']} (${p['price']:.2f})")
        else:
            st.write("No recent interactions.")

    with col2:
        user_idx = user_id - 1
        try:
            recs = hybrid.recommend(user_idx, n=n_recs, exclude_known=True)
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            return

        if not recs:
            st.warning("No recommendations available for this user.")
            return

        st.markdown("### Recommended Products")
        rec_data = []
        for item_id, score in recs:
            pid = item_id + 1
            row = products[products["product_id"] == pid]
            if not row.empty:
                r = row.iloc[0]
                rec_data.append({
                    "Product ID": pid,
                    "Category": r["category"],
                    "Subcategory": r["subcategory"],
                    "Brand": r["brand"],
                    "Price": f"${r['price']:.2f}",
                    "Avg Rating": r["avg_rating"],
                    "Score": round(score, 4),
                })

        if rec_data:
            st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

            # Category distribution of recommendations
            cat_counts = pd.DataFrame(rec_data)["Category"].value_counts()
            fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                         title="Category Distribution of Recommendations")
            st.plotly_chart(fig, use_container_width=True)


def page_similarity():
    st.title("🔍 Product Similarity Explorer")
    st.markdown("Find products similar to a selected product using content-based similarity.")

    _, products, _ = load_data()
    _, cb = load_models()

    col1, col2 = st.columns([1, 3])

    with col1:
        product_id = st.number_input("Product ID", min_value=1, max_value=len(products), value=1, step=1)
        n_similar = st.slider("Number of Similar Products", 5, 20, 10)

        prod = products[products["product_id"] == product_id].iloc[0]
        st.markdown("### Selected Product")
        st.write(f"**Category:** {prod['category']}")
        st.write(f"**Subcategory:** {prod['subcategory']}")
        st.write(f"**Brand:** {prod['brand']}")
        st.write(f"**Price:** ${prod['price']:.2f}")
        st.write(f"**Avg Rating:** {prod['avg_rating']}")

    with col2:
        product_idx = product_id - 1
        try:
            sims = cb.similar_items(product_idx, n=n_similar)
        except Exception as e:
            st.error(f"Error: {e}")
            return

        sim_data = []
        for item_id, score in sims:
            pid = item_id + 1
            row = products[products["product_id"] == pid]
            if not row.empty:
                r = row.iloc[0]
                sim_data.append({
                    "Product ID": pid,
                    "Category": r["category"],
                    "Subcategory": r["subcategory"],
                    "Brand": r["brand"],
                    "Price": f"${r['price']:.2f}",
                    "Avg Rating": r["avg_rating"],
                    "Similarity": round(score, 4),
                })

        if sim_data:
            st.markdown("### Similar Products")
            st.dataframe(pd.DataFrame(sim_data), use_container_width=True, hide_index=True)


def page_performance():
    st.title("📊 Model Performance Comparison")

    metrics = load_metrics()
    if metrics is None:
        st.warning("No metrics found. Run `make train` first.")
        return

    # Metrics table
    st.markdown("### Evaluation Metrics (K=10)")
    table_data = []
    for model_name, m in metrics.items():
        table_data.append({
            "Model": model_name,
            "Precision@10": round(m.get("Precision@10", 0), 4),
            "Recall@10": round(m.get("Recall@10", 0), 4),
            "NDCG@10": round(m.get("NDCG@10", 0), 4),
            "MAP@10": round(m.get("MAP@10", 0), 4),
            "Hit Rate@10": round(m.get("HitRate@10", 0), 4),
            "Coverage": f"{m.get('Coverage', 0):.1%}",
            "Diversity": round(m.get("Diversity", 0), 3),
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # NDCG comparison bar chart
    models = list(metrics.keys())
    ndcg_vals = [metrics[m].get("NDCG@10", 0) for m in models]

    fig = px.bar(x=models, y=ndcg_vals, labels={"x": "Model", "y": "NDCG@10"},
                 title="NDCG@10 Comparison", color=models,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    metric_keys = ["Precision@10", "Recall@10", "NDCG@10", "MAP@10", "HitRate@10"]
    fig_radar = go.Figure()
    for model_name in models:
        values = [metrics[model_name].get(k, 0) for k in metric_keys]
        values.append(values[0])  # close the polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_keys + [metric_keys[0]],
            name=model_name,
        ))
    fig_radar.update_layout(title="Model Performance Radar", polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig_radar, use_container_width=True)

    # Display saved figures if available
    figures_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    if os.path.exists(figures_dir):
        st.markdown("### Saved Evaluation Figures")
        for fname in sorted(os.listdir(figures_dir)):
            if fname.endswith(".png"):
                st.image(os.path.join(figures_dir, fname), caption=fname, use_container_width=True)


def page_segments():
    st.title("👥 User Segment Analysis")

    users, products, interactions = load_data()

    # Interactions by age group
    merged = interactions.merge(users[["user_id", "age_group", "gender", "region"]], on="user_id")

    col1, col2 = st.columns(2)

    with col1:
        agg_age = merged.groupby(["age_group", "interaction_type"]).size().reset_index(name="count")
        fig = px.bar(agg_age, x="age_group", y="count", color="interaction_type",
                     title="Interactions by Age Group", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        agg_region = merged.groupby("region")["interaction_type"].count().reset_index(name="count")
        fig = px.pie(agg_region, values="count", names="region", title="Interactions by Region")
        st.plotly_chart(fig, use_container_width=True)

    # Category preferences by segment
    merged_cat = interactions.merge(products[["product_id", "category"]], on="product_id")
    merged_cat = merged_cat.merge(users[["user_id", "age_group"]], on="user_id")

    agg_cat = merged_cat.groupby(["age_group", "category"]).size().reset_index(name="count")
    fig = px.bar(agg_cat, x="age_group", y="count", color="category",
                 title="Category Preferences by Age Group", barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

    # Purchase funnel
    funnel_data = interactions["interaction_type"].value_counts().reset_index()
    funnel_data.columns = ["Interaction", "Count"]
    order = ["view", "click", "add_to_cart", "purchase"]
    funnel_data["Interaction"] = pd.Categorical(funnel_data["Interaction"], categories=order, ordered=True)
    funnel_data = funnel_data.sort_values("Interaction")

    fig = px.funnel(funnel_data, x="Count", y="Interaction", title="Interaction Funnel")
    st.plotly_chart(fig, use_container_width=True)


def page_explorer():
    st.title("📦 Data Explorer")

    users, products, interactions = load_data()

    tab1, tab2, tab3 = st.tabs(["Users", "Products", "Interactions"])

    with tab1:
        st.write(f"**Total Users:** {len(users):,}")
        st.dataframe(users.head(100), use_container_width=True, hide_index=True)

    with tab2:
        st.write(f"**Total Products:** {len(products):,}")
        cat_filter = st.multiselect("Filter by Category", products["category"].unique())
        if cat_filter:
            st.dataframe(products[products["category"].isin(cat_filter)].head(100),
                         use_container_width=True, hide_index=True)
        else:
            st.dataframe(products.head(100), use_container_width=True, hide_index=True)

        fig = px.histogram(products, x="category", color="category", title="Products by Category")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(products, x="category", y="price", color="category", title="Price Distribution by Category")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.write(f"**Total Interactions:** {len(interactions):,}")
        st.dataframe(interactions.head(100), use_container_width=True, hide_index=True)

        # Interactions over time
        interactions["date"] = interactions["timestamp"].dt.date
        daily = interactions.groupby("date").size().reset_index(name="count")
        fig = px.line(daily, x="date", y="count", title="Daily Interactions Over Time")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Personalized Recommendations":
    page_recommendations()
elif page == "Product Similarity":
    page_similarity()
elif page == "Model Performance":
    page_performance()
elif page == "User Segments":
    page_segments()
elif page == "Data Explorer":
    page_explorer()
