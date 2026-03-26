"""
FastAPI REST API for the Recommendation Engine
Endpoints: /recommend, /similar, /feedback, /health
"""

import os
import json
from typing import Optional

import numpy as np
import joblib
from scipy import sparse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="E-Commerce Recommendation Engine API",
    description="Multi-algorithm product recommendation system with collaborative, content-based, and hybrid approaches.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Load Models (lazy singleton)
# ---------------------------------------------------------------------------

_state = {}


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_models():
    """Load trained models and artifacts into memory."""
    if _state.get("loaded"):
        return

    root = get_project_root()
    models_dir = os.path.join(root, "models")
    results_dir = os.path.join(root, "results")

    _state["hybrid"] = joblib.load(os.path.join(models_dir, "hybrid_model.joblib"))
    _state["content_based"] = joblib.load(os.path.join(models_dir, "content_based_model.joblib"))

    # Load product metadata
    import pandas as pd
    raw_dir = os.path.join(root, "data", "raw")
    _state["products"] = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    _state["users"] = pd.read_csv(os.path.join(raw_dir, "users.csv"))

    # Load metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _state["metrics"] = json.load(f)

    _state["loaded"] = True


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class RecommendationItem(BaseModel):
    product_id: int = Field(..., description="1-indexed product ID")
    score: float = Field(..., description="Recommendation score")
    category: Optional[str] = None
    subcategory: Optional[str] = None
    price: Optional[float] = None
    brand: Optional[str] = None


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]
    model: str = "hybrid"


class SimilarItemResponse(BaseModel):
    product_id: int
    similar_products: list[RecommendationItem]


class FeedbackRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    product_id: int = Field(..., ge=1)
    interaction_type: str = Field(..., pattern="^(view|click|add_to_cart|purchase)$")
    rating: Optional[int] = Field(None, ge=1, le=5)


class FeedbackResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    n_users: int
    n_products: int


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def enrich_recommendations(recs: list) -> list[RecommendationItem]:
    """Add product metadata to recommendation tuples."""
    products = _state["products"]
    enriched = []
    for item_id, score in recs:
        pid = item_id + 1  # convert 0-indexed to 1-indexed
        row = products[products["product_id"] == pid]
        if not row.empty:
            r = row.iloc[0]
            enriched.append(RecommendationItem(
                product_id=pid,
                score=round(score, 4),
                category=r["category"],
                subcategory=r["subcategory"],
                price=float(r["price"]),
                brand=r["brand"],
            ))
        else:
            enriched.append(RecommendationItem(product_id=pid, score=round(score, 4)))
    return enriched


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=_state.get("loaded", False),
        n_users=len(_state.get("users", [])),
        n_products=len(_state.get("products", [])),
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["recommendations"])
def get_recommendations(
    user_id: int,
    n: int = Query(default=10, ge=1, le=100, description="Number of recommendations"),
):
    """Get personalized product recommendations for a user."""
    if not _state.get("loaded"):
        raise HTTPException(status_code=503, detail="Models not loaded")

    max_users = len(_state["users"])
    if user_id < 1 or user_id > max_users:
        raise HTTPException(status_code=404, detail=f"User ID must be between 1 and {max_users}")

    model = _state["hybrid"]
    user_idx = user_id - 1  # 0-indexed

    try:
        recs = model.recommend(user_idx, n=n, exclude_known=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

    enriched = enrich_recommendations(recs)
    return RecommendationResponse(user_id=user_id, recommendations=enriched, model="hybrid")


@app.get("/similar/{product_id}", response_model=SimilarItemResponse, tags=["recommendations"])
def get_similar_products(
    product_id: int,
    n: int = Query(default=10, ge=1, le=100, description="Number of similar products"),
):
    """Get similar products based on content similarity."""
    if not _state.get("loaded"):
        raise HTTPException(status_code=503, detail="Models not loaded")

    max_products = len(_state["products"])
    if product_id < 1 or product_id > max_products:
        raise HTTPException(status_code=404, detail=f"Product ID must be between 1 and {max_products}")

    cb_model = _state["content_based"]
    product_idx = product_id - 1  # 0-indexed

    try:
        sims = cb_model.similar_items(product_idx, n=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity error: {str(e)}")

    enriched = enrich_recommendations(sims)
    return SimilarItemResponse(product_id=product_id, similar_products=enriched)


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def submit_feedback(feedback: FeedbackRequest):
    """Log user feedback / interaction for future model retraining."""
    max_users = len(_state.get("users", []))
    max_products = len(_state.get("products", []))

    if feedback.user_id > max_users:
        raise HTTPException(status_code=404, detail="User not found")
    if feedback.product_id > max_products:
        raise HTTPException(status_code=404, detail="Product not found")

    # In production, this would write to an event store / queue.
    # Here we acknowledge receipt.
    return FeedbackResponse(
        status="ok",
        message=f"Feedback recorded: user={feedback.user_id}, product={feedback.product_id}, "
                f"type={feedback.interaction_type}"
    )


@app.get("/metrics", tags=["system"])
def get_metrics():
    """Return model evaluation metrics."""
    if "metrics" not in _state:
        raise HTTPException(status_code=404, detail="No metrics available. Run training first.")
    return _state["metrics"]
