# 🛒 E-Commerce Product Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit)

A **multi-algorithm recommendation system** combining collaborative filtering, content-based filtering, and hybrid approaches with a rigorous evaluation framework. Built with classical ML and statistical methods — no LLMs or generative AI.

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐           │
│  │  Users.csv   │  │ Products.csv │  │ Interactions.csv │           │
│  │  (10K users) │  │ (5K products)│  │ (~500K events)   │           │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘           │
│         └─────────────────┼───────────────────┘                      │
│                           ▼                                          │
│              ┌────────────────────────┐                              │
│              │   Feature Engineering  │                              │
│              │  • User-Item Matrix    │                              │
│              │  • TF-IDF Vectors      │                              │
│              │  • User/Item Features  │                              │
│              └───────────┬────────────┘                              │
├──────────────────────────┼───────────────────────────────────────────┤
│                    MODEL LAYER                                       │
│     ┌────────────┬───────┴────────┬─────────────┐                   │
│     ▼            ▼                ▼             ▼                   │
│ ┌────────┐ ┌──────────┐  ┌────────────┐ ┌───────────┐              │
│ │User-CF │ │ Item-CF  │  │Content-Based│ │   ALS/SVD │              │
│ │Cosine  │ │ Cosine   │  │  TF-IDF     │ │  Matrix   │              │
│ │Sim.    │ │ Sim.     │  │  Cosine     │ │  Factor.  │              │
│ └───┬────┘ └────┬─────┘  └─────┬──────┘ └─────┬─────┘              │
│     └───────────┴───────────────┴──────────────┘                     │
│                         ▼                                            │
│              ┌─────────────────────┐                                │
│              │  Hybrid Recommender │                                │
│              │  • Score Blending   │                                │
│              │  • Re-ranking       │                                │
│              │  • Cold-start       │                                │
│              └──────────┬──────────┘                                │
├─────────────────────────┼────────────────────────────────────────────┤
│                   SERVING LAYER                                      │
│     ┌───────────────────┼──────────────────┐                        │
│     ▼                                      ▼                        │
│ ┌──────────────┐                  ┌─────────────────┐               │
│ │  FastAPI      │                  │   Streamlit     │               │
│ │  REST API     │                  │   Dashboard     │               │
│ │  /recommend   │                  │   Visualize     │               │
│ │  /similar     │                  │   Explore       │               │
│ └──────────────┘                  └─────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

| Model                | NDCG@10 | MAP@10 | Hit Rate@10 | Coverage | Diversity |
|----------------------|---------|--------|-------------|----------|-----------|
| User-Based CF        | 0.142   | 0.098  | 0.321       | 18.4%    | 0.847     |
| Item-Based CF        | 0.158   | 0.112  | 0.347       | 22.1%    | 0.812     |
| ALS (Matrix Factor.) | 0.203   | 0.151  | 0.425       | 35.6%    | 0.789     |
| SVD                  | 0.195   | 0.143  | 0.411       | 31.2%    | 0.801     |
| Content-Based        | 0.121   | 0.084  | 0.289       | 48.7%    | 0.913     |
| **Hybrid (Best)**    | **0.231** | **0.172** | **0.468** | **42.3%** | **0.856** |

> Results on synthetic dataset with temporal train/test split (80/20).

---

## 🔧 Tech Stack

| Component          | Technology                        |
|--------------------|-----------------------------------|
| Core ML            | scikit-learn, scipy, implicit, surprise |
| Data Processing    | pandas, numpy                     |
| API                | FastAPI, uvicorn, pydantic        |
| Dashboard          | Streamlit, plotly                 |
| Visualization      | matplotlib, seaborn, plotly       |
| Serialization      | joblib                            |
| Containerization   | Docker, docker-compose            |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/naresh-sampangi/ecommerce-recommendation-engine.git
cd ecommerce-recommendation-engine
make install
```

### Full Pipeline

```bash
# Generate synthetic data
make data

# Train all models and evaluate
make train

# Generate evaluation plots
make evaluate

# Start API server
make api

# Start Streamlit dashboard
make dashboard
```

### Docker

```bash
docker-compose up --build
```

---

## 📁 Project Structure

```
ecommerce-recommendation-engine/
├── data/
│   ├── raw/                    # Generated synthetic data
│   └── processed/              # Feature matrices
├── models/                     # Saved trained models
├── results/
│   ├── figures/                # Evaluation plots
│   └── metrics.json            # Model comparison metrics
├── src/
│   ├── data/
│   │   └── generate_data.py    # Synthetic data generation
│   ├── features/
│   │   └── build_matrices.py   # Feature engineering
│   ├── models/
│   │   ├── collaborative_filtering.py
│   │   ├── content_based.py
│   │   ├── hybrid.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── api/
│   │   └── app.py              # FastAPI REST API
│   └── dashboard/
│       └── app.py              # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📖 Methodology

### Collaborative Filtering
- **User-Based CF**: Finds similar users via cosine similarity on interaction vectors, recommends items those neighbors liked.
- **Item-Based CF**: Computes item-item cosine similarity, recommends items similar to what the user has interacted with.
- **ALS (Alternating Least Squares)**: Matrix factorization via the `implicit` library, optimized for implicit feedback.
- **SVD**: Singular Value Decomposition via the `surprise` library for explicit rating prediction.

### Content-Based Filtering
- Builds TF-IDF vectors from product category, subcategory, brand, and price range features.
- Recommends products similar to a user's historical preferences using cosine similarity.

### Hybrid Approach
- Weighted score fusion from collaborative and content-based models.
- Re-ranking layer incorporating item popularity and recency.
- Cold-start handling: content-based fallback for new users, popularity-based for new items.

### Evaluation
- Temporal train/test split (chronological — last 20% of interactions).
- Metrics: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K, Coverage, Diversity.
- Popularity bias and cold-start performance analysis.

---

## 👤 Author

**Naresh Sampangi**

---

## 📄 License

This project is licensed under the MIT License.
