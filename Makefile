.PHONY: install data train evaluate api dashboard clean all

install:
	pip install -r requirements.txt

data:
	python -m src.data.generate_data

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

clean:
	rm -rf data/raw/*.csv
	rm -rf data/processed/*
	rm -rf models/*.joblib
	rm -rf results/figures/*.png
	rm -rf results/metrics.json

all: install data train evaluate
