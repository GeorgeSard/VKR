.PHONY: setup lint test repro repro-force-training experiments-baseline status dag mlflow-ui mlflow-up mlflow-down verify

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e ".[dev]"

lint:
	.venv/bin/ruff check src tests scripts

test:
	.venv/bin/python -m pytest tests -q

repro:
	PATH="$(CURDIR)/.venv/bin:$$PATH" DVC_SITE_CACHE_DIR="$(CURDIR)/.cache/dvc" .venv/bin/dvc repro

repro-force-training:
	PATH="$(CURDIR)/.venv/bin:$$PATH" DVC_SITE_CACHE_DIR="$(CURDIR)/.cache/dvc" .venv/bin/dvc repro -f train_binary train_cause evaluate

experiments-baseline:
	PATH="$(CURDIR)/.venv/bin:$$PATH" DVC_SITE_CACHE_DIR="$(CURDIR)/.cache/dvc" .venv/bin/dvc repro experiments_baseline

status:
	DVC_SITE_CACHE_DIR="$(CURDIR)/.cache/dvc" .venv/bin/dvc status

dag:
	DVC_SITE_CACHE_DIR="$(CURDIR)/.cache/dvc" .venv/bin/dvc dag

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

mlflow-up:
	docker compose up -d postgres minio mlflow

mlflow-down:
	docker compose down

verify:
	.venv/bin/python scripts/verify_requirements.py
