# Churn360

Customer churn and retention intelligence project for the MLOps course.

## Stage 1: Manual Baseline

This stage is intentionally plain Python: no MLflow, Docker, DVC, Airflow, or monitoring yet.

It trains three required models on the IBM Telco Customer Churn dataset:

- Logistic Regression baseline
- Random Forest candidate
- XGBoost candidate

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the IBM Telco Customer Churn CSV at one of these paths:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
data/raw/Telco-Customer-Churn.csv
```

Then run:

```powershell
python src/train/train.py
```

Expected Stage 1 result: the script prints accuracy, precision, recall, and F1 for all three models.

To rerun without MLflow after Stage 2 changes:

```powershell
python src/train/train.py --no-mlflow
```

## Stage 2: MLflow Experiment Tracking

The same three models are now logged to MLflow with parameters, metrics, and model artifacts.
The best model by F1 is registered as `churn360-champion` with the `champion` alias.

Run:

```powershell
pip install -r requirements.txt
python src/train/train.py
```

Open the local MLflow UI:

```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

Then visit:

```text
http://127.0.0.1:5000
```

Expected Stage 2 result: MLflow shows 3 logged runs with comparable metrics, and the Models page contains one registered `churn360-champion` model version.

## Stage 3: FastAPI Serving

The API loads the MLflow registered model alias:

```text
models:/churn360-champion@champion
```

Run the API locally:

```powershell
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open the Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

```text
GET  /health
POST /predict
```

Expected Stage 3 result: `/health` returns 200, and `/predict` returns a churn label plus churn probability for a sample customer payload.

## Stage 4: Docker Containerization

Build and run the API container:

```powershell
docker compose up --build
```

The image uses `requirements-api.txt`, a smaller serving-only dependency set. Full training dependencies remain in `requirements.txt`.

Then test:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

Expected Stage 4 result: the containerized API returns the same response shape as the host-run Stage 3 API.

## Stage 5: Data Versioning with DVC

Create the processed dataset:

```powershell
python src/data/make_dataset.py
```

Restore DVC-tracked artifacts after cloning:

```powershell
dvc pull
```

Expected Stage 5 result: DVC tracks the raw dataset, processed dataset, MLflow model artifacts, and local MLflow registry database metadata.

## Stage 6: Airflow Orchestration

The Airflow DAG is `churn360_training_pipeline`.

Start Airflow:

```powershell
docker compose --profile airflow up --build airflow-init airflow-webserver airflow-scheduler
```

Open the Airflow UI:

```text
http://127.0.0.1:8080
```

Login:

```text
admin / admin
```

Trigger `churn360_training_pipeline`.

Expected Stage 6 result: the DAG completes these tasks in order: ingestion, validation, preprocessing, feature engineering, training, evaluation, and model registration.

## Stage 7: Monitoring with Prometheus and Grafana

Start the API, Prometheus, and Grafana:

```powershell
docker compose --profile monitoring up --build
```

Open the monitoring tools:

```text
FastAPI metrics: http://127.0.0.1:8000/metrics
Prometheus:      http://127.0.0.1:9090
Grafana:         http://127.0.0.1:3000
Grafana login:   admin / admin
```

Grafana automatically loads the `Churn360 API Monitoring` dashboard. It shows request rate,
p95 prediction latency, error rate, churn outcomes, and drift metrics for `MonthlyCharges` and
`tenure`.

To demonstrate drift, keep the stack running and send a shifted batch of unusually high-charge
customers:

```powershell
python scripts/simulate_drift.py --count 25
```

After Prometheus scrapes the API, the `MonthlyCharges PSI` and drift alert panels will rise.
The drift monitor uses the most recent 20 successful prediction inputs and flags PSI values of
`0.2` or higher.

## Stage 8: CI/CD with GitHub Actions

Every push and pull request to `main` runs `.github/workflows/ci.yml`. It performs unit and API
tests, checks that the best saved model has F1 at least `0.60`, and builds the API Docker image.

The CI Docker build intentionally does not need DVC artifacts. To run a functional model-serving
container after a fresh clone, restore tracked artifacts first:

```powershell
dvc pull
docker compose up --build
```
