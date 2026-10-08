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
