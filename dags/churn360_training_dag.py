from __future__ import annotations

from datetime import datetime
import os

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/project"
PYTHON = os.getenv("CHURN360_PYTHON", "python")


default_args = {
    "owner": "churn360",
    "retries": 1,
}


with DAG(
    dag_id="churn360_training_pipeline",
    description="Ingest, validate, preprocess, train, evaluate, and register Churn360 models.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@monthly",
    catchup=False,
    tags=["churn360", "mlops"],
) as dag:
    ingest = BashOperator(
        task_id="ingestion",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/data/ingest.py",
    )

    validate = BashOperator(
        task_id="validation",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/data/validate_data.py",
    )

    preprocess = BashOperator(
        task_id="preprocessing",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/data/preprocess.py",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/data/feature_engineering.py",
    )

    train = BashOperator(
        task_id="training",
        bash_command=(
            f"cd {PROJECT_DIR} && {PYTHON} src/train/train.py "
            "--data-path data/processed/telco_churn_processed.csv "
            "--skip-registration "
            "--results-path reports/training_results.csv"
        ),
    )

    evaluate = BashOperator(
        task_id="evaluation",
        bash_command=(
            f"cd {PROJECT_DIR} && {PYTHON} src/train/evaluate.py "
            "--results-path reports/training_results.csv "
            "--output-path reports/best_model.json"
        ),
    )

    register = BashOperator(
        task_id="model_registration",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/train/register_model.py",
    )

    ingest >> validate >> preprocess >> feature_engineering >> train >> evaluate >> register
