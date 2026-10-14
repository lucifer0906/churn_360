from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_NAME = "churn360-baselines"
DEFAULT_MODEL_NAME = "churn360-champion"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


def normalize_local_mlflow_artifact_uris() -> None:
    """Use valid Windows file URIs so MLflow writes artifacts under ``mlruns``."""
    database_path = PROJECT_ROOT / "mlflow.db"
    if not database_path.exists():
        return

    columns = (
        ("experiments", "artifact_location"),
        ("runs", "artifact_uri"),
        ("logged_models", "artifact_location"),
        ("model_versions", "storage_location"),
    )
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        for table_name, column_name in columns:
            cursor.execute(
                f"""
                UPDATE {table_name}
                SET {column_name} = REPLACE({column_name}, 'file:C:/', 'file:///C:/')
                WHERE {column_name} LIKE 'file:C:/%'
                """
            )
        connection.commit()
