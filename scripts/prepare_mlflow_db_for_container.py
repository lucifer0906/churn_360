from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "mlflow.db"
CONTAINER_PREFIX = "file:///app"
HOST_PREFIXES = (
    "file:C:/Users/cheta/Desktop/FOLDERS/VIT/TY/MLOPS/CP",
    "file:///C:/Users/cheta/Desktop/FOLDERS/VIT/TY/MLOPS/CP",
)


def rewrite_table_column(
    cursor: sqlite3.Cursor,
    table_name: str,
    column_name: str,
    host_prefix: str,
) -> None:
    cursor.execute(
        f"""
        UPDATE {table_name}
        SET {column_name} = REPLACE({column_name}, ?, ?)
        WHERE {column_name} LIKE ?
        """,
        (host_prefix, CONTAINER_PREFIX, f"{host_prefix}%"),
    )


def main() -> None:
    if not DB_PATH.exists():
        print("No local MLflow database found; skipping artifact URI preparation.")
        return

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        for host_prefix in HOST_PREFIXES:
            rewrite_table_column(
                cursor, "experiments", "artifact_location", host_prefix
            )
            rewrite_table_column(
                cursor, "logged_models", "artifact_location", host_prefix
            )
            rewrite_table_column(
                cursor, "model_versions", "storage_location", host_prefix
            )
        connection.commit()


if __name__ == "__main__":
    main()
