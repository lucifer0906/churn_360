from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "mlflow.db"
HOST_PREFIX = "file:C:/Users/cheta/Desktop/FOLDERS/VIT/TY/MLOPS/CP"
CONTAINER_PREFIX = "file:///app"


def rewrite_table_column(
    cursor: sqlite3.Cursor,
    table_name: str,
    column_name: str,
) -> None:
    cursor.execute(
        f"""
        UPDATE {table_name}
        SET {column_name} = REPLACE({column_name}, ?, ?)
        WHERE {column_name} LIKE ?
        """,
        (HOST_PREFIX, CONTAINER_PREFIX, f"{HOST_PREFIX}%"),
    )


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"MLflow database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        rewrite_table_column(cursor, "experiments", "artifact_location")
        rewrite_table_column(cursor, "logged_models", "artifact_location")
        rewrite_table_column(cursor, "model_versions", "storage_location")
        connection.commit()


if __name__ == "__main__":
    main()
