from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI


DEFAULT_BEST_MODEL_PATH = PROJECT_ROOT / "reports" / "best_model.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register the selected MLflow run as the champion model."
    )
    parser.add_argument("--best-model-path", default=str(DEFAULT_BEST_MODEL_PATH))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    best = json.loads(Path(args.best_model_path).read_text(encoding="utf-8"))
    run_id = best.get("run_id")
    if not run_id:
        raise ValueError("Best model file does not contain a run_id")

    mlflow.set_tracking_uri(args.tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri, args.model_name)

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=args.model_name,
        alias="champion",
        version=registered_model.version,
    )
    client.update_model_version(
        name=args.model_name,
        version=registered_model.version,
        description=(
            "Best Stage 6 Airflow-selected churn model. "
            f"Source run model: {best['model']}."
        ),
    )
    print(
        f"Registered {args.model_name} version {registered_model.version} "
        "with alias champion"
    )


if __name__ == "__main__":
    main()
