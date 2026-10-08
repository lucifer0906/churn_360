from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRACKING_URI,
)
from src.data.load_and_clean import (
    RANDOM_STATE,
    load_clean_featured_data,
    split_features_target,
)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def configure_mlflow(tracking_uri: str | Path, experiment_name: str) -> None:
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(experiment_name)


def log_model_run(
    model_name: str,
    estimator: Any,
    pipeline: Pipeline,
    metrics: dict[str, float],
    input_example: pd.DataFrame,
) -> str:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(estimator.get_params())
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=input_example,
        )

        return run.info.run_id


def register_best_model(results: pd.DataFrame, registered_model_name: str) -> None:
    best_row = results.sort_values(by="f1", ascending=False).iloc[0]
    model_uri = f"runs:/{best_row['run_id']}/model"
    registered_model = mlflow.register_model(model_uri, registered_model_name)

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=registered_model.version,
    )
    client.update_model_version(
        name=registered_model_name,
        version=registered_model.version,
        description=(
            "Best Stage 2 churn model selected by validation F1. "
            f"Source run model: {best_row['model']}."
        ),
    )


def train_all_models(
    data_path: str | Path | None = None,
    *,
    use_mlflow: bool = True,
    register_model: bool = True,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    registered_model_name: str = DEFAULT_MODEL_NAME,
    tracking_uri: str | Path = DEFAULT_TRACKING_URI,
) -> pd.DataFrame:
    if use_mlflow:
        configure_mlflow(tracking_uri, experiment_name)

    dataset = load_clean_featured_data(data_path)
    features, target = split_features_target(dataset)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    rows: list[dict[str, float | str]] = []
    for model_name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)
        run_id = ""

        if use_mlflow:
            run_id = log_model_run(
                model_name=model_name,
                estimator=estimator,
                pipeline=pipeline,
                metrics=metrics,
                input_example=x_train.head(5),
            )

        rows.append({"model": model_name, **metrics, "run_id": run_id})

    results = pd.DataFrame(rows).sort_values(by="f1", ascending=False)

    if use_mlflow and register_model:
        register_best_model(results, registered_model_name)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train churn models and log Stage 2 runs to MLflow."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to the IBM Telco Customer Churn CSV.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="MLflow registered model name.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=str(DEFAULT_TRACKING_URI),
        help="MLflow tracking URI. Defaults to the local SQLite database.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Run the Stage 1-style training without MLflow logging.",
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Log MLflow runs but skip model registry registration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_all_models(
        data_path=args.data_path,
        use_mlflow=not args.no_mlflow,
        register_model=not args.skip_registration,
        experiment_name=args.experiment_name,
        registered_model_name=args.model_name,
        tracking_uri=args.tracking_uri,
    )

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    print("\nStage 2 MLflow-tracked results")
    print("=" * 72)
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    if not args.no_mlflow:
        print("\nMLflow experiment:", args.experiment_name)
        print("Registered model:", args.model_name)
        print("Champion alias: champion")
        print("Open the tracking UI with:")
        print(f"  mlflow ui --backend-store-uri {args.tracking_uri}")


if __name__ == "__main__":
    main()
