from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATHS = (
    PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",
    PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42


def find_telco_csv() -> Path:
    """Return the first expected Telco churn CSV path that exists."""
    for path in DEFAULT_DATA_PATHS:
        if path.exists():
            return path

    expected = "\n".join(f"  - {path}" for path in DEFAULT_DATA_PATHS)
    raise FileNotFoundError(
        "Could not find the IBM Telco Customer Churn CSV.\n"
        "Place it in one of these locations:\n"
        f"{expected}\n"
        "The Kaggle filename is usually WA_Fn-UseC_-Telco-Customer-Churn.csv."
    )


def load_raw_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else find_telco_csv()
    return pd.read_csv(path)


def clean_telco_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    target = df[TARGET_COLUMN]
    if pd.api.types.is_numeric_dtype(target):
        df[TARGET_COLUMN] = target.astype(int)
    else:
        df[TARGET_COLUMN] = (
            target.astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains values other than Yes/No or 0/1.")

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only binary churn values.")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def add_basic_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    if {"tenure", "MonthlyCharges"}.issubset(df.columns):
        safe_tenure = df["tenure"].replace(0, np.nan)
        df["MonthlyChargesPerTenure"] = (df["MonthlyCharges"] / safe_tenure).fillna(
            df["MonthlyCharges"]
        )

        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        ).astype(str)

    return df


def add_inference_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    return add_basic_features(feature_df)


def load_clean_featured_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    raw_df = load_raw_data(csv_path)
    clean_df = clean_telco_data(raw_df)
    return add_basic_features(clean_df)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataframe.")

    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
