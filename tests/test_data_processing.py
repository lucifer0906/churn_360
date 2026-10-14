from __future__ import annotations

import pandas as pd

from src.data.load_and_clean import add_basic_features, clean_telco_data


def test_clean_telco_data_converts_target_and_total_charges() -> None:
    raw = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "TotalCharges": ["29.85", " "],
            "Churn": ["Yes", "No"],
        }
    )

    clean = clean_telco_data(raw)

    assert "customerID" not in clean.columns
    assert clean["Churn"].tolist() == [1, 0]
    assert clean["TotalCharges"].isna().sum() == 0
    assert clean["TotalCharges"].dtype.kind == "f"


def test_add_basic_features_creates_charge_and_tenure_features() -> None:
    features = pd.DataFrame({"tenure": [0, 10], "MonthlyCharges": [50.0, 70.0]})

    enriched = add_basic_features(features)

    assert enriched["MonthlyChargesPerTenure"].tolist() == [50.0, 7.0]
    assert enriched["TenureGroup"].tolist() == ["0-12", "0-12"]
