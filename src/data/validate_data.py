from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_and_clean import TARGET_COLUMN, load_raw_data


REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET_COLUMN,
}
REPORT_PATH = PROJECT_ROOT / "reports" / "data_validation.json"


def main() -> None:
    data = load_raw_data()
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    churn_values = set(data[TARGET_COLUMN].dropna().astype(str).str.strip().unique())

    checks = {
        "rows_positive": data.shape[0] > 0,
        "required_columns_present": not missing_columns,
        "target_values_valid": churn_values.issubset({"Yes", "No"}),
        "missing_columns": missing_columns,
        "row_count": int(data.shape[0]),
    }
    checks["passed"] = all(
        checks[key]
        for key in ["rows_positive", "required_columns_present", "target_values_valid"]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    if not checks["passed"]:
        raise ValueError(f"Data validation failed: {checks}")

    print("Data validation passed")


if __name__ == "__main__":
    main()
