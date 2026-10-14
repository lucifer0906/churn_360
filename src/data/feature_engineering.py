from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_and_clean import add_basic_features


DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_processed.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add engineered features to the cleaned Telco churn dataset."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    featured = add_basic_features(pd.read_csv(args.input_path))
    featured.to_csv(output, index=False)
    print(f"Wrote feature-engineered dataset to {output}")


if __name__ == "__main__":
    main()
