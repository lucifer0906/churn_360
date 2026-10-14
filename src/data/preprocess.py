from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_and_clean import clean_telco_data, load_raw_data


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the raw Telco churn dataset.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_telco_data(load_raw_data()).to_csv(output, index=False)
    print(f"Wrote cleaned dataset to {output}")


if __name__ == "__main__":
    main()
