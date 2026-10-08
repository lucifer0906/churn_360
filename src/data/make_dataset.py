from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_and_clean import load_clean_featured_data


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_processed.csv"


def build_processed_dataset(
    data_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_clean_featured_data(data_path)
    dataset.to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the cleaned and feature-engineered Telco churn dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to the raw IBM Telco Customer Churn CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the processed CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_processed_dataset(args.data_path, args.output_path)
    print(f"Wrote processed dataset to {output}")


if __name__ == "__main__":
    main()
