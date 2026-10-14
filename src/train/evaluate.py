from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_PATH = Path("reports/training_results.csv")
DEFAULT_BEST_MODEL_PATH = Path("reports/best_model.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate logged model runs and select the best model."
    )
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_BEST_MODEL_PATH))
    parser.add_argument("--metric", default="f1")
    args = parser.parse_args()

    results = pd.read_csv(args.results_path)
    if args.metric not in results.columns:
        raise ValueError(f"Metric '{args.metric}' not found in {args.results_path}")

    best = results.sort_values(by=args.metric, ascending=False).iloc[0].to_dict()
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Best model by {args.metric}: {best['model']} ({best[args.metric]:.4f})")


if __name__ == "__main__":
    main()
