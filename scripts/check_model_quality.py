from __future__ import annotations

import argparse
import csv
from pathlib import Path


def best_f1_score(results_path: Path) -> float:
    with results_path.open(newline="", encoding="utf-8") as results_file:
        rows = list(csv.DictReader(results_file))

    if not rows:
        raise ValueError(f"No model results found in {results_path}")

    return max(float(row["f1"]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail when the best model F1 score is below the promotion threshold."
    )
    parser.add_argument("--results-path", default="reports/training_results.csv")
    parser.add_argument("--min-f1", type=float, default=0.60)
    args = parser.parse_args()

    score = best_f1_score(Path(args.results_path))
    print(f"Best F1 score: {score:.4f}; required minimum: {args.min_f1:.4f}")
    if score < args.min_f1:
        raise SystemExit("Model quality gate failed: best F1 is below the required minimum.")

    print("Model quality gate passed.")


if __name__ == "__main__":
    main()
