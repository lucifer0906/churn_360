from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Mapping

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram

from src.data.load_and_clean import load_raw_data


DRIFT_FEATURES = ("MonthlyCharges", "tenure")
DRIFT_WINDOW_SIZE = 20
EPSILON = 1e-6

PREDICTIONS_TOTAL = Counter(
    "churn360_predictions_total",
    "Total successful churn predictions returned by the API.",
    ["churn_label"],
)
FEATURE_DRIFT_PSI = Gauge(
    "churn360_feature_drift_psi",
    "Population stability index for recent prediction inputs compared with training data.",
    ["feature"],
)
FEATURE_DRIFT_DETECTED = Gauge(
    "churn360_feature_drift_detected",
    "One when feature PSI exceeds the configured drift threshold.",
    ["feature"],
)
INPUT_MONTHLY_CHARGES = Histogram(
    "churn360_input_monthly_charges",
    "MonthlyCharges values received by the prediction API.",
    buckets=(20, 40, 60, 80, 100, 120, 160, 200, 250, float("inf")),
)
INPUT_TENURE = Histogram(
    "churn360_input_tenure",
    "Tenure values received by the prediction API.",
    buckets=(0, 6, 12, 24, 36, 48, 60, 72, float("inf")),
)


def _load_baseline() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build stable histogram bins from the original training distribution."""
    try:
        raw = load_raw_data()
    except FileNotFoundError:
        raw = pd.DataFrame(
            {
                "MonthlyCharges": [20.0, 45.0, 65.0, 85.0, 110.0],
                "tenure": [1, 12, 24, 48, 72],
            }
        )

    baseline: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feature in DRIFT_FEATURES:
        values = pd.to_numeric(raw[feature], errors="coerce").dropna().to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            edges = np.array([values.min() - 0.5, values.max() + 0.5])
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(values, bins=edges)
        baseline[feature] = (edges, counts / counts.sum())
    return baseline


class DriftMonitor:
    def __init__(self) -> None:
        self.baseline = _load_baseline()
        self.values: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=DRIFT_WINDOW_SIZE)
        )
        self.lock = Lock()

    def record(self, payload: Mapping[str, object]) -> None:
        with self.lock:
            for feature in DRIFT_FEATURES:
                value = float(payload[feature])
                self.values[feature].append(value)
                self._update_feature_metrics(feature)

    def _update_feature_metrics(self, feature: str) -> None:
        recent_values = self.values[feature]
        if len(recent_values) < DRIFT_WINDOW_SIZE:
            FEATURE_DRIFT_PSI.labels(feature=feature).set(0.0)
            FEATURE_DRIFT_DETECTED.labels(feature=feature).set(0.0)
            return

        edges, baseline_distribution = self.baseline[feature]
        recent_counts, _ = np.histogram(np.asarray(recent_values), bins=edges)
        recent_distribution = recent_counts / recent_counts.sum()
        expected = np.clip(baseline_distribution, EPSILON, None)
        actual = np.clip(recent_distribution, EPSILON, None)
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))

        FEATURE_DRIFT_PSI.labels(feature=feature).set(psi)
        FEATURE_DRIFT_DETECTED.labels(feature=feature).set(float(psi >= 0.2))


monitor = DriftMonitor()


def record_prediction_input(payload: Mapping[str, object]) -> None:
    """Observe successful API inputs without allowing monitoring to affect serving."""
    try:
        INPUT_MONTHLY_CHARGES.observe(float(payload["MonthlyCharges"]))
        INPUT_TENURE.observe(float(payload["tenure"]))
        monitor.record(payload)
    except (KeyError, TypeError, ValueError):
        return


def record_prediction_outcome(churn_label: str) -> None:
    PREDICTIONS_TOTAL.labels(churn_label=churn_label).inc()
