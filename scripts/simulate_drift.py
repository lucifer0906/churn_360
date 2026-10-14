from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


SHIFTED_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 72,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 250.0,
    "TotalCharges": 18000.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send high-charge customer inputs to demonstrate input drift."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()

    body = json.dumps(SHIFTED_CUSTOMER).encode("utf-8")
    for _ in range(args.count):
        request = Request(args.url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"Prediction request failed with HTTP {response.status}")

    print(f"Sent {args.count} shifted prediction inputs to {args.url}")


if __name__ == "__main__":
    main()
