from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI
from src.data.load_and_clean import add_inference_features


MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{DEFAULT_MODEL_NAME}@{MODEL_ALIAS}"

BinaryYesNoType = Literal["Yes", "No"]
GenderType = Literal["Female", "Male"]
InternetServiceType = Literal["DSL", "Fiber optic", "No"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerFeatures(BaseModel):
    gender: GenderType = "Female"
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: BinaryYesNoType = "Yes"
    Dependents: BinaryYesNoType = "No"
    tenure: int = Field(default=12, ge=0, le=72)
    PhoneService: BinaryYesNoType = "Yes"
    MultipleLines: Literal["No phone service", "No", "Yes"] = "No"
    InternetService: InternetServiceType = "Fiber optic"
    OnlineSecurity: Literal["No internet service", "No", "Yes"] = "No"
    OnlineBackup: Literal["No internet service", "No", "Yes"] = "No"
    DeviceProtection: Literal["No internet service", "No", "Yes"] = "No"
    TechSupport: Literal["No internet service", "No", "Yes"] = "No"
    StreamingTV: Literal["No internet service", "No", "Yes"] = "No"
    StreamingMovies: Literal["No internet service", "No", "Yes"] = "No"
    Contract: ContractType = "Month-to-month"
    PaperlessBilling: BinaryYesNoType = "Yes"
    PaymentMethod: PaymentMethodType = "Electronic check"
    MonthlyCharges: float = Field(default=70.35, ge=0)
    TotalCharges: float = Field(default=845.5, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.5,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    churn_label: Literal["Yes", "No"]
    churn_probability: float
    model_uri: str


app = FastAPI(
    title="Churn360 API",
    version="0.3.0",
    description="Stage 3 FastAPI service for customer churn prediction.",
)


@lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


def payload_to_features(payload: CustomerFeatures) -> pd.DataFrame:
    row = payload.model_dump()
    features = pd.DataFrame([row])
    return add_inference_features(features)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    try:
        model = load_model()
        features = payload_to_features(payload)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionResponse(
        prediction=prediction,
        churn_label="Yes" if prediction == 1 else "No",
        churn_probability=probability,
        model_uri=MODEL_URI,
    )
