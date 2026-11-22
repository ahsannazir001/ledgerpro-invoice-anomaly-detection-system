from contextlib import asynccontextmanager
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import LAST_TRAINING_DATE, MODEL_VERSION, ModelState
from api.schemas import HealthResponse, InvoicePredictionRequest, PredictionResponse
from src.predict_invoices import predict_with_pipeline


model_state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state.load()
    yield


app = FastAPI(
    title="LedgerPro Invoice Anomaly Detection API",
    description="Fraud probability and threshold-based invoice anomaly prediction using the frozen LedgerPro model.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid request",
            "errors": [
                {"location": list(error["loc"]), "message": error["msg"]}
                for error in exc.errors()
            ],
        },
    )


@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 64 * 1024:
        return JSONResponse(status_code=413, content={"detail": "Request body exceeds the size limit"})
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = model_state.pipeline is not None and model_state.threshold is not None
    if not loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "model_version": MODEL_VERSION,
                "last_training_date": LAST_TRAINING_DATE,
                "threshold": None,
            },
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
        "last_training_date": LAST_TRAINING_DATE,
        "threshold": model_state.threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(invoice: InvoicePredictionRequest):
    if model_state.pipeline is None or model_state.threshold is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Prediction service is unavailable"},
        )

    record = pd.DataFrame([{
        "amount": invoice.amount,
        "payment_terms_days": invoice.payment_terms_days,
        "line_item_count": invoice.line_item_count,
        "invoice_date": invoice.invoice_date.isoformat(),
        "due_date": invoice.due_date.isoformat(),
    }])

    try:
        result = predict_with_pipeline(record, model_state.pipeline, model_state.threshold)
        probability = float(result.loc[0, "fraud_probability"])
        fraud_flag = bool(result.loc[0, "is_fraud_prediction"] == 1)
    except (ValueError, TypeError, AttributeError, KeyError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Model prediction failed"},
        )
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Prediction service is unavailable"},
        )

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Model returned an invalid probability"},
        )

    return {
        "fraud_probability": probability,
        "threshold": float(model_state.threshold),
        "fraud_flag": fraud_flag,
        "prediction": "Fraudulent" if fraud_flag else "Legitimate",
        "model_version": MODEL_VERSION,
    }
