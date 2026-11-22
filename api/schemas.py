from datetime import date
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class InvoicePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., ge=50.0, le=250000.0)
    payment_terms_days: int = Field(...)
    line_item_count: int = Field(..., ge=1, le=15)
    invoice_date: date
    due_date: date

    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("amount must be a numeric value")
        if not math.isfinite(float(value)):
            raise ValueError("amount must be finite")
        return value

    @field_validator("payment_terms_days", mode="before")
    @classmethod
    def validate_payment_terms(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("payment_terms_days must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("payment_terms_days must be an integer")
        if int(value) not in (15, 30, 45, 60):
            raise ValueError("payment_terms_days must be one of 15, 30, 45, or 60")
        return value

    @field_validator("line_item_count", mode="before")
    @classmethod
    def validate_line_item_count(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("line_item_count must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("line_item_count must be an integer")
        return value

    @model_validator(mode="after")
    def validate_dates_and_terms(self) -> "InvoicePredictionRequest":
        if date(2025, 1, 1) > self.invoice_date or self.invoice_date > date(2025, 12, 31):
            raise ValueError("invoice_date must fall within 2025")
        if self.due_date > date(2026, 3, 1):
            raise ValueError("due_date exceeds the supported maximum date of 2026-03-01")
        if self.due_date < self.invoice_date:
            raise ValueError("due_date cannot be earlier than invoice_date")
        if (self.due_date - self.invoice_date).days != self.payment_terms_days:
            raise ValueError("payment_terms_days must match the difference between due_date and invoice_date")
        return self


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    fraud_flag: bool
    prediction: Literal["Fraudulent", "Legitimate"]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str
    last_training_date: str
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
