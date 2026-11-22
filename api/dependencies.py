from dataclasses import dataclass
from typing import Any

import joblib

from src.predict_invoices import load_config, resolve_model_artifact


MODEL_VERSION = "1.0.0"
LAST_TRAINING_DATE = "unknown"


@dataclass
class ModelState:
    pipeline: Any | None = None
    threshold: float | None = None
    error: str | None = None

    def load(self) -> None:
        try:
            config = load_config()
            threshold = float(config["threshold"])
            model_path = resolve_model_artifact(config)
            pipeline = joblib.load(model_path)
            classes = getattr(pipeline, "classes_", None)
            if classes is None or classes.tolist() != [0, 1]:
                raise ValueError("Saved pipeline classes_ must be exactly [0, 1].")
            if not callable(getattr(pipeline, "predict_proba", None)):
                raise ValueError("Saved pipeline does not provide predict_proba().")
            self.pipeline = pipeline
            self.threshold = threshold
            self.error = None
        except Exception as exc:
            self.pipeline = None
            self.threshold = None
            self.error = type(exc).__name__
