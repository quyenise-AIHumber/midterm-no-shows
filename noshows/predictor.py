"""Load the saved model once and turn appointments into decisions."""

import json
import math
from pathlib import Path

import joblib

from .appointment import Appointment


class NoShowPredictor:
    """One object owns the fitted pipeline and its decision threshold."""

    def __init__(self, project_root: Path | None = None):
        root = (
            Path(__file__).resolve().parents[1]
            if project_root is None
            else Path(project_root)
        )
        info_path = root / "models" / "model_info.json"
        model_path = root / "models" / "no_show.joblib"

        self.model_info = json.loads(info_path.read_text(encoding="utf-8"))
        self.threshold = float(self.model_info["threshold"])
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold: invalid model metadata")

        saved_features = (
            self.model_info["numeric_features"]
            + self.model_info["categorical_features"]
        )
        if saved_features != list(Appointment.FIELDS):
            raise ValueError("features: model metadata and Appointment disagree")

        self.model = joblib.load(model_path)

    def should_call(self, probability: float) -> bool:
        """Call at or above the threshold saved with the model."""
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability: must be between 0 and 1")
        return probability >= self.threshold

    def predict(self, appointment: Appointment) -> dict:
        """Return probabilities, action, threshold and any warning."""
        if not isinstance(appointment, Appointment):
            raise TypeError("appointment: an Appointment object is required")

        classes = list(self.model.classes_)
        if 1 not in classes:
            raise ValueError("model: missing the no-show class")
        no_show_index = classes.index(1)
        risk = float(self.model.predict_proba(appointment.to_frame())[0, no_show_index])
        call = self.should_call(risk)

        warnings = []
        if appointment.distance_km is None:
            warnings.append("distance_km: missing; filled using training-data median")

        return {
            "probability_no_show": risk,
            "probability_attend": 1 - risk,
            "call_patient": call,
            "action": "Call patient to confirm" if call else "No call needed",
            "threshold": self.threshold,
            "warnings": warnings,
        }

    def __repr__(self) -> str:
        name = self.model_info.get("model_name", "unknown")
        return f"NoShowPredictor(model={name!r}, threshold={self.threshold:.2f})"
