from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.config import SNR_LABELS
from backend.inference.model_loader import ModelRegistry
from backend.preprocessing.signal_processor import (
    validate_signal,
    preprocess_for_inference,
    SignalValidationError,
)
from backend.routing.snr_router import route_prediction

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    pass


def predict(
    raw_signal: np.ndarray,
    model_registry: ModelRegistry,
) -> dict[str, Any]:
    try:
        validate_signal(raw_signal)
    except SignalValidationError as exc:
        raise PredictionError(str(exc)) from exc

    try:
        preprocessed = preprocess_for_inference(raw_signal)
    except Exception as exc:
        raise PredictionError(f"Preprocessing failed: {exc}") from exc

    try:
        snr_output = model_registry.snr_estimator.predict(preprocessed, verbose=0)
        snr_probs = snr_output[0]

        snr_index = int(np.argmax(snr_probs))
        snr_category = SNR_LABELS.get(snr_index, f"UNKNOWN_{snr_index}")
        snr_confidence = float(snr_probs[snr_index])

        snr_probabilities = {
            SNR_LABELS.get(i, f"Class_{i}"): float(p)
            for i, p in enumerate(snr_probs)
        }

        logger.info(
            "SNR estimation → %s (%.2f%% confidence)",
            snr_category, snr_confidence * 100,
        )
    except Exception as exc:
        raise PredictionError(f"SNR estimation failed: {exc}") from exc

    try:
        modulation_result = route_prediction(
            snr_category, preprocessed, model_registry
        )
    except Exception as exc:
        raise PredictionError(f"Modulation classification failed: {exc}") from exc

    waveform = {
        "i_component": raw_signal[0].tolist(),
        "q_component": raw_signal[1].tolist(),
    }

    return {
        "snr_category": snr_category,
        "snr_confidence": round(snr_confidence, 4),
        "snr_probabilities": {
            k: round(v, 4) for k, v in snr_probabilities.items()
        },
        **modulation_result,
        "waveform": waveform,
    }
