from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.config import MODULATION_CLASSES
from backend.inference.model_loader import ModelRegistry

logger = logging.getLogger(__name__)

NOT_IMPLEMENTED_MESSAGES: dict[str, str] = {
    "LOW": (
        "Low-SNR AMC model not implemented yet. "
        "A specialised low-SNR expert classifier will be added in a future version."
    ),
}


def route_prediction(
    snr_category: str,
    preprocessed_signal: np.ndarray,
    model_registry: ModelRegistry,
) -> dict[str, Any]:
    if not model_registry.has_amc_model(snr_category):
        message = NOT_IMPLEMENTED_MESSAGES.get(
            snr_category,
            f"{snr_category}-SNR AMC model is not available.",
        )
        logger.info("No AMC model for %s-SNR — returning placeholder.", snr_category)
        return {
            "modulation_available": False,
            "modulation_class": None,
            "modulation_confidence": None,
            "modulation_probabilities": None,
            "modulation_message": message,
        }

    amc_model = model_registry.get_amc_model(snr_category)

    # ------------------------------------------------------------------
    # MEDIUM-SNR path: delegate entirely to MedSNRClassifier.
    # It handles 4-channel feature engineering, model.predict(), and
    # label decoding internally, then returns the standard result dict.
    # ------------------------------------------------------------------
    if snr_category == "MEDIUM":
        return model_registry.med_snr_classifier.predict(preprocessed_signal)

    # ------------------------------------------------------------------
    # HIGH-SNR (and any future) path: raw 2-channel input, global dict.
    # ------------------------------------------------------------------
    raw_output = amc_model.predict(preprocessed_signal, verbose=0)
    probabilities = raw_output[0]  # shape (num_classes,)

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    predicted_class = MODULATION_CLASSES.get(predicted_index, f"Class_{predicted_index}")
    all_probs = {
        MODULATION_CLASSES.get(i, f"Class_{i}"): float(p)
        for i, p in enumerate(probabilities)
    }

    logger.info(
        "%s-SNR AMC -> %s (%.2f%% confidence)",
        snr_category, predicted_class, confidence * 100,
    )

    return {
        "modulation_available": True,
        "modulation_class": predicted_class,
        "modulation_confidence": confidence,
        "modulation_probabilities": all_probs,
        "modulation_message": None,
    }
