"""
MedSNRClassifier
================
A self-contained wrapper around the medium-SNR expert Keras model
(medium_snr_model.keras) and its companion label encoder.

The model was trained on 4-channel tensors:
    Channel 0 — I  (In-phase)
    Channel 1 — Q  (Quadrature)
    Channel 2 — Phase (unwrapped, normalised to [-1, 1])
    Channel 3 — Magnitude (normalised per sample)

The feature engineering that builds those 4 channels lives in
feature_engineering.py (same directory) and is called here automatically.

Typical usage (called by the backend's ModelRegistry):
    clf = MedSNRClassifier()
    clf.load()                           # once, at server startup
    result = clf.predict(signal_tensor)  # (1, 2, 128, 1) input
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from med_snr_model.feature_engineering import add_phase_and_magnitude

logger = logging.getLogger(__name__)

# Resolve paths relative to THIS file so the class works regardless of
# the current working directory.
_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH   = _DIR / "medium_snr_model.keras"
_DEFAULT_ENCODER_PATH = _DIR / "label_encoder.pkl"


class MedSNRClassifier:
    """
    Loads and runs the medium-SNR modulation classification model.

    Attributes:
        model_path:    Path to the .keras model file.
        encoder_path:  Path to the sklearn LabelEncoder .pkl file.
        model:         Loaded Keras model (None until load() is called).
        label_encoder: Loaded LabelEncoder (None until load() is called,
                       or if the .pkl file is missing).
    """

    def __init__(
        self,
        model_path: Path = _DEFAULT_MODEL_PATH,
        encoder_path: Path = _DEFAULT_ENCODER_PATH,
    ) -> None:
        self.model_path   = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.model: tf.keras.Model | None = None
        self.label_encoder = None  # sklearn LabelEncoder or None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the Keras model and label encoder from disk.
        Call this once at server startup before any calls to predict().
        """
        self._load_model()
        self._load_label_encoder()

    def predict(self, preprocessed: np.ndarray) -> dict[str, Any]:
        """
        Classify the modulation type of a medium-SNR signal.

        Args:
            preprocessed: (1, 2, 128, 1) float32 NumPy array produced by
                          signal_processor.preprocess_for_inference().

        Returns:
            A dict with these keys (matches the format expected by snr_router):
                modulation_available       (bool)   -- always True on success
                modulation_class           (str)    -- e.g. "QPSK"
                modulation_confidence      (float)  -- top class probability
                modulation_probabilities   (dict)   -- {class_name: probability}
                modulation_message         (None)
        """
        if self.model is None:
            raise RuntimeError(
                "MedSNRClassifier.load() must be called before predict()."
            )

        # Step 1 -- Feature engineering: (1, 2, 128, 1) -> (1, 4, 128, 1)
        model_input = add_phase_and_magnitude(preprocessed)

        # Step 2 -- Run the model
        raw_output    = self.model.predict(model_input, verbose=0)
        probabilities = raw_output[0]  # shape: (num_classes,)

        # Step 3 -- Find the winning class
        predicted_index = int(np.argmax(probabilities))
        confidence      = float(probabilities[predicted_index])

        # Step 4 -- Decode the class index into a human-readable label
        predicted_class, all_probs = self._decode(predicted_index, probabilities)

        logger.info(
            "Med-SNR AMC -> %s (%.2f%% confidence)",
            predicted_class, confidence * 100,
        )

        return {
            "modulation_available":     True,
            "modulation_class":         predicted_class,
            "modulation_confidence":    confidence,
            "modulation_probabilities": all_probs,
            "modulation_message":       None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Med-SNR Keras model not found at {self.model_path}. "
                f"Make sure medium_snr_model.keras is inside the med_snr_model/ directory."
            )
        logger.info("Loading Med-SNR model from %s ...", self.model_path)
        self.model = tf.keras.models.load_model(str(self.model_path))
        logger.info(
            "Med-SNR model loaded -- input shape: %s  output shape: %s",
            self.model.input_shape,
            self.model.output_shape,
        )

    def _load_label_encoder(self) -> None:
        if not self.encoder_path.exists():
            logger.warning(
                "Med-SNR label encoder not found at %s. "
                "Class indices will be returned as labels (e.g. 'Class_0').",
                self.encoder_path,
            )
            return
        with open(self.encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        logger.info(
            "Med-SNR label encoder loaded -- classes: %s",
            list(self.label_encoder.classes_),
        )

    def _decode(
        self,
        predicted_index: int,
        probabilities: np.ndarray,
    ) -> tuple[str, dict[str, float]]:
        """
        Convert a class index + probability array into (label, {label: prob}).

        Uses the LabelEncoder when available; falls back to 'Class_N' strings.
        """
        if self.label_encoder is not None:
            classes = list(self.label_encoder.classes_)
            predicted_class = (
                classes[predicted_index]
                if predicted_index < len(classes)
                else f"Class_{predicted_index}"
            )
            all_probs = {
                classes[i]: float(p)
                for i, p in enumerate(probabilities)
                if i < len(classes)
            }
        else:
            # Fallback: no encoder available
            predicted_class = f"Class_{predicted_index}"
            all_probs = {
                f"Class_{i}": float(p) for i, p in enumerate(probabilities)
            }

        return predicted_class, all_probs
