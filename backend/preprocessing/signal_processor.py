import numpy as np

from backend.config import EXPECTED_SIGNAL_SHAPE, MODEL_INPUT_CHANNELS


class SignalValidationError(Exception):
    pass


def validate_signal(signal: np.ndarray) -> None:
    if signal.size == 0:
        raise SignalValidationError("Uploaded signal is empty.")

    if signal.shape != EXPECTED_SIGNAL_SHAPE:
        raise SignalValidationError(
            f"Expected signal shape {EXPECTED_SIGNAL_SHAPE}, "
            f"got {signal.shape}. "
            f"The model requires 2 × 128 (I and Q channels × 128 time samples)."
        )

    if not np.issubdtype(signal.dtype, np.floating):
        raise SignalValidationError(
            f"Expected floating-point dtype, got {signal.dtype}. "
            f"Signal values must be float32 or float64."
        )

    if np.isnan(signal).any():
        raise SignalValidationError("Signal contains NaN values.")

    if np.isinf(signal).any():
        raise SignalValidationError("Signal contains Inf values.")


def preprocess_for_inference(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)

    signal = signal.reshape(1, *EXPECTED_SIGNAL_SHAPE, MODEL_INPUT_CHANNELS)

    return signal
