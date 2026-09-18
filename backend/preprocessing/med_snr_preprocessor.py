"""
Medium-SNR Feature Engineering
================================
The medium-SNR expert model was trained on 4-channel tensors:
    Channel 0 — I (In-phase)
    Channel 1 — Q (Quadrature)
    Channel 2 — Phase (unwrapped, normalised to [-1, 1])
    Channel 3 — Magnitude (normalised per sample)

Input expected here: the already-validated raw signal as a
(1, 2, 128, 1) NumPy array (standard backbone shape from
`preprocess_for_inference`).

Output: (1, 4, 128, 1) float32 tensor ready for model.predict().
"""

import numpy as np


def preprocess_for_med_snr(preprocessed: np.ndarray) -> np.ndarray:
    """
    Convert a (1, 2, 128, 1) tensor into the (1, 4, 128, 1) tensor
    that the medium-SNR model expects.

    How it works:
    1. Extract I and Q from the standard backbone tensor.
    2. Compute unwrapped phase, normalised to the [-1, 1] range by
       dividing by π (since arctan2 returns values in [-π, π]).
    3. Compute instantaneous magnitude and normalise each sample by
       its own maximum (+ small epsilon to avoid division by zero).
    4. Stack all four channels along axis=1 and re-add the channel dim.

    Args:
        preprocessed: The (1, 2, 128, 1) tensor produced by
                      `signal_processor.preprocess_for_inference`.

    Returns:
        A (1, 4, 128, 1) float32 NumPy array.
    """
    # preprocessed shape: (batch=1, channels=2, samples=128, depth=1)
    # We work on the spatial dims; drop the trailing depth axis first.
    X = preprocessed  # (1, 2, 128, 1)

    I = X[:, 0, :, 0]  # (1, 128)
    Q = X[:, 1, :, 0]  # (1, 128)

    # --- Phase (unwrapped, normalised) ---
    # arctan2 gives values in [-π, π]; unwrap removes discontinuities;
    # dividing by π normalises into roughly [-1, 1].
    phase = np.arctan2(Q, I)                          # (1, 128)
    phase_unwrapped = np.unwrap(phase, axis=-1)       # continuity restored
    phase_norm = phase_unwrapped / np.pi              # normalise to ≈ [-1, 1]

    # --- Magnitude (normalised per sample) ---
    magnitude = np.sqrt(I ** 2 + Q ** 2)             # (1, 128)
    magnitude = magnitude / (
        np.max(magnitude, axis=-1, keepdims=True) + 1e-8
    )

    # --- Stack into 4-channel tensor ---
    # np.stack along axis=1 gives shape (1, 4, 128)
    stacked = np.stack([I, Q, phase_norm, magnitude], axis=1)

    # Re-add the depth dimension: (1, 4, 128) → (1, 4, 128, 1)
    return stacked[..., np.newaxis].astype(np.float32)
