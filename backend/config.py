import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SNR_ESTIMATOR_PATH = MODELS_DIR / "snr_estimator.keras"
HIGH_SNR_AMC_PATH  = MODELS_DIR / "high_snr_amc.keras"
MED_SNR_AMC_PATH   = MODELS_DIR / "medium_snr_amc.keras"
MED_SNR_LABEL_ENC  = MODELS_DIR / "medium_snr_label_encoder.pkl"
MODULATION_CLASSES_PATH = BASE_DIR.parent / "modulation_classes.json"

EXPECTED_SIGNAL_SHAPE = (2, 128)
MODEL_INPUT_CHANNELS = 1

SNR_LABELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
}

def _load_modulation_classes() -> dict[int, str]:
    if MODULATION_CLASSES_PATH.exists():
        with open(MODULATION_CLASSES_PATH, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    else:
        return {
            0: "8PSK",
            1: "AM-DSB",
            2: "BPSK",
            3: "CPFSK",
            4: "GFSK",
            5: "PAM4",
            6: "QAM16",
            7: "QAM64",
            8: "QPSK",
            9: "WBFM",
        }


MODULATION_CLASSES = _load_modulation_classes()

AMC_MODEL_PATHS: dict[str, Path | None] = {
    "HIGH":   HIGH_SNR_AMC_PATH,
    "MEDIUM": None,   # Handled by MedSNRClassifier (med_snr_model/classifier.py)
    "LOW":    None,
}
