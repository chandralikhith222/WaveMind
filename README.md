# SNR-Aware Automatic Modulation Classification

A web-based AI dashboard that classifies the modulation type of uploaded IQ signals using a two-stage deep-learning pipeline:

1. **Stage 1 — SNR Estimator**: A CNN that predicts whether the signal has **LOW**, **MEDIUM**, or **HIGH** SNR.
2. **Stage 2 — AMC Expert**: A specialised CNN + BatchNorm classifier that identifies the modulation type (currently only the **High-SNR** expert is available).

---

## Currently Implemented Models

| Model | Architecture | Input | Output |
|---|---|---|---|
| SNR Estimator | CNN (Conv2D × 3 → Dense) | `(2, 128, 1)` | 3 classes: LOW, MEDIUM, HIGH |
| High-SNR AMC | CNN + BatchNorm + Dropout | `(2, 128, 1)` | 10 modulation classes |

**Modulation classes**: 8PSK, AM-DSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM

## Future Models (Not Yet Implemented)

| Model | Status |
|---|---|
| Low-SNR AMC  | 🔧 Planned — will be added when trained |
| Medium-SNR AMC | 🔧 Planned — will be added when trained |

---

## Project Structure

```
Signal_clas/
├── backend/
│   ├── app.py                       # FastAPI application
│   ├── config.py                    # Label mappings, paths, constants
│   ├── models/
│   │   ├── snr_estimator.keras      # Stage 1: SNR estimator
│   │   └── high_snr_amc.keras       # Stage 2: High-SNR AMC
│   ├── preprocessing/
│   │   └── signal_processor.py      # Signal validation & preprocessing
│   ├── inference/
│   │   ├── model_loader.py          # Model loading & registry
│   │   └── predictor.py             # Full prediction pipeline
│   └── routing/
│       └── snr_router.py            # SNR-based model routing
│
├── frontend/
│   ├── index.html                   # Dashboard page
│   ├── css/style.css                # Design system
│   └── js/app.js                    # Upload, predict, visualise
│
├── data/
│   └── samples/                     # Test .npy samples
│       ├── sample_high_snr.npy
│       ├── sample_medium_snr.npy
│       └── sample_low_snr.npy
│
├── modulation_classes.json          # Class label mapping
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

```bash
# 1. Clone / navigate to the project
cd Signal_clas

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

## Starting the Application

```bash
# Start the backend (serves both API and frontend)
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## How to Use

1. Open the dashboard at `http://localhost:8000`.
2. **Upload** a `.npy` file containing a single IQ signal of shape `(2, 128)`.
3. Click **⚡ Predict**.
4. View the results:
   - **SNR Category** (LOW / MEDIUM / HIGH) with confidence and probability bars.
   - **Modulation Class** (if HIGH-SNR) with confidence bar.
   - **Waveform** visualization (I/Q, Magnitude, Phase tabs).

---

## API Endpoint

### `POST /predict`

Upload a `.npy` file for classification.

**Request**: `multipart/form-data` with a `file` field.

**Response** (JSON):

```json
{
    "snr_category": "HIGH",
    "snr_confidence": 0.987,
    "snr_probabilities": { "LOW": 0.001, "MEDIUM": 0.012, "HIGH": 0.987 },
    "modulation_available": true,
    "modulation_class": "QPSK",
    "modulation_confidence": 0.945,
    "modulation_probabilities": { "8PSK": 0.01, "QPSK": 0.945, ... },
    "modulation_message": null,
    "waveform": {
        "i_component": [0.001, -0.003, ...],
        "q_component": [0.002, 0.001, ...]
    }
}
```

### `GET /health`

Returns server and model status.

---

## Input Format

- **File**: NumPy `.npy` file.
- **Shape**: `(2, 128)` — row 0 is In-phase (I), row 1 is Quadrature (Q).
- **Dtype**: `float32` or `float64`.
- **Signal**: Raw IQ samples (no additional normalization required — the data is expected in the same scale as the RadioML 2016.10a dataset).

---

## Model Routing Logic

```
Upload .npy signal
       ↓
   Validate shape (2, 128)
       ↓
   Reshape → (1, 2, 128, 1)
       ↓
   SNR Estimator → argmax
       ↓
   ┌──────────┬──────────┬──────────┐
   │   HIGH   │  MEDIUM  │   LOW    │
   │   (2)    │   (1)    │   (0)    │
   └────┬─────┴────┬─────┴────┬─────┘
        ↓          ↓          ↓
  High-SNR AMC   "Not yet   "Not yet
        ↓        implemented" implemented"
  Modulation
  + Confidence
```

---

## How to Add Future AMC Models

### Adding a Low-SNR AMC Model

1. Train your Low-SNR AMC model in Colab.
2. Save it as `low_snr_amc.keras`.
3. Copy it to `backend/models/low_snr_amc.keras`.
4. Open `backend/config.py` and update `AMC_MODEL_PATHS`:

```python
AMC_MODEL_PATHS = {
    "HIGH":   HIGH_SNR_AMC_PATH,
    "MEDIUM": None,
    "LOW":    MODELS_DIR / "low_snr_amc.keras",  # ← add this
}
```

5. Restart the server. The router will automatically send LOW-SNR signals to the new model.

### Adding a Medium-SNR AMC Model

Follow the same steps, but use `medium_snr_amc.keras` and set the `"MEDIUM"` key in `AMC_MODEL_PATHS`.

> **Note**: The new model must accept the same input shape `(batch, 2, 128, 1)` and output a softmax over the same 10 modulation classes (matching `modulation_classes.json`).
