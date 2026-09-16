# 📡 WaveMind: SNR-Aware Automatic Modulation Classification (AMC)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Chart.js](https://img.shields.io/badge/Frontend-Vanilla_JS_%2B_Chart.js-F5788D?logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, two-stage Radio Frequency (RF) Signal Intelligence (SIGINT) system designed to classify raw In-Phase & Quadrature (I/Q) digital and analog wireless modulations across varying Signal-to-Noise Ratio (SNR) regimes.

Equipped with a FastAPI inference engine and a cyberpunk tactical terminal dashboard featuring real-time oscillogram decomposition (I/Q, Magnitude, and Phase).

---

## 🔬 Core Concept: Why SNR-Aware AMC?

In wireless communications and electronic warfare, **Automatic Modulation Classification (AMC)** identifies how a received signal was modulated without prior knowledge of transmission parameters.

Standard deep learning AMC models typically suffer from severe performance degradation under low or fluctuating SNR conditions:
- **The Problem**: A single monolithic neural network trained across all SNR levels tends to overfit high-SNR features or make erratic predictions when noise dominates low-SNR signals.
- **The Solution (Hierarchical Routing)**: WaveMind decomposes the classification problem into two distinct stages:
  1. **Stage 1 (SNR Estimator)**: Rapidly quantifies the RF environment into **LOW**, **MEDIUM**, or **HIGH** SNR regimes.
  2. **Stage 2 (Regime-Specific Expert Classifiers)**: Routes the preprocessed signal to a neural network trained specifically for that noise condition.

```
       Uploaded Raw IQ Signal (.npy)
                     │
                     ▼
          [ Validation & Reshape ]
              (2, 128) ➔ (1, 2, 128, 1)
                     │
                     ▼
       ┌───────────────────────────┐
       │   Stage 1: SNR Estimator  │
       │     (Deep 2D CNN Model)   │
       └─────────────┬─────────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   [ LOW SNR ]  [ MEDIUM SNR ] [ HIGH SNR ]
       │             │             │
       ▼             ▼             ▼
  Expert Model  Expert Model  Expert Model
  (Planned)     (Planned)     (High-SNR CNN)
                                   │
                                   ▼
                      Modulation Classification
                       + Confidence Distribution
                                   │
                                   ▼
                      Waveform Decomposition
                       (I/Q, Magnitude, Phase)
```

---

## 🎯 Supported Modulation Schemes

The classifier is configured to recognize 10 foundational analog and digital modulation formats (aligned with standard benchmarks such as RadioML 2016.10a):

| Class Index | Modulation Type | Category | Common Use Cases |
|:-----------:|:----------------|:---------|:-----------------|
| `0` | **8PSK** | Digital Phase Modulation | Satellite links, DVB-S2 |
| `1` | **AM-DSB** | Analog Amplitude Modulation | Aviation voice comms, broadcast |
| `2` | **BPSK** | Digital Phase Modulation | Deep space comms, RFID, telemetry |
| `3` | **CPFSK** | Continuous Phase FSK | Bluetooth LE, tactical datalinks |
| `4` | **GFSK** | Filtered Frequency Shift | Bluetooth classic, DECT |
| `5` | **PAM4** | Pulse Amplitude Modulation | High-speed Ethernet, PCIe 6.0 |
| `6` | **QAM16** | Quadrature Amplitude Mod | Wi-Fi (802.11a/g/n), cellular 3G/4G |
| `7` | **QAM64** | Quadrature Amplitude Mod | Digital Cable TV, Wi-Fi 5/6, LTE |
| `8` | **QPSK** | Digital Phase Modulation | GPS L1, LTE cellular uplink, satellite |
| `9` | **WBFM** | Wideband Frequency Mod | Commercial FM radio broadcasting |

---

## 🚀 Key Features

- **⚡ Two-Stage Inference Pipeline**: Automatic estimation of SNR category followed by dynamic routing to specialized AMC expert models.
- **🎛️ Interactive Cyberpunk Oscilloscope**: Real-time canvas rendering of signal buffers using Chart.js, switchable between:
  - **I / Q**: In-phase ($I$) and Quadrature ($Q$) amplitude components.
  - **Magnitude**: Instantaneous envelope $|s(t)| = \sqrt{I(t)^2 + Q(t)^2}$.
  - **Phase**: Instantaneous phase trajectory $\theta(t) = \arctan2(Q(t), I(t))$.
- **🛡️ Strict Input Validation**: Comprehensive server-side checks validating tensor dimensions `(2, 128)`, floating-point representations, and bounds checking for `NaN`/`Inf` anomalies.
- **🔌 Extensible Architecture**: Plug-and-play model registry enabling seamless drop-in additions of Low-SNR and Medium-SNR models without modifying core server logic.
- **🧪 Built-in Test Vectors**: Includes pre-extracted sample `.npy` files (`sample_high_snr.npy`, `sample_medium_snr.npy`, `sample_low_snr.npy`) for instant validation.

---

## 📁 Project Directory Structure

```
Signal_clas/
├── backend/
│   ├── app.py                     # FastAPI REST API & static file server
│   ├── config.py                  # Label mappings, directory paths, and model registry config
│   ├── inference/
│   │   ├── model_loader.py        # ModelRegistry (loads and verifies Keras models)
│   │   └── predictor.py           # End-to-end prediction coordinator
│   ├── preprocessing/
│   │   └── signal_processor.py    # NumPy array validation, type-casting & reshaping
│   ├── routing/
│   │   └── snr_router.py          # SNR-conditioned expert classifier dispatcher
│   └── models/
│       ├── snr_estimator.keras    # Stage 1: SNR estimator CNN
│       └── high_snr_amc.keras     # Stage 2: High-SNR AMC expert CNN
│
├── frontend/
│   ├── index.html                 # Tactical HUD Web interface
│   ├── css/
│   │   └── style.css              # Cyberpunk HUD styling & animations
│   └── js/
│       └── app.js                 # Drag-and-drop ingest, API bridge & Chart.js renderer
│
├── data/
│   └── samples/                   # Ready-to-test .npy signal files
│       ├── sample_high_snr.npy
│       ├── sample_medium_snr.npy
│       ├── sample_low_snr.npy
│       └── sample_generic.npy
│
├── modulation_classes.json        # Class index to human-readable modulation name mapping
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.10+**
- (Optional but recommended) Virtual environment manager (`venv` or `conda`)

### 1. Clone or Open the Workspace
```bash
cd Signal_clas
```

### 2. Create and Activate a Virtual Environment
- **Windows (Command Prompt / PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

Start the local server using Uvicorn:

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```
*Alternatively, run `python backend/app.py` directly.*

Open your web browser and navigate to:
```
https://wavemind-1.onrender.com/
```

---

## 📡 Usage Guide

### Using the Web Interface
1. Open the dashboard at `https://wavemind-1.onrender.com/`.
2. Drag and drop any `.npy` file from `data/samples/` into the upload zone (or click to browse).
3. Click **`> EXECUTE PREDICTION`**.
4. Inspect the outputs:
   - **SNR Regime**: Displays estimated SNR tier (`HIGH`, `MEDIUM`, or `LOW`) and percentage breakdown.
   - **Modulation Type**: Predicted scheme (e.g. `QPSK`) with confidence rating.
   - **Signal Oscillogram**: Switch between **I/Q**, **Magnitude**, and **Phase** tabs to visually inspect the RF envelope.

### Generating Custom Test Signals in Python

If you wish to synthesize your own test array for upload:

```python
import numpy as np

# Create a 2 x 128 synthetic BPSK signal with small noise
t = np.linspace(0, 1, 128, dtype=np.float32)
i_channel = np.sign(np.sin(2 * np.pi * 5 * t)) + 0.05 * np.random.randn(128).astype(np.float32)
q_channel = np.zeros(128, dtype=np.float32) + 0.05 * np.random.randn(128).astype(np.float32)

signal = np.stack([i_channel, q_channel], axis=0)  # Shape: (2, 128)
np.save("my_test_signal.npy", signal)
print("Saved my_test_signal.npy with shape:", signal.shape)
```

---

## 🌐 API Documentation

FastAPI provides automatic interactive Swagger documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Primary Endpoints

#### 1. `POST /predict`
Uploads a binary `.npy` file and executes hierarchical classification.

- **Content-Type**: `multipart/form-data`
- **Payload**: Form field `file` containing the `.npy` binary.
- **Sample cURL Request**:
  ```bash
  curl -X POST "http://localhost:8000/predict" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@data/samples/sample_high_snr.npy"
  ```
- **Sample JSON Response**:
  ```json
  {
    "snr_category": "HIGH",
    "snr_confidence": 0.9872,
    "snr_probabilities": {
      "LOW": 0.0011,
      "MEDIUM": 0.0117,
      "HIGH": 0.9872
    },
    "modulation_available": true,
    "modulation_class": "QPSK",
    "modulation_confidence": 0.9421,
    "modulation_probabilities": {
      "8PSK": 0.012,
      "AM-DSB": 0.003,
      "BPSK": 0.005,
      "CPFSK": 0.001,
      "GFSK": 0.002,
      "PAM4": 0.004,
      "QAM16": 0.015,
      "QAM64": 0.009,
      "QPSK": 0.9421,
      "WBFM": 0.0068
    },
    "modulation_message": null,
    "waveform": {
      "i_component": [0.034, 0.051, "... 128 float values"],
      "q_component": [-0.012, 0.022, "... 128 float values"]
    }
  }
  ```

#### 2. `GET /health`
Returns system status and loaded model availability.
```json
{
  "status": "healthy",
  "snr_estimator_loaded": true,
  "amc_models": {
    "HIGH": true,
    "MEDIUM": false,
    "LOW": false
  }
}
```

---

## 🧩 Adding Future AMC Expert Models

The system is architected for modular expansion. To plug in a newly trained model (e.g., for **LOW** or **MEDIUM** SNR):

1. **Save Model**: Export the trained model in Keras format: `low_snr_amc.keras` or `medium_snr_amc.keras`.
   - *Requirement*: Expected input tensor shape is `(None, 2, 128, 1)` and output is a softmax vector of 10 classes matching `modulation_classes.json`.
2. **Place Model**: Move the file into `backend/models/`:
   ```
   backend/models/low_snr_amc.keras
   ```
3. **Register Path**: In `backend/config.py`, update `AMC_MODEL_PATHS`:
   ```python
   LOW_SNR_AMC_PATH = MODELS_DIR / "low_snr_amc.keras"

   AMC_MODEL_PATHS: dict[str, Path | None] = {
       "HIGH": HIGH_SNR_AMC_PATH,
       "MEDIUM": None,
       "LOW": LOW_SNR_AMC_PATH,  # Updated from None
   }
   ```
4. **Reload**: Restart the server. Signals classified as `LOW` will now automatically route to your new expert model!

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
