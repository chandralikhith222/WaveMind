import io
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.inference.model_loader import ModelRegistry, ModelLoadError
from backend.inference.predictor import predict, PredictionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

model_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting SNR-Aware AMC server …")
    logger.info("=" * 60)
    try:
        model_registry.load_all()
        logger.info("All models loaded successfully.")
    except ModelLoadError as exc:
        logger.error("FATAL: %s", exc)
        raise
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="SNR-Aware Automatic Modulation Classification",
    description=(
        "Upload an IQ signal (.npy) to estimate its SNR region "
        "and classify its modulation type."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "snr_estimator_loaded": model_registry.snr_estimator is not None,
        "med_snr_classifier_loaded": model_registry.med_snr_classifier is not None,
        "amc_models": {
            region: (model is not None)
            for region, model in model_registry.amc_models.items()
        },
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a .npy file.",
        )

    try:
        contents = await file.read()
        signal = np.load(io.BytesIO(contents), allow_pickle=False)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded file. "
                "Make sure it is a valid .npy file saved with numpy.save()."
            ),
        )

    try:
        result = predict(signal, model_registry)
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction. Please try again.",
        )

    return result


@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port)

