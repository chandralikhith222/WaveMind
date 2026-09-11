import logging
from pathlib import Path

import tensorflow as tf

from backend.config import SNR_ESTIMATOR_PATH, AMC_MODEL_PATHS

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    pass


class ModelRegistry:
    def __init__(self) -> None:
        self.snr_estimator: tf.keras.Model | None = None
        self.amc_models: dict[str, tf.keras.Model | None] = {}

    def load_all(self) -> None:
        self._load_snr_estimator()
        self._load_amc_models()

    def _load_snr_estimator(self) -> None:
        path = SNR_ESTIMATOR_PATH
        if not path.exists():
            raise ModelLoadError(
                f"SNR estimator not found at {path}. "
                f"Place snr_estimator.keras in backend/models/."
            )
        logger.info("Loading SNR estimator from %s …", path)
        self.snr_estimator = tf.keras.models.load_model(str(path))
        logger.info(
            "SNR estimator loaded — input: %s, output: %s",
            self.snr_estimator.input_shape,
            self.snr_estimator.output_shape,
        )

    def _load_amc_models(self) -> None:
        for region, path in AMC_MODEL_PATHS.items():
            if path is None or not Path(path).exists():
                self.amc_models[region] = None
                logger.info(
                    "AMC model for %s-SNR: NOT AVAILABLE (path=%s)",
                    region, path,
                )
            else:
                logger.info("Loading %s-SNR AMC model from %s …", region, path)
                self.amc_models[region] = tf.keras.models.load_model(str(path))
                logger.info(
                    "%s-SNR AMC loaded — input: %s, output: %s",
                    region,
                    self.amc_models[region].input_shape,
                    self.amc_models[region].output_shape,
                )

    def get_amc_model(self, snr_category: str) -> tf.keras.Model | None:
        return self.amc_models.get(snr_category)

    def has_amc_model(self, snr_category: str) -> bool:
        return self.amc_models.get(snr_category) is not None
