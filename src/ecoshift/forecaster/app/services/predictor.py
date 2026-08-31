from datetime import timedelta
import logging

from fastapi import HTTPException, status
import pandas as pd

from ecoshift.forecaster.app.schemas.request import PredictionRequest
from ecoshift.forecaster.app.schemas.response import ForecastDataPoint, PredictResponse
from ecoshift.forecaster.app.core.config import settings
from ecoshift.forecaster.model.forecaster import EnergyForecaster

logger = logging.getLogger(__name__)

class PredictorService:

    def __init__(self):
        model_path = settings.MODEL_PATH
        logger.info(f"Loading the ML model artifact from {model_path} ...")

        self.forecaster: EnergyForecaster | None = None
        self._is_healthy: bool = False
        self._load_and_warmup()


    def _load_and_warmup(self) -> None:
        try:
            self.forecaster = EnergyForecaster.load(settings.MODEL_PATH)
            logger.info("ML model loaded in memory !")

            dummy_dates = pd.date_range(end=pd.Timestamp.now(), periods=settings.MINIMUM_HISTORY_POINTS, freq="30min")
            dummy_df = pd.DataFrame({"price_eur_mwh": [50.0] * settings.MINIMUM_HISTORY_POINTS}, index=dummy_dates)

            _ = self.forecaster.predict(dummy_df)
            self._is_healthy = True
            logger.info("Warm-up prediction completed successfully (30min freq).")

        except Exception as e:
            self._is_healthy = False
            logger.error(f"Failed to initialize ML model service: {e}")


    def is_ready(self) -> bool:
        return self.forecaster is not None and self._is_healthy

    @staticmethod
    def _verify_history(df_history: pd.DataFrame) -> pd.DataFrame:
        full_index = pd.date_range(start=df_history.index.min(), end=df_history.index.max(), freq="30min")

        missing_steps = len(full_index) - len(df_history)

        if missing_steps > 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provided history have {missing_steps} temporal missing steps. We cannot garantee a valid forecast. Please correct the time series"
            )

        df_history = df_history.reindex(full_index).ffill(limit=2)
        return df_history

    def predict(self, request: PredictionRequest) -> PredictResponse:
        records = [data_point.model_dump() for data_point in request.history]
        df_history = pd.DataFrame(records).set_index("timestamp").sort_index()
        df_history = PredictorService._verify_history(df_history)

        predictions = self.forecaster.predict(df_history)

        last_timestamp = df_history.index[-1]
        future_timestamps = [last_timestamp + timedelta(minutes=30 * (i + 1)) for i in range(len(predictions))]

        forecast_points = [ForecastDataPoint(timestamp=ts, predicted_price_eur_mwh=round(pred, 2)) for ts, pred in zip(future_timestamps, predictions)]

        return PredictResponse(
            model_version=settings.VERSION,
            generated_at=pd.Timestamp.now().to_pydatetime(),
            predictions=forecast_points
        )
