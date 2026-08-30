from datetime import timedelta
import logging

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

        self.forecaster = EnergyForecaster.load(model_path)
        logger.info("ML model loaded in memory !")

    def is_ready(self) -> bool:
        return self.forecaster is not None

    def predict(self, request: PredictionRequest) -> PredictResponse:
        records = [data_point.model_dump() for data_point in request.history]
        df_history = pd.DataFrame(records).set_index("timestamp").sort_index()

        predictions = self.forecaster.predict(df_history)

        last_timestamp = df_history.index[-1]
        future_timestamps = [last_timestamp + timedelta(hours=i + 1) for i in range(len(predictions))]

        forecast_points = [ForecastDataPoint(timestamp=ts, predicted_price_eur_mwh=round(pred, 2)) for ts, pred in zip(future_timestamps, predictions)]

        return PredictResponse(
            model_version=settings.VERSION,
            generated_at=pd.Timestamp.now().to_pydatetime(),
            predictions=forecast_points
        )
