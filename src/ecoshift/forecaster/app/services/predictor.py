from datetime import timedelta
import logging
from typing import List

from fastapi import HTTPException, status
import pandas as pd
import anyio

from ecoshift.forecaster.app.schemas.request import PredictionRequest
from ecoshift.forecaster.app.schemas.response import ForecastDataPoint, PredictResponse
from ecoshift.forecaster.app.core.config import settings
from ecoshift.forecaster.model.forecaster import EnergyForecaster

logger = logging.getLogger(__name__)

class PredictorService:

    def __init__(self):
        model_path = settings.MODEL_PATH
        logger.info(f"Loading the ML model artifact from {model_path} ...")

        self.price_forecaster = EnergyForecaster.load(settings.PRICE_MODEL_PATH)
        self.co2_forecaster = EnergyForecaster.load(settings.CO2_MODEL_PATH)        
        self._is_healthy: bool = False


    def load_and_warmup(self) -> None:
        logger.info("Loading forecasting models...")
        self.price_forecaster = EnergyForecaster.load(settings.PRICE_MODEL_PATH)
        self.co2_forecaster = EnergyForecaster.load(settings.CO2_MODEL_PATH)          

        dummy_dates = pd.date_range(end=pd.Timestamp.now(), periods=settings.MINIMUM_HISTORY_POINTS, freq="30min")
        dummy_df = pd.DataFrame({"price_eur_mwh": [50.0] * settings.MINIMUM_HISTORY_POINTS}, index=dummy_dates)

        dummy_df = self._generate_dummy_history()

        logger.info("Execution of models warm-ups...")
        _ = self.price_forecaster.predict(dummy_df)
        _ = self.co2_forecaster.predict(dummy_df)
        logger.info("Warm-up prediction completed successfully (30min freq).")


    def _generate_dummy_history(self) -> pd.DataFrame:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=336, freq="30min")
        return pd.DataFrame(
            {
                "price_eur_mwh": [50.0] * 336,
                "co2_g_kwh": [20.0] * 336,
            },
            index=dates,
        )


    @property
    def is_ready(self) -> bool:
        return self._is_healthy and self.price_forecaster is not None and self.co2_forecaster is not None

    
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
    

    def _predict_price_sync(self, df_history: pd.DataFrame) -> List[float]:
            """Inférence synchrone CPU-Bound pour le Prix électricité."""
            return self.price_forecaster.predict(df_history).tolist()

    def _predict_co2_sync(self, df_history: pd.DataFrame) -> List[float]:
        """Inférence synchrone CPU-Bound pour le CO2."""
        return self.co2_forecaster.predict(df_history).tolist()

    async def predict(self, request: PredictionRequest) -> PredictResponse:
        records = [data_point.model_dump() for data_point in request.history]
        df_history = pd.DataFrame(records).set_index("timestamp").sort_index()

        price_preds, co2_preds = [], []

        async with anyio.create_task_group() as tg:

            async def run_price():
                nonlocal price_preds

                price_preds = await anyio.to_thread.run_sync(self._predict_price_sync, df_history)

            async def run_co2():
                nonlocal co2_preds

                co2_preds = await anyio.to_thread.run_sync(self._predict_co2_sync, df_history)

            tg.start_soon(run_price)
            tg.start_soon(run_co2)




 
