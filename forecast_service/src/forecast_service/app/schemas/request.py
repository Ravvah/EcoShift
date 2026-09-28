from datetime import datetime
from typing import List
from pydantic import BaseModel, Field

class EnergyDataPoint(BaseModel):
    timestamp: datetime
    price_eur_mwh: float = Field(..., description="Price at the demi hour T in €/MWh")
    co2_intensity_g_kwh: float = Field(..., description="C02 intensity at the demi hour g/kWh")


class PredictionRequest(BaseModel):
    horizon_hours: int = Field(default=24, ge=1, le=168, description="Forecasting horizon in hours, fixed to 24 for V1 solution")
    history: List[EnergyDataPoint] = Field(..., 
                                           min_length=384, #because of rolling 24h mean of the oldest 7 th day (336 + 48)
                                           description="History required (384 points) for a prediction")

