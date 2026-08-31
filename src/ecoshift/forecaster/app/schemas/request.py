from datetime import datetime
from typing import List
from pydantic import BaseModel, Field

class EnergyDataPoint(BaseModel):
    timestamp: datetime
    price_eur_mwh: float = Field(..., description="Price at the demi hour T")
    load_mw: float = Field(..., description="Network consumption")


class PredictionRequest(BaseModel):
    horizon_hours: int = Field(default=24, ge=1, le=168, description="Forecasting horizon in hours")
    history: List[EnergyDataPoint] = Field(..., min_length=336, description="History required (7 days minimum) for a prediction")

