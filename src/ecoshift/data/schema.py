import pandera.pandas as pa
from pandera.typing import Series

class EnergyDataSchema(pa.DataFrameModel):
    price_eur_mwh: Series[float] = pa.Field(
        ge=-500, le=3000, nullable=False, description="Day-Ahead Electricity Price"
    )
    co2_intensity_g_kwh: Series[float] = pa.Field(
        ge=0, le=1500, nullable=False, description="CO2 Intensity"
    )

    class Config:
        strict = True
        coerce = True