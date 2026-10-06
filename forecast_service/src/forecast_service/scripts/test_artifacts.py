import pandas as pd
from forecast_service.model.forecaster import EnergyForecaster

price_model = EnergyForecaster.load("artifacts/price_model.joblib")
co2_model = EnergyForecaster.load("artifacts/co2_model.joblib")

df_dummy = pd.DataFrame({
    "price_eur_mwh": [50.0] * 384,
    "co2_intensity_g_kwh": [20.0] * 384
}, index=pd.date_range("2026-01-01", periods=384, freq="30min"))

p_preds = price_model.predict(df_dummy)
c_preds = co2_model.predict(df_dummy)

assert len(p_preds) == 48, "Devrait prédire 48 pas"
assert len(c_preds) == 48, "Devrait prédire 48 pas"
print(" Smoke test réussi ! Les artefacts sont prêts.")