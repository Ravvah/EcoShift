# src/ecoshift/data/constants.py

# --- ENTSO-E Data Constants ---
ENTSOE_DATE_COL_RAW = "MTU (CET/CEST)"
ENTSOE_PRICE_COL_RAW = "Day-ahead Price (EUR/MWh)"
ENTSOE_DATE_FORMAT = "%d/%m/%Y %H:%M:%S"

# --- eCO2mix Data Constants ---
# Remplace par le vrai nom de la colonne date dans ton fichier parquet si différent
ECO2MIX_DATE_COL_RAW = "date_heure" 
ECO2MIX_CO2_COL_RAW = "taux_co2"

# --- Standardized Output Constants ---
TARGET_PRICE_COL = "price_eur_mwh"
TARGET_CO2_COL = "co2_intensity_g_kwh"
RESAMPLE_FREQ = "1h"