import logging
from pathlib import Path

import pandas as pd
from pandera.typing import DataFrame

from ecoshift.data.constants import (
    ECO2MIX_CO2_COL_RAW,
    ECO2MIX_DATE_COL_RAW,
    ENTSOE_DATE_COL_RAW,
    ENTSOE_DATE_FORMAT,
    ENTSOE_PRICE_COL_RAW,
    RESAMPLE_FREQ,
    TARGET_CO2_COL,
    TARGET_PRICE_COL,
)
from ecoshift.data.schema import EnergyDataSchema

logger = logging.getLogger(__name__)


class EnergyDataLoader:
    """
    ETL class to extract, transform, and merge ENTSO-E and eCO2mix data.
    Implements a strict Extract-Transform-Validate-Load pattern.
    """

    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # --- Pure Transformations (Static Methods) ---

    @staticmethod
    def _format_date_entsoe(df: pd.DataFrame) -> pd.DataFrame:
        datetime_str = (
            df[ENTSOE_DATE_COL_RAW]
            .astype(str)
            .str.extract(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})")[0]           
        )
        df["datetime"] = pd.to_datetime(datetime_str, format=ENTSOE_DATE_FORMAT)

        df["datetime"] = (
        df["datetime"]
        .dt.tz_localize("Europe/Paris", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
        return df.set_index("datetime")

    @staticmethod
    def _format_price_entsoe(df: pd.DataFrame) -> pd.DataFrame:
        if df[ENTSOE_PRICE_COL_RAW].dtype == "object":
            df[ENTSOE_PRICE_COL_RAW] = (
                df[ENTSOE_PRICE_COL_RAW].astype(str).str.replace(",", ".")
            )

        df = df[[ENTSOE_PRICE_COL_RAW]].rename(
            columns={ENTSOE_PRICE_COL_RAW: TARGET_PRICE_COL}
        )
        df[TARGET_PRICE_COL] = pd.to_numeric(df[TARGET_PRICE_COL], errors="coerce")
        return df

    @staticmethod
    def _format_date_eco2mix(df: pd.DataFrame) -> pd.DataFrame:
        if ECO2MIX_DATE_COL_RAW in df.columns:
            df["datetime"] = pd.to_datetime(df[ECO2MIX_DATE_COL_RAW], utc=True)
            return df.set_index("datetime")
        
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
            return df
            
        raise ValueError(f"Column {ECO2MIX_DATE_COL_RAW} missing and index is not Datetime.")

    @staticmethod
    def _format_price_eco2mix(df: pd.DataFrame) -> pd.DataFrame:
        df = df[[ECO2MIX_CO2_COL_RAW]].rename(
            columns={ECO2MIX_CO2_COL_RAW: TARGET_CO2_COL}
        )
        df[TARGET_CO2_COL] = pd.to_numeric(df[TARGET_CO2_COL], errors="coerce")
        return df

    # --- Common Utilities ---

    def _clean_time_series(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        if df.index.has_duplicates:
            dup_count = df.index.duplicated().sum()
            logger.warning(
                f"[{source_name}] Found {dup_count} duplicate timestamps. Keeping first."
            )
            df = df[~df.index.duplicated(keep="first")]

        df = df.sort_index()
        self._check_index_continuity(df)
        return df

    def _check_index_continuity(self, df: pd.DataFrame) -> None:
        if df.empty or len(df) < 2:
            return

        time_diffs = df.index.to_series().diff().dropna()
        expected_delta = pd.Timedelta(RESAMPLE_FREQ)
        gaps = time_diffs[time_diffs != expected_delta]

        if not gaps.empty:
            logger.warning(
                f"Detected {len(gaps)} gaps in the time series! "
                f"First gap at {gaps.index[0]} (size: {gaps.iloc[0]})."
            )

    # --- Orchestration ---

    def _load_and_clean_entsoe(self, filename: str) -> pd.DataFrame:
        filepath = self.raw_data_dir / filename
        logger.info(f"Processing ENTSO-E data from {filepath}")

        df = pd.read_csv(filepath)
        df = self._format_date_entsoe(df)
        df = self._format_price_entsoe(df)
        df = df.resample(RESAMPLE_FREQ).mean()
        df = self._clean_time_series(df, source_name="ENTSO-E")


        return df

    def _load_and_clean_eco2mix(self, filename: str) -> pd.DataFrame:
        filepath = self.raw_data_dir / filename
        logger.info(f"Processing eCO2mix data from {filepath}")

        df = pd.read_parquet(filepath)
        df = self._format_date_eco2mix(df)
        df = self._format_price_eco2mix(df)
        df = df.resample(RESAMPLE_FREQ).mean()
        df = self._clean_time_series(df, source_name="eCO2mix")

        return df


    def run_pipeline(
        self, entsoe_filename: str, eco2mix_filename: str, output_filename: str
    ) -> DataFrame[EnergyDataSchema]:
        logger.info("Starting Data Integration Pipeline...")

        try:
            df_prices = self._load_and_clean_entsoe(entsoe_filename)
            df_co2 = self._load_and_clean_eco2mix(eco2mix_filename)

            logger.info("Merging datasets on Datetime index...")
            df_merged = pd.merge(
                df_prices, df_co2, left_index=True, right_index=True, how="inner"
            )

            dropped_prices = len(df_prices) - len(df_merged)
            dropped_co2 = len(df_co2) - len(df_merged)

            if dropped_prices > 0 or dropped_co2 > 0:
                logger.warning(
                    f"Merge misaligned time range! Dropped {dropped_prices} price rows "
                    f"and {dropped_co2} CO2 rows. Integrated range: "
                    f"{df_merged.index.min()} -> {df_merged.index.max()}"
                )

            df_merged = df_merged.dropna()

            # Data Contract Validation
            df_validated = EnergyDataSchema.validate(df_merged)

            output_path = self.processed_data_dir / output_filename
            df_validated.to_parquet(output_path)

            logger.info(
                f"Pipeline finished successfully. Data saved to {output_path}. "
                f"Shape: {df_validated.shape}"
            )
            return df_validated

        except Exception:
            logger.exception("Data pipeline failed")
            raise


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

    loader = EnergyDataLoader(
        raw_data_dir=BASE_DIR / "data" / "raw",
        processed_data_dir=BASE_DIR / "data" / "processed",
    )

    df = loader.run_pipeline(
        entsoe_filename="gui_energy_prices_20251231-20260625.csv",
        eco2mix_filename="eco2mix_national_tr.parquet",
        output_filename="integrated_energy_data.parquet",
    )

    print(df.head())