import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import holidays

from ecoshift.forecaster.features.constants import COUNTRY_CODE, TARGET_PRICE, TARGET_CO2, LAGS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Class to generate time-series features for energy forecasting."""

    def __init__(self, targets: List[str] = None):
        self.targets = targets or [TARGET_PRICE, TARGET_CO2]


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        return self

    @staticmethod
    def _create_cyclical_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
        logger.info("Creating cyclical temporal features...")

        fr_holidays = holidays.country_holidays(COUNTRY_CODE)

        features = {
                "hour_sin": np.sin(2 * np.pi * index.hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * index.hour / 24.0),
                
                "day_sin": np.sin(2 * np.pi * index.dayofweek / 7.0),
                "day_cos": np.cos(2 * np.pi * index.dayofweek / 7.0),
                
                "month_sin": np.sin(2 * np.pi * index.month / 12.0),
                "month_cos": np.cos(2 * np.pi * index.month / 12.0),
                
                "is_weekend": index.dayofweek.isin([5, 6]).astype(int),
                "is_holiday": pd.Series(index.date, index=index).isin(fr_holidays).astype(int),
            }

        return pd.DataFrame(features, index=index)

    @staticmethod
    def _create_lag_features(series: pd.Series, target_name: str, lags: List[str]) -> pd.DataFrame:
        logger.info("Creating lag features...")
        features = {}

        for lag in lags:
            features[f"{target_name}_lag_{lag}"] = series.shift(lag)

        if 1 in lags and 2 in lags:
            features[f"{target_name}_diff_1_2"] = (
                features[f"{target_name}_lag_1"] - features[f"{target_name}_lag_2"]
            )
            
        if 24 in lags and 48 in lags:
            features[f"{target_name}_diff_24_48"] = (
                features[f"{target_name}_lag_24"] - features[f"{target_name}_lag_48"]
            )

        return pd.DataFrame(features, index=series.index)

    @staticmethod
    def _create_rolling_features(
        series: pd.Series,
        target_name: str,
        rolling_windows: List[int],
        shift_lag: int = 1,
    ) -> pd.DataFrame:
        
        logger.info("Creating rolling features...")
        features = {}

        # no-leakage
        base_series = series.shift(shift_lag)

        for window in rolling_windows:
            features[f"{target_name}_roll_mean_{window}"] = (
                base_series.rolling(window=window).mean()
            )
            features[f"{target_name}_roll_std_{window}"] = (
                base_series.rolling(window=window).std()
            )

        return pd.DataFrame(features, index=series.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
       
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Index of dataframe need to be DatetimeIndex.")

        logger.info("Creating feature set...")

        cyclical_df = self._create_cyclical_temporal_features(X.index)

        target_feature_dfs = []
        
        for target in self.targets:
            if target not in X.columns:
                logger.warning(f"Target column '{target}'not in dataframe.")
                continue

            lags_df = self._create_lag_features(X[target], target, LAGS)
            rolling_df = self._create_rolling_features(X[target], target, ROLLING_WINDOWS)
            
            target_feature_dfs.extend([lags_df, rolling_df])

        df_out = pd.concat([X, cyclical_df, *target_feature_dfs], axis=1)

        # NaN of last lag
        initial_len = len(df_out)
        df_out = df_out.dropna()
        
        logger.info(
            f"Feature engineering ended. Shape: {df_out.shape} | "
            f"NaN deleted : {initial_len - len(df_out)}"
        )

        return df_out
