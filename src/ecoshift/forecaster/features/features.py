import logging
from typing import List, Optional
import holidays
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ecoshift.forecaster.features.constants import (
    COUNTRY_CODE,
    LAGS_30MIN,
    ROLLING_WINDOWS_30MIN,
    TARGET_CO2,
    TARGET_PRICE,
)

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, targets: Optional[List[str]] = None):
        self.targets = targets or [TARGET_PRICE, TARGET_CO2]
        self._fr_holidays = holidays.country_holidays(COUNTRY_CODE)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        return self


    def _create_cyclical_temporal_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:

        half_hour_step = index.hour * 2 + (index.minute // 30)

        features = {
            
            "half_hour_sin": np.sin(2 * np.pi * half_hour_step / 48.0),
            "half_hour_cos": np.cos(2 * np.pi * half_hour_step / 48.0),

            "day_sin": np.sin(2 * np.pi * index.dayofweek / 7.0),
            "day_cos": np.cos(2 * np.pi * index.dayofweek / 7.0),

            "month_sin": np.sin(2 * np.pi * index.month / 12.0),
            "month_cos": np.cos(2 * np.pi * index.month / 12.0),

            "is_weekend": index.dayofweek.isin([5, 6]).astype(int),
            "is_holiday": pd.Series(index.date, index=index).isin(self._fr_holidays).astype(int),
        }
        return pd.DataFrame(features, index=index)

    @staticmethod
    def _create_lag_features(series: pd.Series, target_name: str, lags: List[int]) -> pd.DataFrame:
        features = {}
        for lag in lags:
            features[f"{target_name}_lag_{lag}"] = series.shift(lag)

        if 1 in lags and 2 in lags:
            features[f"{target_name}_diff_1_2"] = (
                features[f"{target_name}_lag_1"] - features[f"{target_name}_lag_2"]
            )
        if 48 in lags and 96 in lags:
            features[f"{target_name}_diff_48_96"] = (
                features[f"{target_name}_lag_48"] - features[f"{target_name}_lag_96"]
            )
        if 48 in lags and 336 in lags:
            features[f"{target_name}_diff_48_336"] = (
                features[f"{target_name}_lag_48"] - features[f"{target_name}_lag_336"]
            )

        return pd.DataFrame(features, index=series.index)

    @staticmethod
    def _create_rolling_features(series: pd.Series, target_name: str, rolling_windows: List[int], shift_lag: int = 1) -> pd.DataFrame:
        features = {}
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
            raise TypeError("Index du DataFrame doit être un DatetimeIndex.")

        missing_targets = [t for t in self.targets if t not in X.columns]
        if missing_targets:
            raise ValueError(
                f"Target columns missing in input dataframe : {missing_targets} -> Needed to generate target features"
            )

        cyclical_df = self._create_cyclical_temporal_features(X.index)
        target_feature_dfs = []

        for target in self.targets:
            if target not in X.columns:
                continue

            lags_df = self._create_lag_features(X[target], target, LAGS_30MIN)
            rolling_df = self._create_rolling_features(
                X[target], target, ROLLING_WINDOWS_30MIN
            )
            target_feature_dfs.extend([lags_df, rolling_df])

        df_out = pd.concat([X, cyclical_df, *target_feature_dfs], axis=1)
        
        return df_out