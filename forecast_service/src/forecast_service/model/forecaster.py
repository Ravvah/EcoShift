from pathlib import Path
from typing import List, Optional, Self, Tuple
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from forecast_service.features.constants import TARGET_CO2, TARGET_PRICE
from forecast_service.features.features import FeatureEngineer

logger = logging.getLogger(__name__)

class EnergyForecaster(BaseEstimator):
    def __init__(self, target_col: str, model: BaseEstimator, horizon_steps: int = 48, feature_targets: Optional[List[str]] = None):
        self.target_col = target_col
        self.model = model
        self.horizon_steps = horizon_steps
        self.feature_targets = feature_targets or [TARGET_PRICE, TARGET_CO2]
        self.feature_engineer = FeatureEngineer(targets=self.feature_targets)


    def _construct_multi_output_target(self, df: pd.DataFrame) -> pd.DataFrame:
        for t in range(1, self.horizon_steps + 1):
            df[f'{self.target_col}_plus_{t}'] = df[self.target_col].shift(-t)
        return df             
            

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:

        logger.info(f"Preparing X and y for {self.target_col}")
        df = self.feature_engineer.transform(df)
        df = df.dropna()

        y_column_names = [f"{self.target_col}_plus_{t}" for t in range(1, self.horizon_steps + 1)]

        for t in range(1, self.horizon_steps + 1):
            df[f"{self.target_col}_plus_{t}"] = df[self.target_col].shift(-t)

        X = df.drop(y_column_names).values
        y = df[y_column_names].values
        return X, y


    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:

        logger.info(f"Start fit for target : {self.target_col} ...")

        self.model.fit(X, y)

        logger.info("Training of the model ended successfully !")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        check_is_fitted(self.model)

        df_features = self.feature_engineer.transform(X)

        if df_features.isnull().any().any():
            raise ValueError("Insufficient historical points. Provide at least 384 historical points")

        feature_cols = [col for col in df_features.columns]
        return self.model.predict(df_features)

    def save(self, path_str: str) -> None:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Forecaster saved : {path_str}")


    @classmethod
    def load(cls, path_str: str) -> Self:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at : {path.absolute()}")
        
        forecaster = joblib.load(path_str)
        if not isinstance(forecaster, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__} instance")
        
        return forecaster

    def clone(self) -> Self:
        new_instance = EnergyForecaster(target_col=self.target_col, model=clone(self.model))
        return new_instance
        

