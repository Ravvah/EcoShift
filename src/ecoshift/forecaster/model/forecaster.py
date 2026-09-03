from pathlib import Path
from typing import List, Optional, Self
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from ecoshift.forecaster.features.constants import TARGET_CO2, TARGET_PRICE
from ecoshift.forecaster.features.features import FeatureEngineer

logger = logging.getLogger(__name__)

class EnergyForecaster(BaseEstimator):
    def __init__(self, target_col: str, model: BaseEstimator, feature_targets: Optional[List[str]] = None):
        self.target_col = target_col
        self.model = model
        self.feature_targets = feature_targets or [TARGET_PRICE, TARGET_CO2]
        self.feature_engineer = FeatureEngineer(targets=self.feature_targets)
        

    def fit(self, df: pd.DataFrame) -> Self:
        logger.info(f"Start fit for target : {self.target_col} ...")

        df_features = self.feature_engineer.transform(df)
        df_clean = df_features.dropna()

        X = df_clean.drop(columns=[self.target_col])
        y = df_clean[self.target_col]

        self.feature_names_in_ = X.columns.tolist()

        self.model.fit(X, y)
        logger.info("Training of the model ended successfully !")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_names_in_"])

        df_features = self.feature_engineer.transform(df)
        df_clean = df_features.dropna()
        if df_clean.empty:
            raise ValueError("Given history is not valid")

        X_test = df_clean.drop(columns=[self.target_col])

        X_test = X_test[self.feature_names_in_]
        return self.model.predict(X_test)

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
        

