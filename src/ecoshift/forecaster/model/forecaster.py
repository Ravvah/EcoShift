from pathlib import Path
from typing import Self
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone

from ecoshift.forecaster.features.features import FeatureEngineer

logger = logging.getLogger(__name__)

class EnergyForecaster:

    def __init__(self, target_col: str, model: BaseEstimator):
        self.target_col = target_col
        self.model = model

        self.pipeline = self.build_pipeline()
        pass

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "feature_engineer",
                    FeatureEngineer(targets=[self.target_col]),
                ),
                (
                    "model",
                    self.model
                )
            ]
        )


    def fit(self, df: pd.DataFrame) -> Self:
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        self.pipeline.fit(X, y)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=[self.target_col])
        return self.pipeline.predict(X)

    def save(self, path_str: str) -> None:

        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Forecaster saved : {path_str}")


    def load(cls, path_str: str) -> Self:
        forecaster = joblib.load(path_str)
        if not isinstance(forecaster, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return forecaster

    def clone(self) -> Self:
        new_instance = EnergyForecaster(target_col=self.target_col, model=clone(self.model))
        return new_instance
        

