from typing import Dict
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit

from ecoshift.forecaster.model.forecaster import EnergyForecaster
from ecoshift.forecaster.model.evaluator import Evaluator

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, n_splits: int, test_size: int):
        self.n_splits = n_splits
        self.test_size = test_size
        self.evaluator = Evaluator()

    #TODO: Continue the method
    def cross_validate(self, df: pd.DataFrame, forecaster: EnergyForecaster) -> Dict[str, float]:
        target_col = forecaster.target_col
        logger.info(f"Running Cross Validation with {self.n_splits} folds ..."
                    f"for {forecaster.model.__class__.__name__} model on {target_col} target")

        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            fold_forecaster = forecaster.clone()

            fold_forecaster.fit(train_df)

            y_val_pred = fold_forecaster.predict(val_df)
            y_val_true = val_df[target_col]

            # metrics = self.evaluator.evaluate(y_val_true, y_val_pred)
            pass


        

