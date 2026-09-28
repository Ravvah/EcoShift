from dataclasses import dataclass, field
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from forecast_service.model.evaluator import Evaluator, EvaluatorReport
from forecast_service.model.forecaster import EnergyForecaster
from forecast_service.features.constants import ROLLING_WINDOWS_30MIN, LAGS_30MIN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossValidationReport:
    target_col: str
    model_name: str
    n_folds: int
    fold_reports: List[EvaluatorReport]
    mean_metrics: Dict[str, float] = field(default_factory=dict)


class Trainer:

    def __init__(self, n_folds: int, test_size: int, lookback_steps: int = 384, quantiles: List[float] = None):
        self.n_folds = n_folds
        self.test_size = test_size
        self.lookback_steps = lookback_steps
        self.evaluator = Evaluator(quantiles=quantiles or [0.1, 0.5, 0.9])

    @staticmethod
    def _aggregate_fold_reports(reports: List[EvaluatorReport]) -> Dict[str, float]:
        raw_dicts = [r.to_dict() for r in reports]
        keys = raw_dicts[0].keys()
        return {
            key: round(float(np.mean([d[key] for d in raw_dicts if not np.isnan(d[key])])), 4)
            for key in keys
        }





    def cross_validate(
        self, df: pd.DataFrame, forecaster: EnergyForecaster
    ) -> CrossValidationReport:
        target_column_name = forecaster.target_col
        horizon_steps = forecaster.horizon_steps
        model_name = forecaster.model.__class__.__name__

        df = forecaster._construct_multi_output_target(df)
        df = df.dropna()

        #to refactor, moved X and y to energy forecaster class for Single Responsability Design 

        gap_window_size = int(max(LAGS_30MIN, ROLLING_WINDOWS_30MIN) + 1) # add gap ? 
        tscv = TimeSeriesSplit(n_splits=self.n_folds, test_size=self.test_size, 
                            #    gap=gap_window_size 
                               )
        fold_reports = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
            X_train, X_test = X[train_idx], X[val_idx]
            y_train, y_test = y[train_idx], y[val_idx]

            fold_forecaster = forecaster.clone()
            fold_forecaster.fit(X_train)

            fold_forecaster.predict(X_test)

            # lookback_df = train_df.iloc[-self.lookback_steps :] # lookback useful to calculate first validation features

            # predict_input_df = pd.concat([lookback_df, val_df])

            predict_input_df = val_df.iloc[-fold_forecaster.horizon_steps :]

            y_val_pred = fold_forecaster.predict(predict_input_df)

            # y_val_pred = y_pred_all[-len(val_df) :]
            
            y_val_true = val_df[target_column_name].values

            report = self.evaluator.evaluate(y_val_true, y_val_pred)
            fold_reports.append(report)

            logger.info(
                f"Fold {fold+1}/{self.n_folds} - MAE: {report.mae:.4f} | "
                f"RMSE: {report.rmse:.4f} | DA: {report.directional_accuracy_pct:.2f}%"
            )

        mean_metrics = self._aggregate_fold_reports(fold_reports)

        return CrossValidationReport(
            target_col=target_column_name,
            model_name=model_name,
            n_folds=self.n_folds,
            fold_reports=fold_reports,
            mean_metrics=mean_metrics,
        )