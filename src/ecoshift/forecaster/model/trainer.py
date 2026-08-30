from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit

from ecoshift.forecaster.model.forecaster import EnergyForecaster
from ecoshift.forecaster.model.evaluator import Evaluator, EvaluatorReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossValidationReport:
    target_col: str
    model_name: str
    n_folds: int
    fold_reports: List[EvaluatorReport]
    mean_metrics: Dict[str, float] = field(default_factory=Dict)


class Trainer:
    def __init__(self, n_folds: int, test_size: int):
        self.n_folds = n_folds
        self.test_size = test_size
        self.evaluator = Evaluator()

    @staticmethod
    def _aggregate_fold_reports(reports: List[EvaluatorReport]) -> Dict[str, float]:
        raw_dicts = [r.to_dict() for r in reports]
        keys = raw_dicts[0].keys()

        return {
            key: round(np.mean([d[key] for d in raw_dicts if not np.isnan(d[key])]), 4) for key in keys
        }

    def cross_validate(self, df: pd.DataFrame, forecaster: EnergyForecaster) -> CrossValidationReport:
        target_col = forecaster.target_col
        model_name = forecaster.model.__class__.__name__
        logger.info(f"Running Cross Validation with {self.n_folds} folds ..."
                    f"for {model_name} model on {target_col} target")

        tscv = TimeSeriesSplit(n_splits=self.n_folds, test_size=self.test_size)
        fold_reports = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            fold_forecaster = forecaster.clone()

            fold_forecaster.fit(train_df)

            y_val_pred = fold_forecaster.predict(val_df)
            
            y_val_true = val_df[target_col]
            y_val_true_aligned = y_val_true.iloc[-len(y_val_pred) :]

            evaluation_report = self.evaluator.evaluate(y_val_true_aligned, y_val_pred)
            fold_reports.append(evaluation_report)

            logger.info(
                f"Fold {fold} / {self.n_folds} - MAE : {evaluation_report.mae:.4f} | "
                f"RMSE : {evaluation_report.rmse:.4f} | WAPE : {evaluation_report.wape:.2f}% | DA : {evaluation_report.directional_accuracy_pct:.2f}% "
            )
        mean_metrics = self._aggregate_fold_reports(fold_reports)

        logger.info(f"Cross Validation finished for model {model_name} on {target_col} target")

        return CrossValidationReport(
                target_col=target_col,
                model_name=model_name,
                n_folds=self.n_folds,
                fold_reports=fold_reports,
                mean_metrics=mean_metrics
            )


        

