from typing import Any, Callable, Dict, Type
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
import logging

from ecoshift.forecaster.model.forecaster import EnergyForecaster
from ecoshift.forecaster.model.trainer import Trainer

logger = logging.getLogger(__name__)


class Tuner:
    def __init__(self, trainer: Trainer, n_trials: int, timeout: int, quantile_weight: float):
        self.trainer = trainer
        self.n_trials = n_trials
        self.timeout = timeout
        self.quantile_weight = quantile_weight

    def optimize(self, df: pd.DataFrame, target_col: str, estimator_cls: Type[BaseEstimator], search_space_func: Callable[[optuna.Trial], Dict[str, Any]]) -> Dict[str, Any]:
        model_name = estimator_cls.__name__

        def objective(trial: optuna.Trial) -> float:
            suggested_params = search_space_func(trial)
            model = estimator_cls(**suggested_params)
            forecaster = EnergyForecaster(target_col=target_col, model=model)
            cv_report = self.trainer.cross_validate(df, forecaster)

            mae = cv_report.mean_metrics["mae"]
            pinball_q_90 = cv_report.mean_metrics["pinball__q_90"]

            return mae + (self.quantile_weight * pinball_q_90)

        logger.info(f"Running HPO for model {model_name} on {self.n_trials} trials ...")
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        logger.info(f"Best composite score : {model_name} : {study.best_value:.4f}")
        logger.info(f"Best parameters for {model_name} : {study.best_params}")

        return {**study.best_params}

    
            

