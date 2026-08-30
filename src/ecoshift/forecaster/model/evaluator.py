from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, mean_pinball_loss
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EvaluatorReport:
    mae: float
    rmse: float
    wape: float
    directional_accuracy_pct: float
    pinball_losses: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        res = {
            "mae": self.mae,
            "rmse": self.rmse,
            "wape": self.wape,
            "directional_accuracy_pct": self.directional_accuracy_pct,
        }
        res.update(self.pinball_losses)
        return res


class Evaluator:
    def __init__(self, quantiles: List[float]):
        self.quantiles = quantiles


    @staticmethod
    def _compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        total_real = np.sum(np.abs(y_true))
        if total_real == 0:
            return np.nan
        return float((np.sum(np.abs(y_true - y_pred)) / total_real) * 100)

    @staticmethod
    def _compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:
            return np.nan
        diff_true = np.diff(y_true)
        diff_pred = np.diff(y_pred)
        return float(np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100)

    
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> EvaluatorReport:
        y_true_arr = np.asarray(y_true, dtype=np.float64)

        if len(y_true_arr) != len(y_pred):
            raise ValueError(f"Mismatched y lengths : y_true '{len(y_true_arr)} and y_pred ({len(y_pred)})" )

        mae_val = mean_absolute_error(y_true_arr, y_pred)
        rmse_val = root_mean_squared_error(y_true_arr, y_pred)

        wape_val = self._compute_wape(y_true_arr, y_pred)
        da_val = self._compute_directional_accuracy(y_true_arr, y_pred)

        pinball_dict = {
            f"pinball__q_{int(q * 100)}": round(float(mean_pinball_loss(y_true_arr, y_pred, alpha=q)), 4) for q in self.quantiles
        }

        return EvaluatorReport(
            mae=round(mae_val, 4),
            rmse=round(rmse_val, 4),
            wape=round(wape_val, 4),
            directional_accuracy_pct=round(da_val, 4),
            pinball_losses=pinball_dict,
        )

