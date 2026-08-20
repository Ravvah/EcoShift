from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

class Evaluator:

    def __init__(self):
        pass

    
    def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:

        mae = mean_absolute_error(y_true, y_pred)
        pass


