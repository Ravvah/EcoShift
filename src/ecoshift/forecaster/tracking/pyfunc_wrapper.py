from mlflow.pyfunc import PythonModel
import pandas as pd

from ecoshift.forecaster.model.forecaster import EnergyForecaster


class EnergyForecasterPyfuncWrapper(PythonModel):
    def __init__(self, forecaster: EnergyForecaster):
        self.forecaster = forecaster

    def predict(self, context, model_input: pd.DataFrame):
        return self.forecaster.predict(model_input)
