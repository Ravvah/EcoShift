import argparse
import logging
import os
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd

os.environ["SKIP_ARTIFACT_VALIDATION"] = "true"

from forecast_service.app.core.config import settings
from forecast_service.features.constants import TARGET_CO2, TARGET_PRICE
from forecast_service.model.forecaster import EnergyForecaster
from forecast_service.model.search_spaces import get_lgbm_search_space
from forecast_service.model.trainer import Trainer
from forecast_service.model.tuner import Tuner
from forecast_service.tracking.mlflow_tracker import MLflowTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_pipeline")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at path: {path.absolute()}")

    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = next((col for col in ["datetime", "timestamp"] if col in df.columns), None)
        if time_col is None:
            raise ValueError("No DatetimeIndex or temporal column found in DataFrame.")
        df = df.set_index(time_col)

    return df.sort_index()


def run_pipeline(data_path: Path, n_trials: int, timeout: int) -> None:
    df = load_dataset(data_path)
    logger.info(f"Loaded dataset ({len(df)} records from {df.index.min()} to {df.index.max()})")

    tracker = MLflowTracker(
        experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    )
    
    trainer = Trainer(n_folds=5, test_size=48, lookback_steps=384) # One week of half-hour
    tuner = Tuner(trainer=trainer, n_trials=n_trials, timeout=timeout, quantile_weight=0.0)

    targets = [
        (TARGET_PRICE, settings.MLFLOW_PRICE_MODEL_NAME),
        (TARGET_CO2, settings.MLFLOW_CO2_MODEL_NAME),
    ]

    for target_col, registry_name in targets:
        logger.info(f"Executing HPO and evaluation for target: {target_col}")

        with tracker.start_run(run_name=f"optuna_lgb_{target_col}"):
            best_params = tuner.optimize(
                df=df,
                target_col=target_col,
                estimator_cls=lgb.LGBMRegressor,
                search_space_func=get_lgbm_search_space,
            )

            forecaster = EnergyForecaster(
                target_col=target_col,
                model=lgb.LGBMRegressor(**best_params),
                horizon_steps=settings.DEFAULT_FORECASTING_HORIZON
            )

            cv_report = trainer.cross_validate(df, forecaster)
            forecaster.fit(df)

            tracker.log_model_params(forecaster)
            tracker.log_cv_report(cv_report)
            tracker.log_model_native(
                forecaster=forecaster,
                model_name_registry=registry_name,
            )

            logger.info(f"Model [{target_col}] registered as '{registry_name}'")

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps training pipeline for Forecaster service")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/integrated_energy_data.parquet"),
        help="Path to input Parquet file",
    )
    parser.add_argument("--n-trials", type=int, default=15, help="Optuna trials per target")
    parser.add_argument("--timeout", type=int, default=300, help="Optuna timeout in seconds")

    args = parser.parse_args()
    run_pipeline(data_path=args.data_path, n_trials=args.n_trials, timeout=args.timeout)