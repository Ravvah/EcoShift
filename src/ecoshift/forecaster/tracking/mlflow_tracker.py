from contextlib import contextmanager
from pathlib import Path
from typing import Generator
import logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PyFuncModel

from ecoshift.forecaster.model.forecaster import EnergyForecaster
from ecoshift.forecaster.model.trainer import CrossValidationReport
from ecoshift.forecaster.tracking.pyfunc_wrapper import EnergyForecasterPyfuncWrapper

logger = logging.getLogger(__name__)

class MLflowTracker:
    def __init__(self, experiment_name: str, tracking_uri: str):
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        run = mlflow.start_run(run_name=run_name)
        try:
            logger.info(f"MLflow run started : [ID : {run.info.run_id}] - [NAME : {run_name}]")
            yield
        except Exception as e:
            logger.error(f"Error during Mlflow run: {e}")
            mlflow.set_tag("run_status", "FAILED")
            raise
        else:
            mlflow.set_tag("run_status", "SUCCESS")
            logger.info(f"MLflow run finished : [ID : {run.info.run_id}]")
        finally:
            mlflow.end_run()

    def log_model_params(self, forecaster: EnergyForecaster) -> None:
        model_params = forecaster.model.get_params(deep=False)
        mlflow.log_params(model_params)


    def log_cv_report(self, report: CrossValidationReport) -> None:
        cv_params = {
            "target_name": report.target_col,
            "model_name": report.model_name,
            "n_folds": report.n_folds
        }

        mlflow.log_params(cv_params)
        mlflow.log_metrics(report.mean_metrics)

        for i, fold_report in enumerate(report.fold_reports, start=1):
            fold_metrics = {f"fold_{i}_{k}": v for k, v in fold_report.to_dict().items()}
            mlflow.log_metrics(fold_metrics)


    # def log_model_artifact(self, forecaster: EnergyForecaster, artifact_path: str, model_name_registry: str) -> None:
    #     artifact_path = Path(artifact_path)
    #     artifact_dir = "artifacts"
    #     forecaster.save(artifact_path)

    #     mlflow.log_artifact(artifact_path, artifact_path=artifact_dir)

    #     model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_dir}/{Path(artifact_path).name}"

    #     try:
    #         model_version = mlflow.register_model(model_uri=model_uri, name=model_name_registry)
    #         logger.info(f"Model saved in MLflow Registry with name : '{model_name_registry}' in {model_uri} "
    #                     f"Model version : {model_version.version}"
    #                     )

    #         self.client.transition_model_version_stage(
    #             name=model_name_registry,
    #             version=model_version.version,
    #             stage="staging",
    #             archive_existing_versions=False
    #         )
    #     except Exception as e:
    #         logger.error(f"Error when saving model to MLflow Registry : {e}")
    #         raise
        

    def log_model_native(self, forecaster: EnergyForecaster, model_name_registry: str) -> None:
        model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=EnergyForecasterPyfuncWrapper(forecaster), registered_model_name=model_name_registry)
        logger.info(f"Model registered natively in MLflow Registry '{model_name_registry}' at URI : {model_info.model_uri}")

        latest_version = str(model_info.registered_model_version)
        self.client.set_registered_model_alias(name=model_name_registry, version=latest_version, alias="staging")
        logger.info(f"Model version {latest_version} assigned alis '@staging'")


    def load_model_from_registry(self, model_name: str, alias: str = "production") -> PyFuncModel:
        model_uri = f"models:/{model_name}@{alias}"
        logger.info(f"Loading model from MLflow Registry : {model_uri} ...")

        try:
            model = mlflow.pyfunc.load_model(model_uri=model_uri)
            logger.info(f"Successfully loaded model '{model_name}' (@{alias})")
            return model

        except Exception as e:
            logger.error(f"Failed to load model '{model_name}' (@{alias}) from MLflow Registry: {e}")
            raise e



