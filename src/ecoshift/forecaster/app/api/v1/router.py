from fastapi import APIRouter

from ecoshift.forecaster.app.api.v1.endpoints import health, predict

api_router = APIRouter()

api_router.include_router(
    health.router,
    tags=["Monitoring & Health"]
)

api_router.include_router(
    predict.router,
    tags=["Forecasting"]
)