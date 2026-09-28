from fastapi import APIRouter

from forecast_service.app.api.v1.endpoints import predict
from forecast_service.app.api.v1.endpoints import health

api_router = APIRouter()

api_router.include_router(
    health.router,
    tags=["Monitoring & Health"]
)

api_router.include_router(
    predict.router,
    tags=["Forecasting"]
)