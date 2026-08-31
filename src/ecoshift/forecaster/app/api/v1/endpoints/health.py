from fastapi import APIRouter, Depends, status, Response
from ecoshift.forecaster.app.services.predictor import PredictorService

router = APIRouter()

@router.get(
        "/healthz",
        status_code=status.HTTP_200_OK,
        summary="Liveness Probe"
            )
async def healthz():
    return {"status": "ok"}


@router.get(
        "/readyz",
        summary="Readiness Probe"
)
async def readyz(response: Response):
    from ecoshift.forecaster.app.main import predictor_service
    if predictor_service is None or not predictor_service.is_ready():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready"}
    return {"status": "ready"}