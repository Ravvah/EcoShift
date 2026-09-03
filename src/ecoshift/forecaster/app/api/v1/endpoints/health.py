from fastapi import APIRouter, Depends, status, Response
from ecoshift.forecaster.app.api.dependencies import get_predictor_service
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
async def readyz(response: Response, service: PredictorService | None = Depends(get_predictor_service)):
    if service is None or not service.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready"}
    return {"status": "ready"}