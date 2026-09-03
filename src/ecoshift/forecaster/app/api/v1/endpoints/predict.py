from fastapi import HTTPException, APIRouter, Depends, status

from ecoshift.forecaster.app.schemas.request import PredictionRequest
from ecoshift.forecaster.app.schemas.response import PredictResponse
from ecoshift.forecaster.app.services.predictor import PredictorService
from ecoshift.forecaster.app.core.security import verify_api_key

router = APIRouter()

def get_predictor_service() -> PredictorService:
    from ecoshift.forecaster.app.main import predictor_service
    if predictor_service is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not ready."
        )
    return predictor_service

@router.post(
        "/predict",
        response_model=PredictResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
        summary="Generate a prediction of the energy : electricity price and co2 quantity"
)
async def predict_energy_signals(request: PredictionRequest, service: PredictorService = Depends(get_predictor_service)) -> PredictResponse:
    try:
        return await service.predict(request=request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during inference : {str(e)}"
        )