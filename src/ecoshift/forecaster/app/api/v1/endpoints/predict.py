import logging
from fastapi import HTTPException, APIRouter, Depends, status

from ecoshift.forecaster.app.api.dependencies import get_predictor_service
from ecoshift.forecaster.app.schemas.request import PredictionRequest
from ecoshift.forecaster.app.schemas.response import PredictResponse
from ecoshift.forecaster.app.services.predictor import PredictorService
from ecoshift.forecaster.app.core.security import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


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
    except ValueError as ve:
        logger.warning(f"Validation error during inference: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.exception("Unexpected error during inference execution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during inference : {str(e)}"
        )