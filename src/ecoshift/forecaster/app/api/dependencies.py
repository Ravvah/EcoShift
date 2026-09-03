from fastapi import HTTPException, Request, status
from ecoshift.forecaster.app.services.predictor import PredictorService


def get_predictor_service(request: Request) -> PredictorService | None:
    service: PredictorService | None = getattr(request.app.state, "predictor_service", None)
    if service is None or not service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Le service d'inférence n'est pas prêt ou les modèles ne sont pas chargés.",
            )

    return service