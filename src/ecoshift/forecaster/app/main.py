from contextlib import asynccontextmanager
import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ecoshift.forecaster.app.api.v1.router import api_router
from ecoshift.forecaster.app.core.config import settings
from ecoshift.forecaster.app.services.predictor import PredictorService

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting the ecoshift forecaster service...")

    predictor_service = PredictorService()
    try:
        predictor_service.load_and_warmup()
    except Exception as e:
        logger.critical(f"Loading & Warm up Failed : Could not prepare the models : {e}")
        raise e
    
    app.state.predictor_service = predictor_service
    yield

    logger.info("Stoping the ecoshift forecaster service")
    app.state.predictor_service = None


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan
)

if settings.ALLOWED_HOSTS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ecoshift.forecaster.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.DEBUG
    )


