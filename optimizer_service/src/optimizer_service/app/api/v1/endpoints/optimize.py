import logging
from fastapi import APIRouter, HTTPException, Depends, status


logger = logging.getLogger(__name__)
router = APIRouter()

def optimize_allocation_loads(request: OptimizationRequest, service: OptimizerService = Depends(get_optimizer_service)) -> OptimizationResponse:
    ...
