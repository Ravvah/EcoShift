from pydantic import BaseModel, Field


class OptimizationRequest(BaseModel):
    energy_required_kw: float = Field(..., description="Energy requested by the ")
    ...