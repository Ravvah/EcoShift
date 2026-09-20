from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass(frozen=True)
class TimeSlotDecision:
    timestep_decision: int
    power_allocated_kw: float
    energy_allocated_kwh: float
    price_eur_kwh: float
    co2_intensity_kg_kwh: float
    cost_eur: float
    co2_emissions_kg: float


class OptimizationStatus(str, Enum):
    SUCCESS = "success"
    INFEASIBLE = "infeasible"
    ITERATION_LIMIT = "iteration_limit"
    UNBOUNDED = "unbounded"
    NUMERICAL_DIFFICULTY = "numerical_difficulty"
    OTHER_ERROR = "other_error"

class SolverName(str, Enum):
    SCIPY_HIGHS = "scipy_highs"




@dataclass(frozen=True)
class OptimizationResult:
    status: OptimizationStatus
    solver_name: SolverName
    total_cost_eur: float
    total_co2_emissions_kg: float
    schedule: List[TimeSlotDecision]

    
    