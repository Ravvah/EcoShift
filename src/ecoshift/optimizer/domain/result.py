from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TimeSlotDecision:
    timestamp_decision: str
    power_allocated_kw: float
    energy_allocated_kwh: float
    price_eur_kwh: float
    co2_intensity_kg_kwh: float
    cost_eur: float
    co2_emissions_kg: float


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    solver_name: str
    total_cost_eur: float
    total_co2_emissions_kg: float
    schedule: List[TimeSlotDecision]

    
    