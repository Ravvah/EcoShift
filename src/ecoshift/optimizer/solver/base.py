from abc import ABC, abstractmethod
from typing import List

from ecoshift.optimizer.domain.constraints import LoadConstraint
from ecoshift.optimizer.domain.result import OptimizationResult

class BaseSolver(ABC):

    def __init__(self, time_limit_seconds):
        self.time_limit_seconds = time_limit_seconds


    @abstractmethod
    def solve(self, price_eur_kwh_list: List[float], co2_intensity_kg_kwh_list: List[float], load_constraint: LoadConstraint, alpha_trade_price_emissions: float = 0.5) -> OptimizationResult: ...