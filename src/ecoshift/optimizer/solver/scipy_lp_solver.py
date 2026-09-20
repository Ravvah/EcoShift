

from typing import List, Tuple
from scipy.optimize import linprog

from ecoshift.optimizer.domain.constraints import LoadConstraint
from ecoshift.optimizer.domain.result import OptimizationResult, TimeSlotDecision, OptimizationStatus, SolverName
from ecoshift.optimizer.solver.base import BaseSolver


class ScipyLPSolver(BaseSolver):

    def __init__(self, time_limit_seconds):
        super().__init__(time_limit_seconds)

    def solve(self, price_eur_kwh_list: List[float], co2_intensity_kg_kwh_list: List[float], load_constraint: LoadConstraint, alpha_trade_price_emissions: float = 0.5): 
        vector_costs = self.generate_vector_costs(price_eur_kwh_list, co2_intensity_kg_kwh_list, alpha_trade_price_emissions)
        time_steps_number = len(price_eur_kwh_list)
        variable_bounds_list = self.generate_variable_bounds(time_steps_number, load_constraint)
        matrix_half_hour = [[0.5] * time_steps_number]
        energy_required_kwh = [load_constraint.energy_required_kwh]
        res = linprog(c=vector_costs, A_eq=matrix_half_hour, b_eq=energy_required_kwh, bounds=variable_bounds_list, method="highs")

        if res.status == 1:
            return OptimizationResult(
                status=OptimizationStatus.ITERATION_LIMIT,
                solver_name=SolverName.SCIPY_HIGHS,
                total_cost_eur=0.0,
                total_co2_emissions_kg=0.0,
                schedule=[]
            )
    
        elif res.status == 2:
            return OptimizationResult(
                status=OptimizationStatus.INFEASIBLE,
                solver_name=SolverName.SCIPY_HIGHS,
                total_cost_eur=0.0,
                total_co2_emissions_kg=0.0,
                schedule=[]
            )

        elif res.status == 3:
            return OptimizationResult(
                status=OptimizationStatus.UNBOUNDED,
                solver_name=SolverName.SCIPY_HIGHS,
                total_cost_eur=0.0,
                total_co2_emissions_kg=0.0,
                schedule=[]
            )

        elif res.status == 4:
            return OptimizationResult(
                status=OptimizationStatus.NUMERICAL_DIFFICULTY,
                solver_name=SolverName.SCIPY_HIGHS,
                total_cost_eur=0.0,
                total_co2_emissions_kg=0.0,
                schedule=[]
            )

        elif res.status == 0:
            total_cost = 0
            total_co2_emissions = 0

            power_allocated_list: List[float] = res.x.tolist()
            schedule: List[TimeSlotDecision] = []

            for t, power_at_t in enumerate(power_allocated_list):
                energy_allocated_at_t = power_at_t * 0.5
                price_at_t = price_eur_kwh_list[t]
                co2_intensity_at_t = co2_intensity_kg_kwh_list[t]
                cost_at_t = energy_allocated_at_t * price_at_t
                co2_emissions_at_t = energy_allocated_at_t * co2_intensity_at_t

                total_cost += cost_at_t
                total_co2_emissions += co2_emissions_at_t

                schedule.append(
                    TimeSlotDecision(
                        timestep_decision=t,
                        power_allocated_kw = power_at_t,
                        energy_allocated_kwh = energy_allocated_at_t,
                        price_eur_kwh=price_at_t,
                        co2_intensity_kg_kwh=co2_intensity_at_t,
                        cost_eur=cost_at_t,
                        co2_emissions_kg=co2_emissions_at_t,
                        ))

            return OptimizationResult(
                status=OptimizationStatus.SUCCESS,
                solver_name=SolverName.SCIPY_HIGHS,
                total_cost_eur=total_cost,
                total_co2_emissions_kg=total_co2_emissions,
                schedule=schedule
            )
        
        return OptimizationResult(
                status=OptimizationStatus.OTHER_ERROR,
                solver_name=SolverName.SCIPY_HIGHS,
                total_cost_eur=0.0,
                total_co2_emissions_kg=0.0,
                schedule=schedule
            )

        

    def generate_vector_costs(self, price_eur_kwh_list: List[float], co2_intensity_kg_kwh_list: List[float], alpha_trade_price_emissions: float = 0.5) -> List[float]:
        vector_costs = []
        for price_eur_kwh, co2_intensity_kg_kwh in zip(price_eur_kwh_list, co2_intensity_kg_kwh_list):
            cost = (price_eur_kwh * alpha_trade_price_emissions + (1 - alpha_trade_price_emissions) * co2_intensity_kg_kwh) * 0.5
            vector_costs.append(cost)
        return vector_costs

    def generate_variable_bounds(self, time_steps_number: int, load_constraint: LoadConstraint) -> List[Tuple[float, float]]:
        variable_bounds_list = []
        for i_price in range(time_steps_number):
            if i_price < load_constraint.start_time_step or i_price >= load_constraint.end_time_step:
                variable_bounds_list.append((0.0, 0.0))
            else:
                variable_bounds_list.append((load_constraint.power_min_kw, load_constraint.power_max_kw))

        return variable_bounds_list



    


