from optimizer_service.domain.result import SolverName
from optimizer_service.solver.base import BaseSolver
from optimizer_service.solver.scipy_lp_solver import ScipyLPSolver


class SolverFactory:

    @staticmethod
    def get_solver(solver_name: SolverName, time_limit_seconds: float = 10.0) -> BaseSolver:
        if solver_name == SolverName.SCIPY_HIGHS:
            return ScipyLPSolver(time_limit_seconds=time_limit_seconds)
        raise ValueError(f"Unknown solver name: {solver_name}")

