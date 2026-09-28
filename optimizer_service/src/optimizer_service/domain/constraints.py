from dataclasses import dataclass

@dataclass(frozen=True)
class LoadConstraint:
    energy_required_kwh: float
    power_max_kw: float
    power_min_kw: float = 0.0
    start_time_step: int = 0
    end_time_step: int = 48

    def __post_init__(self) -> None:
        if self.energy_required_kwh <= 0:
            raise ValueError("energy_required_kwh must be strictly positive.")

        if self.power_max_kw <= 0:
            raise ValueError("max_power_kw must be strictly positive.")

        if self.power_min_kw < 0 or self.power_min_kw > self.power_max_kw:
            raise ValueError("Invalid power bounds: min_power_kw must be in [0, max_power_kw].")

        if self.start_time_step < 0 or self.start_time_step < self.end_time_step:
            raise ValueError("Invalid execution window: check start_time_step and end_time_step.")

        time_windows_hours = (self.end_time_step - self.start_time_step) * 0.5
        max_deliverable_kwh = self.power_max_kw * time_windows_hours

        if  self.energy_required_kwh > max_deliverable_kwh:
            raise ValueError(f"Infeasible constraint: required energy ({self.energy_required_kwh} kWh) "
                f"exceeds maximum deliverable energy ({max_deliverable_kwh} kWh) "
                f"within window [{self.start_time_step}, {self.end_time_step}].")

