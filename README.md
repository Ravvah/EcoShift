# EcoShift – Smart Industrial Load Shifting

EcoShift is an in-progress platform for intelligent industrial task scheduling.  
Its goal is to minimize **electricity costs** and **CO2 emissions** while respecting strict operational constraints such as **deadlines** and **maximum available power**.

The system is designed as two independent services:

- **Forecaster Service**: predicts electricity prices and carbon intensity over a future horizon.
- **Optimizer Service**: computes an optimal task allocation/schedule using those forecasts and plant constraints.

These services communicate through APIs and are intended to run in containerized, orchestrated environments.

---

## Project Objective

Industrial plants must execute tasks on time, but energy prices and carbon intensity vary over time.  
EcoShift aims to provide a software tool that schedules operations intelligently by balancing:

1. **Economic objective**: reduce electricity bill.
2. **Environmental objective**: reduce carbon footprint.
3. **Operational feasibility**: satisfy deadlines and power limits.

---

## Target Architecture

### 1) Forecaster (Service A)
Inputs:
- Historical market and contextual data (prices, CO2 intensity, weather/calendar/time features, etc.)

Outputs:
- Horizon forecasts (e.g., next 24–168 hours):
  - electricity price forecast
  - CO2 intensity forecast

API role:
- Expose prediction endpoints.
- Publish forecast payloads consumable by Optimizer.

### 2) Optimizer (Service B)
Inputs:
- Forecasts from Forecaster.
- Industrial task metadata:
  - duration
  - earliest start / deadline
  - power demand
  - optional priorities/penalties
- Site constraints:
  - max power capacity
  - business/operational rules

Outputs:
- Feasible and optimized schedule/allocation plan.
- KPIs: expected cost, emissions, utilization, SLA/deadline compliance.

API role:
- Receive forecasts + constraints.
- Return optimized planning results.

---

## API Communication Contract (Concept)

Forecaster → Optimizer payload :
- `horizon_start`, `horizon_end`, `time_step`
- `series[]` where each item includes:
  - `timestamp`
  - `predicted_price_eur_mwh`
  - `predicted_co2_kg_mwh`
  - (optional) confidence intervals

Optimizer request (concept):
- Forecast payload
- Task list
- Plant/site constraints
- Objective weights (cost vs CO2)

Optimizer response (concept):
- Task-to-timeslot allocation
- Aggregate objective values
- Constraint satisfaction report

---

## Current Implementation Status

### Implemented now
- Python project scaffold with `src/` layout.
- **Forecaster service API skeleton** using **FastAPI**:
  - app bootstrapping, versioned API routing, docs path, service lifespan management.
- Predictor service warm-up triggered at startup.
- Basic logging and optional CORS middleware configuration.
- Rich ML/data dependencies already declared (LightGBM, scikit-learn, statsmodels, Optuna, MLflow, Pandera, etc.).

### Not yet visible (at current repo state)
- Public README content (currently empty in repo).
- Optimizer service implementation.
- Explicit Docker/Kubernetes manifests in tracked files.
- Finalized API contract between services.
- MILP model implementation details and solver integration.

---

## Tech Stack (Planned / Current)

- **Language**: Python
- **API framework**: FastAPI (+ Uvicorn/Gunicorn)
- **ML stack**: scikit-learn, LightGBM, statsmodels, Optuna, MLflow
- **Data stack**: pandas, numpy, pyarrow, pandera
- **Deployment**: Docker
- **Orchestration**: Kubernetes
- **Optimization (target)**: MILP solver integration (to be finalized)

---

## Proposed Repository Structure (target)

```text
src/
  ecoshift/
    forecaster/
      app/
        api/
        core/
        services/
        schemas/
    optimizer/
      app/
        api/
        core/
        models/
        solvers/
        schemas/
infra/
  docker/
  k8s/
tests/
docs/
```

---

## Local Development (initial)

```bash
# create environment and install
uv sync

# run forecaster service (example)
uv run python -m ecoshift.forecaster.app.main
```

---

## Roadmap 

- [ ] Finalize forecaster API schema (price + CO2 horizon predictions) and deploy forecaster service 
- [ ] Implement optimizer service (MILP design, optimizer api )
- [ ] Define inter-service contract and versioning strategy
- [ ] Add Kubernetes manifests (Deployments, Services, ConfigMaps, Secrets, HPA)
- [ ] Add end-to-end integration tests (Forecaster → Optimizer)
- [ ] Add observability (metrics, logs, tracing)
- [ ] Benchmark optimization quality and runtime on realistic industrial workloads

---

## Vision

EcoShift will provide factories with actionable, optimized schedules that are both cost-efficient and low-carbon—bridging predictive analytics and operations research in a production-ready microservice architecture.
