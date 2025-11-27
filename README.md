# Worldline

Worldline is a full-stack reference app for modeling a career as a directed acyclic graph, applying strategies, and running Monte Carlo simulations to evaluate 5-year and 10-year outcomes.

## What this app is for

- Model a career as a graph of states (roles, locations, employment statuses) and transitions (moves, layoffs, promotions) instead of a single straight-line plan.
- Encode preferences as strategies that emphasize or forbid certain moves, enforce paycut floors, and pick where you start.
- Run repeated simulations to understand outcome distributions: expected value, downside, risk-penalized utility, and non-financial fit.
- Compare strategies and starting points side-by-side, then export the raw data for deeper analysis.

## How to use Worldline

1. Launch the backend (`uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`) and the frontend (`npm run dev` inside `frontend/`).
2. Open http://localhost:5173. The app loads a sample config and posts it to the backend automatically.
3. Use the **Interactive Config Builder** to add/edit locations, career states, transitions, strategies, and scoring weights. Every change updates the JSON preview.
4. If you prefer editing JSON directly, adjust it in the **JSON Preview** and click **Apply** to push it back into the builder.
5. Click **Save Config** to persist to the backend or **Run Simulations** to auto-save and simulate. Results populate the simulation panels and charts.
6. Export the latest simulation payload with **Export JSON** for offline analysis or visualization.

## How the simulation works (high level)

- **Scenario expansion:** Each strategy combined with each initial state becomes a scenario. The engine runs separate Monte Carlo batches for short (5y) and long (10y) horizons.
- **Time stepping:** Simulations advance in fixed steps (monthly by default). Annual transition hazards are converted to per-step probabilities; portfolio return mean/std are scaled to the step size.
- **Cash flow + portfolio:** For the active state, the engine calculates after-tax income (base + expected bonus + vesting + one-time cash), subtracts cost of living, adds contributions (if positive cash), and applies a stochastic portfolio return. Relocation costs and compensation adjustments from transitions are applied when moves occur.
- **Transition choice:** Eligible outgoing edges are filtered by strategy rules (disallowed locations, paycut floors, min time ordering). A weighted random draw picks a move or "stay" based on per-step probabilities and optional lag months.
- **Tracking metrics per run:** The engine records unemployment duration, liquidity versus cost-of-living multiples, pay haircuts on re-entry, location time shares, wellbeing, and identity/brand scores to estimate career capital and enjoyment.
- **Aggregating outcomes:** After all runs, it computes NPV (discounted cash flows), portfolio percentiles, downside probabilities (e.g., P(liquid < 1x COL), unemployment spells), and non-financial averages. Utility = financial expected value minus a variance penalty, then blended with career capital, enjoyment/identity, location fit, and legacy weights.
- **Choosing winners and sensitivity:** The best 5y and 10y scenarios are selected by utility. A small sensitivity sweep perturbs portfolio returns/volatility and transition probabilities to show how fragile the top result is.
- **Assumptions reporting:** Returned results include the executor mode, worker count, run count, time step, risk penalty, and CVaR alpha used during the simulation.

## Project layout

- `backend/`: FastAPI + Python 3.11. Pydantic models, simulation engine, and API endpoints.
- `frontend/`: React + TypeScript (Vite). Configuration editor, DAG visualization, charts, and export.

## Quickstart

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` – sanity check.
- `POST /config` – save configuration payload.
- `GET /config` – fetch current configuration.
- `POST /simulate` – run simulations (optionally override `SimulationSettings`).
- `GET /export` – export last simulation result as structured JSON.

Run tests:

```bash
python -m pytest backend/tests
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and ensure the backend is running on http://localhost:8000 (CORS enabled by default).

## Concepts

- **Career states (nodes):** roles/contexts with comp, location, wellbeing, and identity brand.
- **Transitions (edges):** moves between states with hazards, lags, and deltas (relocation costs, pay adjustments).
- **Strategies:** policies that filter/shape transitions (paycut floors, preferred/disallowed locations, initial choices).
- **Simulation settings:** time step (monthly/quarterly), horizons (5/10y), risk aversion, CVaR level, run count, discount rate.
- **Portfolio model:** liquid wealth with stochastic returns plus net cash flow from compensation minus COL/taxes, contributions, and vesting.
- **Outcomes:** EV/variance/CVaR of NPV, downside probabilities (unemployment spells, liquidity vs COL), non-financial scores, and overall utility using configurable weights.

## Example configuration

The frontend ships a starter config and auto-posts it to the backend. You can also POST a payload like:

```json
{
  "locations": [
    { "id": "home", "name": "Home City", "col_annual": 60000, "state_tax_rate": 0.05 },
    { "id": "hub", "name": "Tech Hub", "col_annual": 90000, "state_tax_rate": 0.1 }
  ],
  "portfolio_settings": {
    "initial_liquid": 250000,
    "mean_annual_return": 0.06,
    "std_annual_return": 0.12,
    "contribution_rate": 0.1
  },
  "career_states": [
    {
      "id": "current",
      "label": "Current Role",
      "role_title": "Engineering Manager",
      "location_id": "home",
      "employment_status": "employed",
      "compensation": { "base_annual": 200000, "bonus_target_annual": 50000, "bonus_prob_pay": 0.7, "equity": [], "one_times": [] }
    },
    {
      "id": "promotion",
      "label": "Director",
      "role_title": "Director of Engineering",
      "location_id": "home",
      "employment_status": "employed",
      "compensation": { "base_annual": 260000, "bonus_target_annual": 70000, "bonus_prob_pay": 0.7, "equity": [], "one_times": [] }
    },
    {
      "id": "startup",
      "label": "Startup CTO",
      "role_title": "CTO",
      "location_id": "hub",
      "employment_status": "employed",
      "compensation": { "base_annual": 180000, "bonus_target_annual": 30000, "bonus_prob_pay": 0.4, "equity": [{ "type": "RSU", "grant_value": 200000, "vest_years": 4, "cliff_months": 12 }], "one_times": [] }
    },
    { "id": "unemployed", "label": "Unemployment", "role_title": "Unemployed", "location_id": "home", "employment_status": "unemployed", "compensation": null }
  ],
  "transitions": [
    { "id": "t1", "from_state_id": "current", "to_state_id": "promotion", "type": "promotion", "base_annual_prob": 0.25 },
    { "id": "t2", "from_state_id": "current", "to_state_id": "startup", "type": "external_switch", "base_annual_prob": 0.12, "lag_months": 1, "delta": { "relocation_cost": 15000 } },
    { "id": "t3", "from_state_id": "current", "to_state_id": "unemployed", "type": "layoff", "base_annual_prob": 0.05 },
    { "id": "t4", "from_state_id": "promotion", "to_state_id": "unemployed", "type": "layoff", "base_annual_prob": 0.04 },
    { "id": "t5", "from_state_id": "startup", "to_state_id": "unemployed", "type": "startup_failure", "base_annual_prob": 0.18 },
    { "id": "t6", "from_state_id": "unemployed", "to_state_id": "current", "type": "reentry", "base_annual_prob": 0.35 }
  ],
  "strategies": [
    { "id": "stability", "name": "Stability", "description": "Keep current role and optimize promotion odds", "initial_choice_state_ids": ["current"], "preferred_locations": ["home"], "disallowed_locations": [], "paycut_floor_pct": -0.2, "rules": [] },
    { "id": "upswing", "name": "Upswing", "description": "Bias toward startup growth moves", "initial_choice_state_ids": ["current"], "preferred_locations": ["hub"], "disallowed_locations": [], "paycut_floor_pct": -0.35, "rules": [] }
  ],
  "simulation_settings": { "time_step_months": 1, "horizon_years_short": 5, "horizon_years_long": 10, "discount_rate_real": 0.02, "risk_penalty_lambda": 0.5, "cvar_alpha": 0.1, "num_runs_per_scenario": 500, "random_seed": 7 }
}
```

## Frontend UX map

- **Configuration:** JSON editor (ready for swapping in a form-driven editor), per-section tabs can be added incrementally.
- **DAG view:** React Flow visualizes nodes/edges, color-coded by employment status.
- **Simulation:** Trigger `/simulate`, then view best 5y/10y summaries, bar charts for utility/EV/CVaR, downside dashboard, sensitivity (tornado), and scenario list. Export downloads the last `/export` payload.

## Notes and guardrails

- Graph must remain a DAG; backend validates against cycles.
- Strategies filter transitions (disallowed locations, paycut floors) and shape eligibility.
- Risk modeling: monthly returns derived from annual mean/std; NPV discounted using real rate; risk-adjusted score = EV − λ·Var with CVaR reporting.
- Downside metrics: liquidity vs COL, unemployment spell probabilities (6/12/24m), lower-pay re-entry probability, median haircut.
- Sensitivity: perturb portfolio returns/volatility and transition hazards to populate tornado data.

## Instructions recap
- Backend: `cd backend && python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
- Frontend: `cd frontend && npm install && npm run dev` (after backend is up on 8000).
- Run tests: `cd backend && python -m pytest backend/tests`

## Glossary
- DAG (Directed Acyclic Graph): career model of states (nodes) and transitions (edges) with no backward time moves.
- Strategy: policy that filters/weights transitions and defines starting choices.
- Scenario: a specific (strategy × initial_state) pair evaluated over a horizon.
- Utility: combined score using EV minus risk penalty plus weighted non-financial scores.
- EV (Expected Value): mean of simulated NPVs (net present value of cash flows/portfolio).
- CVaR (Conditional Value at Risk): average of the worst α fraction of outcomes (downside tail).
- Risk penalty λ: weight applied to variance in the financial score (higher λ = more risk-averse).
- cvar α: percentile of worst outcomes used for CVaR (e.g., 0.10 = worst 10%).
- COL: cost of living for a location (after-tax annual baseline).
- Paycut floor: minimum allowed compensation change vs current when considering transitions.
- Downside liquidity (LIO shorthand): checks like P(liquid < 1x or 2x COL) during the path.
- Tornado sensitivity: ± perturbations of key parameters to see utility swings (plotted as horizontal bars).
