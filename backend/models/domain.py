from __future__ import annotations

"""Pydantic models for the Worldline backend.

Defines the configuration payload (locations, states, transitions, strategies),
simulation settings, and result schemas shared between API and simulation engine.
All classes carry rich docstrings so callers understand how the backend expects
each piece of the career graph and simulation knobs to be shaped.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class Location(BaseModel):
    """Represents a geographic location with cost-of-living and tax context.

    Locations act as anchors for career states and strategies; they capture core
    financial assumptions (annual cost of living and state tax rate) that flow
    into after-tax cash calculations, and optional metadata leaves room for
    extensibility (e.g., housing market, inflation assumptions).
    """
    id: str
    name: str
    col_annual: float
    state_tax_rate: float
    metadata: Optional[Dict[str, Any]] = None


class SeverancePolicy(BaseModel):
    """Describes how severance is computed when a layoff occurs.

    The parameters allow the simulation to approximate severance payouts based on
    tenure-sensitive formulas (weeks per year of service) with floors and caps.
    """
    weeks_per_year: float
    cap_weeks: float
    min_weeks: float


class EquityGrant(BaseModel):
    """Single equity grant with value and vesting schedule.

    Grants are simplified to a dollar value that vests linearly over years after
    an optional cliff; this keeps the Monte Carlo engine lightweight while still
    modeling the cash-like flow of equity over time.
    """
    type: str
    grant_value: float
    vest_years: float
    cliff_months: int = 0


class OneTimeCashFlow(BaseModel):
    """Represents a one-off cash event (e.g., signing bonus) at a month offset.

    These payments bypass recurring income flows and are injected exactly once
    during the simulation timeline, letting scenarios capture spikes like bonuses
    or relocation stipends.
    """
    amount: float
    month_offset: int


class Compensation(BaseModel):
    """Compensation package composed of salary, bonus, equity, and one-time items.

    The structure mirrors common corporate pay practices: expected bonus with a
    probability of payout, a list of equity grants with vesting, and one-time
    cash flows for events like sign-on bonuses. Severance is optional and used
    only for scenarios that simulate layoffs.
    """
    base_annual: float
    bonus_target_annual: float
    bonus_prob_pay: float
    equity: List[EquityGrant] = Field(default_factory=list)
    one_times: List[OneTimeCashFlow] = Field(default_factory=list)
    severance_policy: Optional[SeverancePolicy] = None


class PortfolioSettings(BaseModel):
    """Portfolio assumptions for liquid assets and contributions.

    These settings seed the Monte Carlo return process: initial liquid net worth,
    expected mean and volatility of annual returns, and the contribution rate
    applied to positive monthly cash flow before investment compounding.
    """
    initial_liquid: float
    mean_annual_return: float
    std_annual_return: float
    contribution_rate: float = 0.0


class IdentityBrand(BaseModel):
    """Subjective markers for external and internal career capital.

    External portability approximates marketability outside the company, while
    internal stature tracks reputation and influence inside the org. Both feed
    non-financial scoring to balance purely monetary outcomes.
    """
    external_portability: float
    internal_stature: float


class Constraints(BaseModel):
    """Optional constraints applied to a career state when evaluating transitions.

    Paycut floors enforce minimum acceptable compensation deltas, while the
    clear-COL flag forces after-tax income to exceed cost of living in guards
    that respect it. Defaults assume conservative risk tolerance.
    """
    paycut_floor_pct: float = -0.40
    must_clear_col_after_tax: bool = True


class CareerState(BaseModel):
    """Node in the career DAG with compensation, location, wellbeing, and brand markers.

    States are the vertices of the simulation graph. They capture job metadata,
    compensation (or absence thereof), identity brand scores, wellbeing, and
    optional minimum tenure in months to prevent too-rapid transitions.
    """
    id: str
    label: str
    t_months_min: int = 0
    role_title: str
    level: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    location_id: str
    employment_status: str
    compensation: Optional[Compensation]
    wellbeing: float = 0.5
    performance_band: Optional[str] = None
    constraints: Constraints = Constraints()
    beliefs: Dict[str, Any] = Field(default_factory=dict)
    identity_brand: IdentityBrand = IdentityBrand(
        external_portability=0.5, internal_stature=0.5
    )


class TransitionGuards(BaseModel):
    """Flags controlling whether strategy constraints must be respected on a transition.

    These toggles allow certain edges to bypass paycut or COL checks when
    strategies permit it, enabling fine-grained modeling of mandatory moves or
    exceptions to normal policies.
    """
    respect_paycut_floor: bool = True
    must_clear_col_after_tax: bool = True
    requires_terms_accepted: bool = False
    custom_conditions: Dict[str, Any] = Field(default_factory=dict)


class TransitionDelta(BaseModel):
    """Captures financial and identity shocks associated with a transition.

    Deltas adjust portfolio balances (relocation costs), ongoing compensation,
    and soft identity signals, letting a single edge encapsulate both hard and
    soft consequences of a move.
    """
    relocation_cost: Optional[float] = None
    partner_income_impact_annual: Optional[float] = None
    comp_adjustment_pct: Optional[float] = None
    identity_shock: Optional[float] = None
    brand_penalty: Optional[float] = None

class Transition(BaseModel):
    """Edge between states with hazard, optional lag, deltas (relocation/comp/identity), and guards.

    Transitions represent probabilistic moves out of a state. The base annual
    probability is scaled into step probabilities during simulation, with
    optional desire multipliers and lags to model delayed transitions. Deltas
    capture monetary shocks (relocation costs, comp adjustments) and soft
    signals (identity shocks), while guards enforce strategy-specific rules.
    """
    id: str
    from_state_id: str
    to_state_id: str
    type: str
    base_annual_prob: float
    desire_multiplier: float = 1.0
    lag_months: int = 0
    delta: TransitionDelta = TransitionDelta()
    guards: TransitionGuards = TransitionGuards()

    @validator("base_annual_prob")
    def check_prob(cls, v: float) -> float:
        """Ensure hazards are non-negative to keep Poisson conversion sane.

        Negative hazards would break the exponential transform in the simulation
        engine and imply backwards probabilities, so we fail fast at validation
        time to surface schema issues early.
        """
        if v < 0:
            raise ValueError("base_annual_prob must be non-negative")
        return v


class StrategyRule(BaseModel):
    """Represents a custom rule applied by a strategy, parametrized for flexibility.

    Rules are intentionally generic (id + params) so downstream components can
    interpret them to inject domain-specific logic without changing the schema.
    """
    id: str
    description: str
    params: Dict[str, Any]


class Strategy(BaseModel):
    """Policy layer describing start states plus constraints and preferences applied to transitions.

    Strategies define how simulations choose initial states, enforce paycut
    floors, and bias moves via preferred/disallowed locations. Rules allow for
    future extension while scoring_weights_override can tweak utility weighting
    without mutating the global config.
    """
    id: str
    name: str
    description: str
    initial_choice_state_ids: List[str]
    paycut_floor_pct: Optional[float] = None
    preferred_locations: List[str] = Field(default_factory=list)
    disallowed_locations: List[str] = Field(default_factory=list)
    max_unemployment_months_tolerated: Optional[int] = None
    rules: List[StrategyRule] = Field(default_factory=list)
    scoring_weights_override: Optional[Dict[str, float]] = None


class SimulationSettings(BaseModel):
    """Knobs that shape each Monte Carlo run (horizons, cadence, risk, run counts, seed).

    These settings can be overridden per request to explore sensitivities without
    mutating the persisted config.
    """
    time_step_months: int = 1
    horizon_years_short: int = 5
    horizon_years_long: int = 10
    discount_rate_real: float = 0.02
    risk_penalty_lambda: float = 0.5
    cvar_alpha: float = 0.10
    num_runs_per_scenario: int = 5000
    random_seed: Optional[int] = None

    @validator("time_step_months")
    def check_step(cls, v: int) -> int:
        """Guard against degenerate simulation cadence (must be between 1 and 12 months).

        Extremely small or large steps would distort hazard conversions and
        portfolio math, so we enforce a sane range up front.
        """
        if v <= 0 or v > 12:
            raise ValueError("time_step_months must be between 1 and 12")
        return v


class ScoringWeights(BaseModel):
    """Weights used to blend financial EV/variance with career capital, enjoyment/identity, location fit, and legacy.

    The weights are normalized to maintain proportional influence even if users
    input values that do not sum to one.
    """
    financial: float = 0.65
    career_capital: float = 0.20
    enjoyment_identity: float = 0.10
    location_fit: float = 0.10
    legacy: float = 0.05

    @root_validator
    def normalize(cls, values: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0 to keep utility scoring comparable across configs.

        If all weights are zero we skip scaling to avoid division-by-zero; otherwise
        each weight is rescaled so users can provide arbitrary magnitudes without
        changing relative preferences.
        """
        total = sum(values.get(k, 0) for k in values)
        if total == 0:
            return values
        scale = 1.0 / total
        for k in list(values.keys()):
            values[k] = values[k] * scale
        return values


class ConfigPayload(BaseModel):
    """Root config payload combining locations, states, transitions, strategies, weights, and sim settings."""
    locations: List[Location]
    portfolio_settings: PortfolioSettings
    career_states: List[CareerState]
    transitions: List[Transition]
    strategies: List[Strategy]
    scoring_weights: ScoringWeights = ScoringWeights()
    simulation_settings: SimulationSettings = SimulationSettings()


class ScenarioResult(BaseModel):
    """Outcome metrics for a single scenario (strategy + initial state + horizon).

    Combines financial stats (EV/variance/CVaR, portfolio percentiles), downside
    probabilities, and non-financial scores so the frontend can present a full
    picture of each simulated worldline.
    """
    strategy_id: str
    initial_state_id: str
    label: str
    horizon_years: int
    ev_npv: float
    var_npv: float
    cvar_npv: float
    utility_score: float
    portfolio_stats: Dict[str, Any]
    downside: Dict[str, Any]
    nonfinancial: Dict[str, Any]


class SimulationResult(BaseModel):
    """Aggregate of all scenarios plus summary dashboards and the assumptions used.

    Includes top performers per horizon, downside rollups by strategy, sensitivity
    experiments, and the settings that produced them for reproducibility.
    """
    best_5y: Optional[ScenarioResult]
    best_10y: Optional[ScenarioResult]
    all_scenarios: List[ScenarioResult]
    downside_dashboard: Dict[str, Any]
    sensitivity: Dict[str, Any]
    assumptions: Dict[str, Any]
