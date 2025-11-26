from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class Location(BaseModel):
    id: str
    name: str
    col_annual: float
    state_tax_rate: float
    metadata: Optional[Dict[str, Any]] = None


class SeverancePolicy(BaseModel):
    weeks_per_year: float
    cap_weeks: float
    min_weeks: float


class EquityGrant(BaseModel):
    type: str
    grant_value: float
    vest_years: float
    cliff_months: int = 0


class OneTimeCashFlow(BaseModel):
    amount: float
    month_offset: int


class Compensation(BaseModel):
    base_annual: float
    bonus_target_annual: float
    bonus_prob_pay: float
    equity: List[EquityGrant] = Field(default_factory=list)
    one_times: List[OneTimeCashFlow] = Field(default_factory=list)
    severance_policy: Optional[SeverancePolicy] = None


class PortfolioSettings(BaseModel):
    initial_liquid: float
    mean_annual_return: float
    std_annual_return: float
    contribution_rate: float = 0.0


class IdentityBrand(BaseModel):
    external_portability: float
    internal_stature: float


class Constraints(BaseModel):
    paycut_floor_pct: float = -0.40
    must_clear_col_after_tax: bool = True


class CareerState(BaseModel):
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
    respect_paycut_floor: bool = True
    must_clear_col_after_tax: bool = True
    requires_terms_accepted: bool = False
    custom_conditions: Dict[str, Any] = Field(default_factory=dict)


class TransitionDelta(BaseModel):
    relocation_cost: Optional[float] = None
    partner_income_impact_annual: Optional[float] = None
    comp_adjustment_pct: Optional[float] = None
    identity_shock: Optional[float] = None
    brand_penalty: Optional[float] = None


class Transition(BaseModel):
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
        if v < 0:
            raise ValueError("base_annual_prob must be non-negative")
        return v


class StrategyRule(BaseModel):
    id: str
    description: str
    params: Dict[str, Any]


class Strategy(BaseModel):
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
        if v <= 0 or v > 12:
            raise ValueError("time_step_months must be between 1 and 12")
        return v


class ScoringWeights(BaseModel):
    financial: float = 0.65
    career_capital: float = 0.20
    enjoyment_identity: float = 0.10
    location_fit: float = 0.10
    legacy: float = 0.05

    @root_validator
    def normalize(cls, values: Dict[str, float]) -> Dict[str, float]:
        total = sum(values.get(k, 0) for k in values)
        if total == 0:
            return values
        scale = 1.0 / total
        for k in list(values.keys()):
            values[k] = values[k] * scale
        return values


class ConfigPayload(BaseModel):
    locations: List[Location]
    portfolio_settings: PortfolioSettings
    career_states: List[CareerState]
    transitions: List[Transition]
    strategies: List[Strategy]
    scoring_weights: ScoringWeights = ScoringWeights()
    simulation_settings: SimulationSettings = SimulationSettings()


class ScenarioResult(BaseModel):
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
    best_5y: Optional[ScenarioResult]
    best_10y: Optional[ScenarioResult]
    all_scenarios: List[ScenarioResult]
    downside_dashboard: Dict[str, Any]
    sensitivity: Dict[str, Any]
    assumptions: Dict[str, Any]
