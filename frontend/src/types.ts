// Shared TypeScript types mirroring backend models for configs and results.
export type Location = {
  id: string;
  name: string;
  col_annual: number;
  state_tax_rate: number;
  metadata?: Record<string, unknown>;
};

export type EquityGrant = {
  type: string;
  grant_value: number;
  vest_years: number;
  cliff_months?: number;
};

export type OneTimeCashFlow = {
  amount: number;
  month_offset: number;
};

export type SeverancePolicy = {
  weeks_per_year: number;
  cap_weeks: number;
  min_weeks: number;
};

export type Compensation = {
  base_annual: number;
  bonus_target_annual: number;
  bonus_prob_pay: number;
  equity: EquityGrant[];
  one_times: OneTimeCashFlow[];
  severance_policy?: SeverancePolicy | null;
};

export type PortfolioSettings = {
  initial_liquid: number;
  mean_annual_return: number;
  std_annual_return: number;
  contribution_rate: number;
};

export type IdentityBrand = {
  external_portability: number;
  internal_stature: number;
};

export type Constraints = {
  paycut_floor_pct: number;
  must_clear_col_after_tax: boolean;
};

export type CareerState = {
  id: string;
  label: string;
  t_months_min?: number;
  role_title: string;
  level?: string | null;
  company?: string | null;
  industry?: string | null;
  location_id: string;
  employment_status: string;
  compensation: Compensation | null;
  wellbeing: number;
  performance_band?: string | null;
  constraints?: Constraints;
  beliefs?: Record<string, unknown>;
  identity_brand: IdentityBrand;
};

export type TransitionDelta = {
  relocation_cost?: number | null;
  partner_income_impact_annual?: number | null;
  comp_adjustment_pct?: number | null;
  identity_shock?: number | null;
  brand_penalty?: number | null;
};

export type TransitionGuards = {
  respect_paycut_floor?: boolean;
  must_clear_col_after_tax?: boolean;
  requires_terms_accepted?: boolean;
  custom_conditions?: Record<string, unknown>;
};

export type Transition = {
  id: string;
  from_state_id: string;
  to_state_id: string;
  type: string;
  base_annual_prob: number;
  desire_multiplier?: number;
  lag_months?: number;
  delta?: TransitionDelta;
  guards?: TransitionGuards;
};

export type StrategyRule = {
  id: string;
  description: string;
  params: Record<string, unknown>;
};

export type Strategy = {
  id: string;
  name: string;
  description: string;
  initial_choice_state_ids: string[];
  paycut_floor_pct?: number | null;
  preferred_locations: string[];
  disallowed_locations: string[];
  max_unemployment_months_tolerated?: number | null;
  rules: StrategyRule[];
  scoring_weights_override?: Record<string, number> | null;
};

export type ScoringWeights = {
  financial: number;
  career_capital: number;
  enjoyment_identity: number;
  location_fit: number;
  legacy: number;
};

export type SimulationSettings = {
  time_step_months: number;
  horizon_years_short: number;
  horizon_years_long: number;
  discount_rate_real: number;
  risk_penalty_lambda: number;
  cvar_alpha: number;
  num_runs_per_scenario: number;
  random_seed?: number | null;
};

export type ConfigPayload = {
  locations: Location[];
  portfolio_settings: PortfolioSettings;
  career_states: CareerState[];
  transitions: Transition[];
  strategies: Strategy[];
  scoring_weights: ScoringWeights;
  simulation_settings: SimulationSettings;
};
