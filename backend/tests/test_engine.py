import math
import numpy as np
import pytest

from backend.models.domain import (
    CareerState,
    Compensation,
    ConfigPayload,
    EquityGrant,
    IdentityBrand,
    Location,
    OneTimeCashFlow,
    PortfolioSettings,
    ScenarioResult,
    ScoringWeights,
    SimulationSettings,
    Strategy,
    TransitionDelta,
    Transition,
)
from backend.simulation.engine import (
    _annual_to_step_params,
    _best_for_horizon,
    _aggregate_downside,
    _build_result,
    _calc_monthly_cash_flow,
    _downside_metrics,
    _npv,
    _normalize_dict,
    _portfolio_stats,
    _select_transition,
    _simulate_scenario,
    _run_sensitivity,
    _validate_dag,
    _utility,
    hazard_to_step,
    simulate_config,
)


class DummyRNG:
    """Minimal RNG stub to make simulations deterministic in tests."""

    def __init__(self, choice_index: int = 0):
        self.choice_index = choice_index

    def normal(self, mean: float, std: float) -> float:
        return mean

    def choice(self, n: int, p: np.ndarray):  # type: ignore[override]
        return min(self.choice_index, n - 1)


def test_hazard_conversion():
    monthly = hazard_to_step(0.24, 1)
    assert 0 < monthly < 1
    quarterly = hazard_to_step(0.24, 3)
    assert quarterly > monthly


@pytest.mark.parametrize(
    "annual,step_months,expected",
    [
        (-0.1, 1, 0.0),
        (0.0, 1, 0.0),
        (1e-9, 1, 0.0),
        (0.999999, 1, 1.0),
    ],
)
def test_hazard_edge_cases(annual, step_months, expected):
    val = hazard_to_step(annual, step_months)
    if expected == 0.0:
        assert 0 <= val < 1e-6
    else:
        assert 0 < val < 1


def test_simple_simulation_runs():
    locations = [Location(id="home", name="Home", col_annual=60000, state_tax_rate=0.05)]
    comp = Compensation(base_annual=120000, bonus_target_annual=20000, bonus_prob_pay=0.5)
    states = [
        CareerState(
            id="start",
            label="Start",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
        ),
        CareerState(
            id="promo",
            label="Promotion",
            role_title="Sr Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
        ),
    ]
    transitions = [
        Transition(
            id="t1",
            from_state_id="start",
            to_state_id="promo",
            type="promotion",
            base_annual_prob=0.20,
        )
    ]
    strategies = [
        Strategy(
            id="s1",
            name="Default",
            description="",
            initial_choice_state_ids=["start"],
        )
    ]
    cfg = ConfigPayload(
        locations=locations,
        portfolio_settings=PortfolioSettings(
            initial_liquid=50000, mean_annual_return=0.05, std_annual_return=0.10
        ),
        career_states=states,
        transitions=transitions,
        strategies=strategies,
        scoring_weights=ScoringWeights(),
        simulation_settings=SimulationSettings(num_runs_per_scenario=50, random_seed=123),
    )
    res = simulate_config(cfg)
    assert res.best_5y is not None
    assert len(res.all_scenarios) == 2  # 5y + 10y


def _base_components():
    loc = Location(id="home", name="Home", col_annual=60000, state_tax_rate=0.05)
    comp = Compensation(base_annual=120000, bonus_target_annual=20000, bonus_prob_pay=0.5)
    portfolio = PortfolioSettings(initial_liquid=100000, mean_annual_return=0.05, std_annual_return=0.10)
    settings = SimulationSettings(num_runs_per_scenario=80, random_seed=7, horizon_years_short=5, horizon_years_long=10)
    return loc, comp, portfolio, settings


def test_simulation_deterministic_with_seed():
    loc, comp, portfolio, settings = _base_components()
    state = CareerState(
        id="s1",
        label="Stay",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=comp,
    )
    strategy = Strategy(id="strat", name="Stay", description="", initial_choice_state_ids=[state.id])
    cfg = ConfigPayload(
        locations=[loc],
        portfolio_settings=portfolio,
        career_states=[state],
        transitions=[],
        strategies=[strategy],
        scoring_weights=ScoringWeights(),
        simulation_settings=settings,
    )

    res1 = simulate_config(cfg)
    res2 = simulate_config(cfg)
    util1 = [round(s.utility_score, 6) for s in res1.all_scenarios]
    util2 = [round(s.utility_score, 6) for s in res2.all_scenarios]
    assert util1 == util2


def test_best_selection_prefers_higher_location_fit():
    loc, comp, portfolio, settings = _base_components()
    states = [
        CareerState(
            id="s1",
            label="Home",
            role_title="Engineer",
            location_id=loc.id,
            employment_status="employed",
            compensation=comp,
        )
    ]
    strategies = [
        Strategy(id="aligned", name="Aligned", description="", initial_choice_state_ids=["s1"], preferred_locations=[loc.id]),
        Strategy(id="neutral", name="Neutral", description="", initial_choice_state_ids=["s1"], preferred_locations=[]),
    ]
    cfg = ConfigPayload(
        locations=[loc],
        portfolio_settings=portfolio,
        career_states=states,
        transitions=[],
        strategies=strategies,
        scoring_weights=ScoringWeights(location_fit=0.2, financial=0.6, career_capital=0.1, enjoyment_identity=0.05, legacy=0.05),
        simulation_settings=settings,
    )
    res = simulate_config(cfg)
    assert res.best_5y is not None
    assert res.best_5y.strategy_id == "aligned"


def test_validate_dag_blocks_time_reversal():
    loc, comp, portfolio, settings = _base_components()
    early = CareerState(
        id="early",
        label="Early",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=comp,
        t_months_min=0,
    )
    late = CareerState(
        id="late",
        label="Late",
        role_title="Engineer II",
        location_id=loc.id,
        employment_status="employed",
        compensation=comp,
        t_months_min=12,
    )
    transitions = [
        Transition(id="t_bad", from_state_id="late", to_state_id="early", type="backwards", base_annual_prob=0.2)
    ]
    cfg = ConfigPayload(
        locations=[loc],
        portfolio_settings=portfolio,
        career_states=[early, late],
        transitions=transitions,
        strategies=[Strategy(id="s", name="S", description="", initial_choice_state_ids=["late"])],
        scoring_weights=ScoringWeights(),
        simulation_settings=settings,
    )
    with pytest.raises(ValueError):
        simulate_config(cfg)


def test_sensitivity_and_downside_shapes():
    loc, comp, portfolio, settings = _base_components()
    states = [
        CareerState(
            id="s1",
            label="Home",
            role_title="Engineer",
            location_id=loc.id,
            employment_status="employed",
            compensation=comp,
        )
    ]
    cfg = ConfigPayload(
        locations=[loc],
        portfolio_settings=portfolio,
        career_states=states,
        transitions=[],
        strategies=[Strategy(id="s", name="S", description="", initial_choice_state_ids=["s1"])],
        scoring_weights=ScoringWeights(),
        simulation_settings=settings,
    )
    res = simulate_config(cfg)
    params = res.sensitivity.get("parameters", [])
    assert len(params) == 3
    assert all("base_utility" in p and "low_utility" in p and "high_utility" in p for p in params)
    downside = res.downside_dashboard.get("by_strategy", {})
    assert "s" in downside
    metrics = downside["s"]
    for key in ["p_liquid_lt_1x_col", "p_liquid_lt_2x_col", "p_unemp_ge_6m", "p_unemp_ge_12m", "p_lower_pay_reentry"]:
        assert key in metrics


def test_best_for_horizon_picks_max_utility():
    dummy = [
        type("R", (), {"horizon_years": 5, "utility_score": 10}),
        type("R", (), {"horizon_years": 5, "utility_score": 20}),
        type("R", (), {"horizon_years": 10, "utility_score": 30}),
    ]
    best_5 = _best_for_horizon(dummy, 5)
    assert best_5.utility_score == 20
    best_10 = _best_for_horizon(dummy, 10)
    assert best_10.utility_score == 30


def test_utility_penalizes_variance():
    npv = np.array([100.0, 300.0])
    downside = {}
    nonfin = {"career_capital_avg": 0, "enjoyment_avg": 0, "location_fit_avg": 0}
    weights = ScoringWeights(financial=1.0, career_capital=0, enjoyment_identity=0, location_fit=0, legacy=0)
    low_risk = _utility(npv, weights, downside, nonfin, risk_lambda=0.0)
    high_risk = _utility(npv, weights, downside, nonfin, risk_lambda=1.0)
    assert low_risk > high_risk


def test_portfolio_stats_monotonic():
    stats = _portfolio_stats(np.array([1, 2, 3, 4]))
    assert stats["p10"] <= stats["p50"] <= stats["p90"]
    assert stats["mean"] == pytest.approx(2.5)


def test_downside_metrics_probabilities():
    downside = _downside_metrics(
        min_ratio=np.array([0.5, 1.5]),
        unemployment_6=np.array([True, False]),
        unemployment_12=np.array([False, False]),
        unemployment_24=np.array([False, False]),
        lower_pay_reentry=np.array([True, True]),
        haircut_values=[-0.1, -0.3],
    )
    assert downside["p_liquid_lt_1x_col"] == 0.5
    assert downside["p_liquid_lt_2x_col"] == 1.0
    assert downside["p_unemp_ge_6m"] == 0.5
    assert downside["median_pay_haircut"] == -0.2


def test_select_transition_respects_disallowed_location():
    loc = Location(id="home", name="Home", col_annual=60000, state_tax_rate=0.05)
    comp = Compensation(base_annual=100000, bonus_target_annual=0, bonus_prob_pay=0.0)
    states = {
        "from": CareerState(
            id="from",
            label="From",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
        ),
        "to_good": CareerState(
            id="to_good",
            label="Good",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
        ),
        "to_bad": CareerState(
            id="to_bad",
            label="Bad",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
        ),
    }
    transitions = {
        "from": [
            Transition(id="t_good", from_state_id="from", to_state_id="to_good", type="move", base_annual_prob=0.999999),
            Transition(id="t_bad", from_state_id="from", to_state_id="to_bad", type="move", base_annual_prob=0.999999),
        ]
    }
    strat = Strategy(
        id="s",
        name="S",
        description="",
        initial_choice_state_ids=["from"],
        disallowed_locations=[states["to_bad"].location_id],
    )
    rng = np.random.default_rng(0)
    settings = SimulationSettings(time_step_months=1)
    seen = set()
    for _ in range(200):
        chosen, _ = _select_transition("from", strat, states, transitions, settings, rng)
        if chosen:
            seen.add(chosen.id)
    assert "t_bad" not in seen


def test_select_transition_allows_stay_when_probs_low():
    comp = Compensation(base_annual=100000, bonus_target_annual=0, bonus_prob_pay=0.0)
    states = {
        "from": CareerState(
            id="from",
            label="From",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
        ),
        "to": CareerState(
            id="to",
            label="To",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
        ),
    }
    transitions = {"from": [Transition(id="t1", from_state_id="from", to_state_id="to", type="move", base_annual_prob=1e-5)]}
    strat = Strategy(id="s", name="S", description="", initial_choice_state_ids=["from"])
    rng = np.random.default_rng(1)
    settings = SimulationSettings(time_step_months=1)
    stay_count = 0
    move_count = 0
    for _ in range(500):
        chosen, _ = _select_transition("from", strat, states, transitions, settings, rng)
        if chosen is None:
            stay_count += 1
        else:
            move_count += 1
    assert stay_count == 500
    assert move_count == 0


def test_hazard_monotonic_in_step():
    annual = 0.3
    monthly = hazard_to_step(annual, 1)
    quarterly = hazard_to_step(annual, 3)
    annualized = hazard_to_step(annual, 12)
    assert monthly < quarterly < annualized


def test_aggregate_downside_averages():
    res = [
        ScenarioResult(
            strategy_id="s",
            initial_state_id="a",
            label="",
            horizon_years=5,
            ev_npv=0,
            var_npv=0,
            cvar_npv=0,
            utility_score=0,
            portfolio_stats={},
            downside={"p_liquid_lt_1x_col": 0.2},
            nonfinancial={},
        ),
        ScenarioResult(
            strategy_id="s",
            initial_state_id="b",
            label="",
            horizon_years=5,
            ev_npv=0,
            var_npv=0,
            cvar_npv=0,
            utility_score=0,
            portfolio_stats={},
            downside={"p_liquid_lt_1x_col": 0.4},
            nonfinancial={},
        ),
    ]
    agg = _aggregate_downside(res)
    assert agg["by_strategy"]["s"]["p_liquid_lt_1x_col"] == pytest.approx(0.3)


def test_normalize_dict_handles_zero_total():
    assert _normalize_dict({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    normalized = _normalize_dict({"a": 1.0, "b": 1.0})
    assert normalized["a"] == pytest.approx(0.5)
    assert normalized["b"] == pytest.approx(0.5)


def test_scenario_count_matches_strategies_and_initial_states():
    loc, comp, portfolio, settings = _base_components()
    states = [
        CareerState(id="a", label="A", role_title="Engineer", location_id=loc.id, employment_status="employed", compensation=comp),
        CareerState(id="b", label="B", role_title="Engineer", location_id=loc.id, employment_status="employed", compensation=comp),
    ]
    strategies = [
        Strategy(id="s1", name="S1", description="", initial_choice_state_ids=["a"]),
        Strategy(id="s2", name="S2", description="", initial_choice_state_ids=["b"]),
    ]
    cfg = ConfigPayload(
        locations=[loc],
        portfolio_settings=portfolio,
        career_states=states,
        transitions=[],
        strategies=strategies,
        scoring_weights=ScoringWeights(),
        simulation_settings=settings,
    )
    res = simulate_config(cfg)
    # each strategy * initial state pair yields 5y + 10y scenario
    assert len(res.all_scenarios) == 4


def test_sensitivity_returns_empty_when_no_results():
    cfg = ConfigPayload(
        locations=[],
        portfolio_settings=PortfolioSettings(initial_liquid=0, mean_annual_return=0, std_annual_return=0),
        career_states=[],
        transitions=[],
        strategies=[],
        scoring_weights=ScoringWeights(),
        simulation_settings=SimulationSettings(),
    )
    res = simulate_config(cfg, override_settings=SimulationSettings(num_runs_per_scenario=1))
    assert res.sensitivity == {}


def test_downside_metrics_empty_haircuts():
    downside = _downside_metrics(
        min_ratio=np.array([2.0, 3.0]),
        unemployment_6=np.array([False, False]),
        unemployment_12=np.array([False, False]),
        unemployment_24=np.array([False, False]),
        lower_pay_reentry=np.array([False, False]),
        haircut_values=[],
    )
    assert downside["median_pay_haircut"] == 0.0


def test_select_transition_respects_paycut_floor():
    comp_from = Compensation(base_annual=100000, bonus_target_annual=0, bonus_prob_pay=0.0)
    comp_to = Compensation(base_annual=70000, bonus_target_annual=0, bonus_prob_pay=0.0)
    states = {
        "from": CareerState(
            id="from",
            label="From",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp_from,
        ),
        "to": CareerState(
            id="to",
            label="To",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp_to,
        ),
    }
    transitions = {"from": [Transition(id="t1", from_state_id="from", to_state_id="to", type="move", base_annual_prob=1.0)]}
    strat = Strategy(
        id="s",
        name="S",
        description="",
        initial_choice_state_ids=["from"],
        paycut_floor_pct=-0.2,
    )
    rng = np.random.default_rng(0)
    chosen, _ = _select_transition("from", strat, states, transitions, SimulationSettings(), rng)
    assert chosen is None


def test_annual_to_step_params_recompose():
    step_mean, step_std = _annual_to_step_params(0.12, 0.24, 3)
    steps_per_year = 4
    assert step_mean * steps_per_year == pytest.approx(0.12)
    assert step_std * math.sqrt(steps_per_year) == pytest.approx(0.24)


def test_npv_respects_horizon_and_discount():
    cash_flows = [100.0] * 5  # last element should be beyond horizon
    discount = 0.12
    npv = _npv(cash_flows, discount_rate=discount, step_months=6, horizon_months=18)
    expected = sum(cf / ((1 + discount) ** ((i * 6) / 12.0)) for i, cf in enumerate(cash_flows) if i < 4)
    assert npv == pytest.approx(expected)


def test_calc_monthly_cash_flow_components_and_unemployment():
    comp = Compensation(
        base_annual=120000,
        bonus_target_annual=24000,
        bonus_prob_pay=0.5,
        equity=[EquityGrant(type="rsu", grant_value=12000, vest_years=2, cliff_months=6)],
        one_times=[OneTimeCashFlow(amount=5000, month_offset=2)],
    )
    loc = Location(id="loc", name="Home", col_annual=24000, state_tax_rate=0.1)
    state = CareerState(
        id="s",
        label="Employed",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=comp,
    )
    cf_month6 = _calc_monthly_cash_flow(state, loc, month_index=6, comp_adjust_factor=1.1)
    assert cf_month6 == pytest.approx(9385.0)
    cf_month2 = _calc_monthly_cash_flow(state, loc, month_index=2, comp_adjust_factor=1.0)
    assert cf_month2 == pytest.approx(12400.0)
    unemployed = CareerState(
        id="u",
        label="Unemployed",
        role_title="",
        location_id=loc.id,
        employment_status="unemployed",
        compensation=None,
    )
    assert _calc_monthly_cash_flow(unemployed, loc, month_index=0, comp_adjust_factor=1.0) == -loc.col_annual / 12.0


def test_select_transition_ignores_invalid_and_returns_lag():
    comp = Compensation(base_annual=50000, bonus_target_annual=0, bonus_prob_pay=0)
    states = {
        "from": CareerState(
            id="from",
            label="From",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
            t_months_min=2,
        ),
        "to": CareerState(
            id="to",
            label="To",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
            t_months_min=2,
        ),
        "early": CareerState(
            id="early",
            label="Early",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp,
            t_months_min=0,
        ),
    }
    transitions = {
        "from": [
            Transition(id="t_valid", from_state_id="from", to_state_id="to", type="move", base_annual_prob=0.8, lag_months=2),
            Transition(id="t_zero", from_state_id="from", to_state_id="to", type="move", base_annual_prob=0.0),
            Transition(id="t_missing", from_state_id="from", to_state_id="missing", type="move", base_annual_prob=0.8),
            Transition(id="t_time_reversal", from_state_id="from", to_state_id="early", type="move", base_annual_prob=0.8),
        ]
    }
    strat = Strategy(id="s", name="S", description="", initial_choice_state_ids=["from"])
    rng = np.random.default_rng(3)
    seen = []
    for _ in range(200):
        chosen, lag = _select_transition("from", strat, states, transitions, SimulationSettings(), rng)
        if chosen:
            assert chosen.id == "t_valid"
            assert lag == 2
            seen.append(chosen.id)
    assert seen  # ensure the valid edge was reachable


def test_select_transition_allows_paycut_when_floor_none():
    comp_from = Compensation(base_annual=100000, bonus_target_annual=0, bonus_prob_pay=0.0)
    comp_to = Compensation(base_annual=70000, bonus_target_annual=0, bonus_prob_pay=0.0)
    states = {
        "from": CareerState(
            id="from",
            label="From",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp_from,
        ),
        "to": CareerState(
            id="to",
            label="To",
            role_title="Engineer",
            location_id="home",
            employment_status="employed",
            compensation=comp_to,
        ),
    }
    transitions = {"from": [Transition(id="t1", from_state_id="from", to_state_id="to", type="move", base_annual_prob=1.0)]}
    strat = Strategy(
        id="s",
        name="S",
        description="",
        initial_choice_state_ids=["from"],
        paycut_floor_pct=None,
    )
    chosen, _ = _select_transition("from", strat, states, transitions, SimulationSettings(), DummyRNG())
    assert chosen is not None and chosen.id == "t1"


def test_validate_dag_allows_equal_time_and_no_mutation():
    comp = Compensation(base_annual=50000, bonus_target_annual=0, bonus_prob_pay=0)
    state_a = CareerState(
        id="a",
        label="A",
        role_title="Engineer",
        location_id="home",
        employment_status="employed",
        compensation=comp,
        t_months_min=6,
    )
    state_b = CareerState(
        id="b",
        label="B",
        role_title="Engineer",
        location_id="home",
        employment_status="employed",
        compensation=comp,
        t_months_min=6,
    )
    transition = Transition(id="t", from_state_id="a", to_state_id="b", type="move", base_annual_prob=0.1)
    states = {"a": state_a, "b": state_b}
    transitions_by_from = {"a": [transition]}
    before = (state_a.t_months_min, state_b.t_months_min)
    _validate_dag(states, transitions_by_from, SimulationSettings())
    after = (state_a.t_months_min, state_b.t_months_min)
    assert before == after


def test_simulate_scenario_contributions_positive_and_negative_cash():
    positive_loc = Location(id="pos", name="Pos", col_annual=0, state_tax_rate=0.0)
    positive_state = CareerState(
        id="stay",
        label="Stay",
        role_title="Engineer",
        location_id=positive_loc.id,
        employment_status="employed",
        compensation=Compensation(base_annual=12000, bonus_target_annual=0, bonus_prob_pay=0),
    )
    strategy = Strategy(id="s", name="S", description="", initial_choice_state_ids=[positive_state.id])
    settings = SimulationSettings(
        time_step_months=1,
        horizon_years_short=1,
        horizon_years_long=2,
        discount_rate_real=0.0,
        risk_penalty_lambda=0.0,
        num_runs_per_scenario=1,
    )
    portfolio = PortfolioSettings(initial_liquid=0, mean_annual_return=0, std_annual_return=0, contribution_rate=0.5)
    res_short, res_long = _simulate_scenario(
        strategy=strategy,
        initial_state_id=positive_state.id,
        states={positive_state.id: positive_state},
        locations={positive_loc.id: positive_loc},
        transitions_by_from={},
        portfolio_settings=portfolio,
        scoring_weights=ScoringWeights(),
        settings=settings,
        rng=DummyRNG(),
    )
    assert res_short.portfolio_stats["p50"] == pytest.approx(13000.0)
    assert res_long.portfolio_stats["p50"] == pytest.approx(25000.0)

    negative_loc = Location(id="neg", name="Neg", col_annual=36000, state_tax_rate=0.0)
    negative_state = CareerState(
        id="stay_neg",
        label="StayNeg",
        role_title="Engineer",
        location_id=negative_loc.id,
        employment_status="employed",
        compensation=Compensation(base_annual=12000, bonus_target_annual=0, bonus_prob_pay=0),
    )
    portfolio_neg = PortfolioSettings(initial_liquid=0, mean_annual_return=0, std_annual_return=0, contribution_rate=0.5)
    res_short_neg, res_long_neg = _simulate_scenario(
        strategy=strategy.copy(update={"initial_choice_state_ids": [negative_state.id]}),
        initial_state_id=negative_state.id,
        states={negative_state.id: negative_state},
        locations={negative_loc.id: negative_loc},
        transitions_by_from={},
        portfolio_settings=portfolio_neg,
        scoring_weights=ScoringWeights(),
        settings=settings,
        rng=DummyRNG(),
    )
    assert res_long_neg.portfolio_stats["p50"] == pytest.approx(0.0)


def test_simulate_scenario_transition_adjustments_and_relocation_cost_with_lag():
    loc = Location(id="loc", name="Loc", col_annual=0, state_tax_rate=0.0)
    start = CareerState(
        id="start",
        label="Start",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=Compensation(base_annual=12000, bonus_target_annual=0, bonus_prob_pay=0),
    )
    after = CareerState(
        id="after",
        label="After",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=start.compensation,
    )
    transition = Transition(
        id="t",
        from_state_id=start.id,
        to_state_id=after.id,
        type="move",
        base_annual_prob=1.0,
        lag_months=6,
        delta=TransitionDelta(relocation_cost=1000, comp_adjustment_pct=0.5),
    )
    settings = SimulationSettings(
        time_step_months=6,
        horizon_years_short=1,
        horizon_years_long=2,
        discount_rate_real=0.0,
        risk_penalty_lambda=0.0,
        num_runs_per_scenario=1,
    )
    portfolio = PortfolioSettings(initial_liquid=0, mean_annual_return=0, std_annual_return=0, contribution_rate=0.0)
    res_short, res_long = _simulate_scenario(
        strategy=Strategy(id="s", name="S", description="", initial_choice_state_ids=[start.id]),
        initial_state_id=start.id,
        states={start.id: start, after.id: after},
        locations={loc.id: loc},
        transitions_by_from={start.id: [transition]},
        portfolio_settings=portfolio,
        scoring_weights=ScoringWeights(),
        settings=settings,
        rng=DummyRNG(),
    )
    assert res_short.portfolio_stats["p50"] == pytest.approx(1500.0)
    assert res_long.portfolio_stats["p50"] == pytest.approx(3500.0)


def test_unemployment_and_haircut_tracking_in_simulation():
    loc = Location(id="loc", name="Loc", col_annual=12000, state_tax_rate=0.0)
    high = CareerState(
        id="high",
        label="High",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=Compensation(base_annual=100000, bonus_target_annual=0, bonus_prob_pay=0),
    )
    unemp = CareerState(
        id="unemp",
        label="Unemp",
        role_title="",
        location_id=loc.id,
        employment_status="unemployed",
        compensation=None,
    )
    low = CareerState(
        id="low",
        label="Low",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=Compensation(base_annual=80000, bonus_target_annual=0, bonus_prob_pay=0),
    )
    transitions = {
        high.id: [Transition(id="to_unemp", from_state_id=high.id, to_state_id=unemp.id, type="layoff", base_annual_prob=1.0)],
        unemp.id: [Transition(id="to_low", from_state_id=unemp.id, to_state_id=low.id, type="rehire", base_annual_prob=1.0)],
    }
    settings = SimulationSettings(
        time_step_months=12,
        horizon_years_short=2,
        horizon_years_long=3,
        discount_rate_real=0.0,
        risk_penalty_lambda=0.0,
        num_runs_per_scenario=1,
    )
    res_short, _ = _simulate_scenario(
        strategy=Strategy(id="s", name="S", description="", initial_choice_state_ids=[high.id]),
        initial_state_id=high.id,
        states={high.id: high, unemp.id: unemp, low.id: low},
        locations={loc.id: loc},
        transitions_by_from=transitions,
        portfolio_settings=PortfolioSettings(initial_liquid=10000, mean_annual_return=0, std_annual_return=0, contribution_rate=0.0),
        scoring_weights=ScoringWeights(),
        settings=settings,
        rng=DummyRNG(),
    )
    downside = res_short.downside
    assert downside["p_unemp_ge_12m"] == pytest.approx(1.0)
    assert downside["p_lower_pay_reentry"] == pytest.approx(1.0)
    assert downside["median_pay_haircut"] == pytest.approx(-0.2)


def test_unemployment_draws_cost_of_living_from_portfolio():
    loc = Location(id="loc", name="Loc", col_annual=12000, state_tax_rate=0.0)
    unemployed = CareerState(
        id="u",
        label="Unemployed",
        role_title="",
        location_id=loc.id,
        employment_status="unemployed",
        compensation=None,
    )
    settings = SimulationSettings(
        time_step_months=1,
        horizon_years_short=0,  # capture after first month
        horizon_years_long=1,
        discount_rate_real=0.0,
        risk_penalty_lambda=0.0,
        num_runs_per_scenario=1,
    )
    portfolio = PortfolioSettings(initial_liquid=5000, mean_annual_return=0.0, std_annual_return=0.0, contribution_rate=0.0)
    res_short, res_long = _simulate_scenario(
        strategy=Strategy(id="s", name="S", description="", initial_choice_state_ids=[unemployed.id]),
        initial_state_id=unemployed.id,
        states={unemployed.id: unemployed},
        locations={loc.id: loc},
        transitions_by_from={},
        portfolio_settings=portfolio,
        scoring_weights=ScoringWeights(),
        settings=settings,
        rng=DummyRNG(),
    )
    # After one month, portfolio should drop by monthly COL (12000/12 = 1000)
    assert res_short.portfolio_stats["p50"] == pytest.approx(4000.0)
    # Long horizon should drain to zero after several months of unemployment
    assert res_long.portfolio_stats["p50"] == pytest.approx(0.0)


def test_nonfinancial_location_fit_and_time_shares():
    loc_a = Location(id="la", name="A", col_annual=0, state_tax_rate=0.0)
    loc_b = Location(id="lb", name="B", col_annual=0, state_tax_rate=0.0)
    state_a = CareerState(
        id="a",
        label="A",
        role_title="Engineer",
        location_id=loc_a.id,
        employment_status="employed",
        compensation=None,
        wellbeing=0.2,
        identity_brand=IdentityBrand(external_portability=0.2, internal_stature=0.4),
    )
    state_b = CareerState(
        id="b",
        label="B",
        role_title="Engineer",
        location_id=loc_b.id,
        employment_status="employed",
        compensation=None,
        wellbeing=0.8,
        identity_brand=IdentityBrand(external_portability=0.6, internal_stature=0.8),
    )
    transitions = {
        state_a.id: [Transition(id="a_to_b", from_state_id=state_a.id, to_state_id=state_b.id, type="move", base_annual_prob=1.0)],
        state_b.id: [Transition(id="b_to_a", from_state_id=state_b.id, to_state_id=state_a.id, type="move", base_annual_prob=1.0)],
    }
    settings = SimulationSettings(
        time_step_months=3,
        horizon_years_short=1,
        horizon_years_long=1,
        discount_rate_real=0.0,
        risk_penalty_lambda=0.0,
        num_runs_per_scenario=1,
    )
    strategy = Strategy(id="s", name="S", description="", initial_choice_state_ids=[state_a.id], preferred_locations=[loc_a.id])
    res_short, _ = _simulate_scenario(
        strategy=strategy,
        initial_state_id=state_a.id,
        states={state_a.id: state_a, state_b.id: state_b},
        locations={loc_a.id: loc_a, loc_b.id: loc_b},
        transitions_by_from=transitions,
        portfolio_settings=PortfolioSettings(initial_liquid=0, mean_annual_return=0, std_annual_return=0, contribution_rate=0.0),
        scoring_weights=ScoringWeights(),
        settings=settings,
        rng=DummyRNG(),
    )
    nonfin = res_short.nonfinancial
    assert nonfin["location_fit_avg"] == pytest.approx(0.6)
    assert nonfin["location_time_shares"]["la"] == pytest.approx(0.6)
    assert nonfin["location_time_shares"]["lb"] == pytest.approx(0.4)
    assert nonfin["career_capital_avg"] == pytest.approx(0.46)
    assert nonfin["enjoyment_avg"] == pytest.approx(0.44)


def test_portfolio_stats_single_value():
    stats = _portfolio_stats(np.array([42.0]))
    assert stats["p10"] == pytest.approx(42.0)
    assert stats["p50"] == pytest.approx(42.0)
    assert stats["p90"] == pytest.approx(42.0)
    assert stats["mean"] == pytest.approx(42.0)


def test_build_result_cvar_and_weight_normalization():
    strategy = Strategy(id="s", name="S", description="", initial_choice_state_ids=["state"])
    state = CareerState(
        id="state",
        label="State",
        role_title="Engineer",
        location_id="loc",
        employment_status="employed",
        compensation=None,
    )
    npv = np.array([-100.0, 0.0, 200.0])
    res = _build_result(
        strategy=strategy,
        initial_state_id=state.id,
        states={state.id: state},
        npv_arr=npv,
        portfolio_stats={"p50": 0, "p10": 0, "p90": 0, "mean": 0},
        downside={},
        nonfinancial={"career_capital_avg": 0, "enjoyment_avg": 0, "location_fit_avg": 0},
        scoring_weights=ScoringWeights(financial=1.0, career_capital=0.0, enjoyment_identity=0.0, location_fit=0.0, legacy=0.0),
        horizon_years=5,
        risk_lambda=0.0,
        cvar_alpha=0.5,
    )
    assert res.cvar_npv == pytest.approx(-100.0)
    assert res.utility_score == pytest.approx(np.mean(npv))


def test_run_sensitivity_caps_runs_and_preserves_config(monkeypatch):
    loc = Location(id="loc", name="Loc", col_annual=0, state_tax_rate=0.0)
    state = CareerState(
        id="state",
        label="State",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=None,
    )
    strategy = Strategy(id="s", name="S", description="", initial_choice_state_ids=[state.id])
    config = ConfigPayload(
        locations=[loc],
        portfolio_settings=PortfolioSettings(initial_liquid=0, mean_annual_return=0.05, std_annual_return=0.1),
        career_states=[state],
        transitions=[],
        strategies=[strategy],
        scoring_weights=ScoringWeights(),
        simulation_settings=SimulationSettings(num_runs_per_scenario=10),
    )
    base_result = ScenarioResult(
        strategy_id=strategy.id,
        initial_state_id=state.id,
        label="",
        horizon_years=1,
        ev_npv=0.0,
        var_npv=0.0,
        cvar_npv=0.0,
        utility_score=1.0,
        portfolio_stats={},
        downside={},
        nonfinancial={},
    )
    calls = []

    def fake_simulate_scenario(*args, **kwargs):
        settings_arg = kwargs.get("settings") or (args[7] if len(args) > 7 else None)
        calls.append(settings_arg.num_runs_per_scenario)
        return base_result, base_result

    monkeypatch.setattr("backend.simulation.engine._simulate_scenario", fake_simulate_scenario)
    settings = SimulationSettings(time_step_months=12, horizon_years_short=1, horizon_years_long=1, num_runs_per_scenario=600)
    sensitivity = _run_sensitivity(config, [base_result], settings)
    assert calls and all(call == 500 for call in calls)
    assert config.portfolio_settings.mean_annual_return == 0.05
    assert sensitivity["parameters"]


def test_simulate_config_override_settings_applied():
    loc = Location(id="loc", name="Loc", col_annual=60000, state_tax_rate=0.05)
    state = CareerState(
        id="state",
        label="State",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=Compensation(base_annual=100000, bonus_target_annual=0, bonus_prob_pay=0.0),
    )
    cfg = ConfigPayload(
        locations=[loc],
        portfolio_settings=PortfolioSettings(initial_liquid=0, mean_annual_return=0.05, std_annual_return=0.1),
        career_states=[state],
        transitions=[],
        strategies=[Strategy(id="s", name="S", description="", initial_choice_state_ids=[state.id])],
        scoring_weights=ScoringWeights(),
        simulation_settings=SimulationSettings(num_runs_per_scenario=4, random_seed=1),
    )
    override = SimulationSettings(num_runs_per_scenario=2, random_seed=42)
    res = simulate_config(cfg, override_settings=override)
    assert res.assumptions["num_runs_per_scenario"] == 2
    assert res.assumptions["time_step_months"] == override.time_step_months


def test_simulation_settings_and_transition_validation():
    with pytest.raises(ValueError):
        SimulationSettings(time_step_months=0)
    with pytest.raises(ValueError):
        Transition(id="bad", from_state_id="a", to_state_id="b", type="move", base_annual_prob=-0.1)


def test_cost_of_living_zero_safe_min_ratio():
    loc = Location(id="loc", name="Loc", col_annual=0, state_tax_rate=0.0)
    state = CareerState(
        id="state",
        label="State",
        role_title="Engineer",
        location_id=loc.id,
        employment_status="employed",
        compensation=None,
    )
    cfg = ConfigPayload(
        locations=[loc],
        portfolio_settings=PortfolioSettings(initial_liquid=1000, mean_annual_return=0, std_annual_return=0),
        career_states=[state],
        transitions=[],
        strategies=[Strategy(id="s", name="S", description="", initial_choice_state_ids=[state.id])],
        scoring_weights=ScoringWeights(),
        simulation_settings=SimulationSettings(num_runs_per_scenario=1, random_seed=0, risk_penalty_lambda=0.0),
    )
    res = simulate_config(cfg)
    downside = res.all_scenarios[0].downside
    assert downside["p_liquid_lt_1x_col"] == pytest.approx(0.0)
    assert downside["p_liquid_lt_2x_col"] == pytest.approx(0.0)
