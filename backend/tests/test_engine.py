import numpy as np
import pytest

from backend.models.domain import (
    CareerState,
    Compensation,
    ConfigPayload,
    Location,
    PortfolioSettings,
    ScenarioResult,
    ScoringWeights,
    SimulationSettings,
    Strategy,
    Transition,
)
from backend.simulation.engine import (
    _best_for_horizon,
    _aggregate_downside,
    _downside_metrics,
    _normalize_dict,
    _portfolio_stats,
    _select_transition,
    _utility,
    hazard_to_step,
    simulate_config,
)


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
