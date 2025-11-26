import pytest

from backend.models.domain import (
    CareerState,
    Compensation,
    ConfigPayload,
    Location,
    PortfolioSettings,
    ScoringWeights,
    SimulationSettings,
    Strategy,
    Transition,
)
from backend.simulation.engine import hazard_to_step, simulate_config


def test_hazard_conversion():
    monthly = hazard_to_step(0.24, 1)
    assert 0 < monthly < 1
    quarterly = hazard_to_step(0.24, 3)
    assert quarterly > monthly


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
