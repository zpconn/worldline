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
