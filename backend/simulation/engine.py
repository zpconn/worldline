from __future__ import annotations

import copy
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from backend.models.domain import (
    CareerState,
    Compensation,
    ConfigPayload,
    Location,
    PortfolioSettings,
    ScenarioResult,
    SimulationResult,
    SimulationSettings,
    Strategy,
    Transition,
    ScoringWeights,
)


def hazard_to_step(annual_hazard: float, step_months: int) -> float:
    """Convert an annual hazard/probability to per-step probability."""
    annual_hazard = max(0.0, annual_hazard)
    lam = -math.log(max(1e-12, 1 - min(0.999999, annual_hazard)))
    step_rate = lam * (step_months / 12.0)
    return 1 - math.exp(-step_rate)


def _annual_to_step_params(mean: float, std: float, step_months: int) -> Tuple[float, float]:
    """Rough conversion of annual mean/std to per-step for normal returns."""
    steps_per_year = 12.0 / step_months
    step_mean = mean / steps_per_year
    step_std = std / math.sqrt(steps_per_year)
    return step_mean, step_std


def _calc_monthly_cash_flow(
    state: CareerState,
    location: Location,
    month_index: int,
    comp_adjust_factor: float,
) -> float:
    comp = state.compensation
    if not comp:
        income = 0.0
    else:
        income = comp.base_annual / 12.0
        income += comp.bonus_target_annual * comp.bonus_prob_pay / 12.0
        for grant in comp.equity:
            if month_index >= grant.cliff_months:
                income += grant.grant_value / (grant.vest_years * 12.0)
        for ot in comp.one_times:
            if ot.month_offset == month_index:
                income += ot.amount
        income *= comp_adjust_factor
    after_tax_income = income * (1 - location.state_tax_rate)
    col = location.col_annual / 12.0
    return after_tax_income - col


def _npv(cash_flows: List[float], discount_rate: float, step_months: int, horizon_months: int) -> float:
    total = 0.0
    for i, cf in enumerate(cash_flows):
        t_months = i * step_months
        if t_months > horizon_months:
            break
        total += cf / ((1 + discount_rate) ** (t_months / 12.0))
    return total


def _scenario_label(strategy: Strategy, initial_state_id: str, states: Dict[str, CareerState]) -> str:
    state_label = states.get(initial_state_id).label if initial_state_id in states else initial_state_id
    return f"{strategy.name} | start: {state_label}"


@dataclass
class _RunResult:
    npv_short: float
    npv_long: float
    final_portfolio_short: float
    final_portfolio_long: float
    min_ratio: float
    unemployment_6: bool
    unemployment_12: bool
    unemployment_24: bool
    lower_pay_reentry: bool
    haircut_values: List[float]
    career_capital_score: float
    enjoyment_score: float
    location_fit_score: float
    location_counts: Dict[str, int]


def _resolve_max_workers(max_workers: int | None, runs: int) -> int:
    if runs <= 1:
        return 1
    if max_workers is None:
        return max(1, min(runs, os.cpu_count() or 1))
    return max(1, min(max_workers, runs))


def _chunk(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _choose_executor(parallel: bool, executor: str, rng: Any) -> str:
    if not parallel:
        return "none"
    if rng is not None and not isinstance(rng, np.random.Generator):
        return "none"
    if executor not in ("process", "thread", "none"):
        return "process"
    return executor


def simulate_config(
    config: ConfigPayload,
    override_settings: SimulationSettings | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    executor: str = "process",
) -> SimulationResult:
    settings = override_settings or config.simulation_settings
    states: Dict[str, CareerState] = {s.id: s for s in config.career_states}
    locations: Dict[str, Location] = {l.id: l for l in config.locations}
    transitions_by_from: Dict[str, List[Transition]] = {}
    for t in config.transitions:
        transitions_by_from.setdefault(t.from_state_id, []).append(t)

    _validate_dag(states, transitions_by_from, settings)

    scenario_results: List[ScenarioResult] = []
    scenario_count = sum(len(s.initial_choice_state_ids) for s in config.strategies)
    scenario_seeds = np.random.SeedSequence(settings.random_seed).spawn(scenario_count) if scenario_count else []
    scenario_idx = 0
    for strategy in config.strategies:
        for init_state in strategy.initial_choice_state_ids:
            res_short, res_long = _simulate_scenario(
                strategy=strategy,
                initial_state_id=init_state,
                states=states,
                locations=locations,
                transitions_by_from=transitions_by_from,
                portfolio_settings=config.portfolio_settings,
                scoring_weights=config.scoring_weights,
                settings=settings,
                seed_sequence=scenario_seeds[scenario_idx] if scenario_idx < len(scenario_seeds) else None,
                parallel=parallel,
                max_workers=max_workers,
                executor=executor,
            )
            scenario_idx += 1
            scenario_results.extend([res_short, res_long])

    best_5y = _best_for_horizon(scenario_results, 5)
    best_10y = _best_for_horizon(scenario_results, 10)

    downside_dashboard = _aggregate_downside(scenario_results)
    sensitivity = _run_sensitivity(config, scenario_results, settings)

    assumptions = {
        "time_step_months": settings.time_step_months,
        "risk_penalty_lambda": settings.risk_penalty_lambda,
        "cvar_alpha": settings.cvar_alpha,
        "num_runs_per_scenario": settings.num_runs_per_scenario,
        "max_workers": _resolve_max_workers(max_workers, settings.num_runs_per_scenario)
        if _choose_executor(parallel, executor, None) != "none"
        else 1,
        "executor": _choose_executor(parallel, executor, None),
    }

    return SimulationResult(
        best_5y=best_5y,
        best_10y=best_10y,
        all_scenarios=scenario_results,
        downside_dashboard=downside_dashboard,
        sensitivity=sensitivity,
        assumptions=assumptions,
    )


def _simulate_single_run(
    strategy: Strategy,
    initial_state_id: str,
    states: Dict[str, CareerState],
    locations: Dict[str, Location],
    transitions_by_from: Dict[str, List[Transition]],
    portfolio_settings: PortfolioSettings,
    settings: SimulationSettings,
    rng: Any,
) -> _RunResult:
    horizon_long_months = settings.horizon_years_long * 12
    horizon_short_months = settings.horizon_years_short * 12
    step_months = settings.time_step_months

    state_id = initial_state_id
    month = 0
    portfolio = portfolio_settings.initial_liquid
    cash_flows: List[float] = []
    longest_unemp = 0
    current_unemp = 0
    comp_adjust_factor = 1.0
    last_comp_total = _total_comp(states[state_id].compensation)
    min_ratio_run = np.inf
    location_counts: Dict[str, int] = {}
    identity_sum = 0.0
    portability_sum = 0.0
    wellbeing_sum = 0.0
    steps = 0
    haircut_list: List[float] = []
    final_portfolio_short = 0.0
    lower_pay_reentry = False

    while month <= horizon_long_months:
        state = states[state_id]
        loc = locations[state.location_id]
        net_cash = _calc_monthly_cash_flow(state, loc, month, comp_adjust_factor)
        contribution = max(net_cash, 0.0) * portfolio_settings.contribution_rate
        net_cash_after_contrib = net_cash - contribution
        step_mean, step_std = _annual_to_step_params(
            portfolio_settings.mean_annual_return,
            portfolio_settings.std_annual_return,
            step_months,
        )
        monthly_return = rng.normal(step_mean, step_std)
        portfolio = max(0.0, (portfolio + net_cash_after_contrib + contribution) * (1.0 + monthly_return))
        cash_flows.append(net_cash_after_contrib)

        col_ratio = portfolio / max(1e-6, loc.col_annual / 12.0)
        min_ratio_run = min(min_ratio_run, col_ratio)

        location_counts[state.location_id] = location_counts.get(state.location_id, 0) + 1
        identity_sum += state.identity_brand.internal_stature
        portability_sum += state.identity_brand.external_portability
        wellbeing_sum += state.wellbeing
        steps += 1

        if state.employment_status.lower() == "unemployed":
            current_unemp += step_months
            longest_unemp = max(longest_unemp, current_unemp)
        else:
            if current_unemp > 0 and state.compensation:
                new_comp_total = _total_comp(state.compensation) * comp_adjust_factor
                if last_comp_total > 0 and new_comp_total < last_comp_total:
                    lower_pay_reentry = True
                    haircut_list.append((new_comp_total - last_comp_total) / last_comp_total)
                last_comp_total = new_comp_total
            current_unemp = 0

        if month >= horizon_short_months and final_portfolio_short == 0.0:
            final_portfolio_short = portfolio

        chosen_transition, lag = _select_transition(
            state_id,
            strategy,
            states,
            transitions_by_from,
            settings,
            rng,
        )
        if chosen_transition:
            comp_adjust_factor = 1.0 + (chosen_transition.delta.comp_adjustment_pct or 0.0)
            if chosen_transition.delta.relocation_cost:
                portfolio = max(0.0, portfolio - chosen_transition.delta.relocation_cost)
            state_id = chosen_transition.to_state_id
        else:
            comp_adjust_factor = 1.0

        month += step_months + (lag or 0)

    npv_short_val = _npv(cash_flows, settings.discount_rate_real, step_months, horizon_short_months)
    npv_long_val = _npv(cash_flows, settings.discount_rate_real, step_months, horizon_long_months)
    final_portfolio_long = portfolio

    total_steps = max(steps, 1)
    career_capital = (identity_sum + portability_sum) / (2 * total_steps)
    enjoyment = wellbeing_sum / total_steps
    preferred = 0
    if strategy.preferred_locations:
        for loc_id, count in location_counts.items():
            if loc_id in strategy.preferred_locations:
                preferred += count
    location_fit = preferred / total_steps if total_steps else 0.0

    return _RunResult(
        npv_short=npv_short_val,
        npv_long=npv_long_val,
        final_portfolio_short=final_portfolio_short,
        final_portfolio_long=final_portfolio_long,
        min_ratio=min_ratio_run,
        unemployment_6=longest_unemp >= 6,
        unemployment_12=longest_unemp >= 12,
        unemployment_24=longest_unemp >= 24,
        lower_pay_reentry=lower_pay_reentry,
        haircut_values=haircut_list,
        career_capital_score=career_capital,
        enjoyment_score=enjoyment,
        location_fit_score=location_fit,
        location_counts=location_counts,
    )


def _run_batch(
    strategy: Strategy,
    initial_state_id: str,
    states: Dict[str, CareerState],
    locations: Dict[str, Location],
    transitions_by_from: Dict[str, List[Transition]],
    portfolio_settings: PortfolioSettings,
    settings: SimulationSettings,
    seed_batch: List[np.random.SeedSequence],
) -> List[_RunResult]:
    results: List[_RunResult] = []
    for seq in seed_batch:
        rng = np.random.default_rng(seq)
        results.append(
            _simulate_single_run(
                strategy=strategy,
                initial_state_id=initial_state_id,
                states=states,
                locations=locations,
                transitions_by_from=transitions_by_from,
                portfolio_settings=portfolio_settings,
                settings=settings,
                rng=rng,
            )
        )
    return results


def _run_batch_process(args: Tuple[Any, ...]) -> List[_RunResult]:
    return _run_batch(*args)


def _run_batch_thread(args: Tuple[Any, ...]) -> List[_RunResult]:
    # Thread pool fallback shares code with process pool; kept separate for clarity.
    return _run_batch(*args)


def _simulate_scenario(
    strategy: Strategy,
    initial_state_id: str,
    states: Dict[str, CareerState],
    locations: Dict[str, Location],
    transitions_by_from: Dict[str, List[Transition]],
    portfolio_settings: PortfolioSettings,
    scoring_weights: ScoringWeights,
    settings: SimulationSettings,
    rng: np.random.Generator | None = None,
    seed_sequence: np.random.SeedSequence | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    executor: str = "process",
) -> Tuple[ScenarioResult, ScenarioResult]:
    runs = settings.num_runs_per_scenario
    if runs <= 0:
        raise ValueError("num_runs_per_scenario must be positive")
    mode = _choose_executor(parallel, executor, rng)
    worker_count = _resolve_max_workers(max_workers, runs) if mode != "none" else 1

    if mode == "none":
        if rng is not None and not isinstance(rng, np.random.Generator):
            run_rngs = [rng] * runs
        else:
            seed_seq = seed_sequence or np.random.SeedSequence(settings.random_seed)
            run_rngs = [np.random.default_rng(seq) for seq in seed_seq.spawn(runs)]
        run_results = [
            _simulate_single_run(
                strategy=strategy,
                initial_state_id=initial_state_id,
                states=states,
                locations=locations,
                transitions_by_from=transitions_by_from,
                portfolio_settings=portfolio_settings,
                settings=settings,
                rng=run_rng,
            )
            for run_rng in run_rngs
        ]
    else:
        seed_seq = seed_sequence or np.random.SeedSequence(settings.random_seed)
        child_sequences = seed_seq.spawn(runs)
        target_tasks = max(1, worker_count * 4)
        batch_size = max(1, math.ceil(runs / target_tasks))
        batches = _chunk(child_sequences, batch_size)

        worker_func = _run_batch_process if mode == "process" else _run_batch_thread
        ExecutorCls = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor

        with ExecutorCls(max_workers=worker_count) as pool:
            run_results = []
            for batch_results in pool.map(
                worker_func,
                [
                    (
                        strategy,
                        initial_state_id,
                        states,
                        locations,
                        transitions_by_from,
                        portfolio_settings,
                        settings,
                        batch,
                    )
                    for batch in batches
                ],
            ):
                run_results.extend(batch_results)

    npv_short = np.array([r.npv_short for r in run_results])
    npv_long = np.array([r.npv_long for r in run_results])
    final_portfolio_short = np.array([r.final_portfolio_short for r in run_results])
    final_portfolio_long = np.array([r.final_portfolio_long for r in run_results])
    min_ratio = np.array([r.min_ratio for r in run_results])
    unemployment_6 = np.array([r.unemployment_6 for r in run_results])
    unemployment_12 = np.array([r.unemployment_12 for r in run_results])
    unemployment_24 = np.array([r.unemployment_24 for r in run_results])
    lower_pay_reentry = np.array([r.lower_pay_reentry for r in run_results])
    haircut_values: List[float] = []
    location_time_accum: Dict[str, float] = {}
    career_capital_scores: List[float] = []
    enjoyment_scores: List[float] = []
    location_fit_scores: List[float] = []

    for r in run_results:
        haircut_values.extend(r.haircut_values)
        career_capital_scores.append(r.career_capital_score)
        enjoyment_scores.append(r.enjoyment_score)
        location_fit_scores.append(r.location_fit_score)
        for loc_id, count in r.location_counts.items():
            location_time_accum[loc_id] = location_time_accum.get(loc_id, 0) + count

    downside = _downside_metrics(
        min_ratio,
        unemployment_6,
        unemployment_12,
        unemployment_24,
        lower_pay_reentry,
        haircut_values,
    )
    portfolio_stats_short = _portfolio_stats(final_portfolio_short)
    portfolio_stats_long = _portfolio_stats(final_portfolio_long)

    nonfinancial = {
        "career_capital_avg": float(np.mean(career_capital_scores)) if career_capital_scores else 0.0,
        "enjoyment_avg": float(np.mean(enjoyment_scores)) if enjoyment_scores else 0.0,
        "location_fit_avg": float(np.mean(location_fit_scores)) if location_fit_scores else 0.0,
        "location_time_shares": _normalize_dict(location_time_accum),
    }

    res_short = _build_result(
        strategy,
        initial_state_id,
        states,
        npv_short,
        portfolio_stats_short,
        downside,
        nonfinancial,
        scoring_weights,
        horizon_years=settings.horizon_years_short,
        risk_lambda=settings.risk_penalty_lambda,
        cvar_alpha=settings.cvar_alpha,
    )
    res_long = _build_result(
        strategy,
        initial_state_id,
        states,
        npv_long,
        portfolio_stats_long,
        downside,
        nonfinancial,
        scoring_weights,
        horizon_years=settings.horizon_years_long,
        risk_lambda=settings.risk_penalty_lambda,
        cvar_alpha=settings.cvar_alpha,
    )
    return res_short, res_long


def _select_transition(
    state_id: str,
    strategy: Strategy,
    states: Dict[str, CareerState],
    transitions_by_from: Dict[str, List[Transition]],
    settings: SimulationSettings,
    rng: np.random.Generator,
) -> Tuple[Transition | None, int]:
    transitions = transitions_by_from.get(state_id, [])
    if not transitions:
        return None, 0
    probs: List[float] = []
    eligible: List[Transition] = []
    for tr in transitions:
        to_state = states.get(tr.to_state_id)
        from_state = states.get(tr.from_state_id)
        if not to_state or not from_state:
            continue
        if strategy.disallowed_locations and to_state.location_id in strategy.disallowed_locations:
            continue
        if to_state.t_months_min < from_state.t_months_min:
            continue
        if strategy.paycut_floor_pct is not None and from_state.compensation and to_state.compensation:
            current_total = _total_comp(from_state.compensation)
            target_total = _total_comp(to_state.compensation)
            if target_total < current_total * (1 + strategy.paycut_floor_pct):
                continue
        step_prob = hazard_to_step(tr.base_annual_prob * tr.desire_multiplier, settings.time_step_months)
        if step_prob <= 0:
            continue
        eligible.append(tr)
        probs.append(step_prob)

    if not eligible:
        return None, 0

    stay_prob = max(0.0, 1.0 - sum(probs))
    probs.append(stay_prob)
    probs_array = np.array(probs, dtype=float)
    probs_array /= probs_array.sum()
    idx = rng.choice(len(probs_array), p=probs_array)
    if idx >= len(eligible):
        return None, 0
    chosen = eligible[idx]
    return chosen, chosen.lag_months


def _total_comp(comp: Compensation | None) -> float:
    if not comp:
        return 0.0
    return comp.base_annual + comp.bonus_target_annual * comp.bonus_prob_pay


def _downside_metrics(
    min_ratio: np.ndarray,
    unemployment_6: np.ndarray,
    unemployment_12: np.ndarray,
    unemployment_24: np.ndarray,
    lower_pay_reentry: np.ndarray,
    haircut_values: List[float],
) -> Dict[str, Any]:
    probs = lambda arr: float(np.mean(arr))
    return {
        "p_liquid_lt_1x_col": probs(min_ratio < 1.0),
        "p_liquid_lt_2x_col": probs(min_ratio < 2.0),
        "p_unemp_ge_6m": probs(unemployment_6),
        "p_unemp_ge_12m": probs(unemployment_12),
        "p_unemp_ge_24m": probs(unemployment_24),
        "p_lower_pay_reentry": probs(lower_pay_reentry),
        "median_pay_haircut": float(np.median(haircut_values)) if haircut_values else 0.0,
    }


def _portfolio_stats(final_values: np.ndarray) -> Dict[str, Any]:
    return {
        "p50": float(np.percentile(final_values, 50)),
        "p10": float(np.percentile(final_values, 10)),
        "p90": float(np.percentile(final_values, 90)),
        "mean": float(np.mean(final_values)),
    }


def _utility(npv: np.ndarray, scoring_weights: ScoringWeights, downside: Dict[str, Any], nonfinancial: Dict[str, Any], risk_lambda: float) -> float:
    ev = float(np.mean(npv))
    var = float(np.var(npv))
    financial_score = ev - risk_lambda * var
    career_capital = nonfinancial.get("career_capital_avg", 0.0)
    enjoyment = nonfinancial.get("enjoyment_avg", 0.0)
    location_fit = nonfinancial.get("location_fit_avg", 0.0)
    legacy = 0.0
    return (
        scoring_weights.financial * financial_score
        + scoring_weights.career_capital * career_capital
        + scoring_weights.enjoyment_identity * enjoyment
        + scoring_weights.location_fit * location_fit
        + scoring_weights.legacy * legacy
    )


def _build_result(
    strategy: Strategy,
    initial_state_id: str,
    states: Dict[str, CareerState],
    npv_arr: np.ndarray,
    portfolio_stats: Dict[str, Any],
    downside: Dict[str, Any],
    nonfinancial: Dict[str, Any],
    scoring_weights: ScoringWeights,
    horizon_years: int,
    risk_lambda: float,
    cvar_alpha: float,
) -> ScenarioResult:
    ev = float(np.mean(npv_arr))
    var = float(np.var(npv_arr))
    sorted_npv = np.sort(npv_arr)
    cutoff = max(1, int(len(sorted_npv) * cvar_alpha))
    cvar = float(np.mean(sorted_npv[:cutoff]))
    utility_score = _utility(npv_arr, scoring_weights, downside, nonfinancial, risk_lambda=risk_lambda)
    return ScenarioResult(
        strategy_id=strategy.id,
        initial_state_id=initial_state_id,
        label=_scenario_label(strategy, initial_state_id, states),
        horizon_years=horizon_years,
        ev_npv=ev,
        var_npv=var,
        cvar_npv=cvar,
        utility_score=utility_score,
        portfolio_stats=portfolio_stats,
        downside=downside,
        nonfinancial=nonfinancial,
    )


def _best_for_horizon(results: List[ScenarioResult], horizon: int) -> ScenarioResult | None:
    filtered = [r for r in results if r.horizon_years == horizon]
    if not filtered:
        return None
    return max(filtered, key=lambda r: r.utility_score)


def _aggregate_downside(results: List[ScenarioResult]) -> Dict[str, Any]:
    by_strategy: Dict[str, Dict[str, List[float]]] = {}
    for res in results:
        bucket = by_strategy.setdefault(res.strategy_id, {k: [] for k in res.downside})
        for k, v in res.downside.items():
            bucket[k].append(v)
    agg = {
        sid: {k: float(np.mean(v)) for k, v in metrics.items()}
        for sid, metrics in by_strategy.items()
    }
    return {"by_strategy": agg}


def _run_sensitivity(config: ConfigPayload, results: List[ScenarioResult], settings: SimulationSettings) -> Dict[str, Any]:
    if not results:
        return {}
    base = max(results, key=lambda r: r.utility_score)
    params = [
        ("portfolio_return_mean", 0.9, 1.1),
        ("portfolio_vol", 0.9, 1.1),
        ("transition_prob", 0.8, 1.2),
    ]
    entries = []
    for name, low_factor, high_factor in params:
        low_cfg = copy.deepcopy(config)
        high_cfg = copy.deepcopy(config)
        if name == "portfolio_return_mean":
            low_cfg.portfolio_settings.mean_annual_return *= low_factor
            high_cfg.portfolio_settings.mean_annual_return *= high_factor
        elif name == "portfolio_vol":
            low_cfg.portfolio_settings.std_annual_return *= low_factor
            high_cfg.portfolio_settings.std_annual_return *= high_factor
        elif name == "transition_prob":
            for cfg, factor in ((low_cfg, low_factor), (high_cfg, high_factor)):
                for t in cfg.transitions:
                    t.base_annual_prob *= factor
        base_strategy = next((s for s in config.strategies if s.id == base.strategy_id), None)
        if not base_strategy:
            continue
        low_res, _ = _simulate_scenario(
            base_strategy,
            base.initial_state_id,
            {s.id: s for s in config.career_states},
            {l.id: l for l in config.locations},
            _build_transition_map(low_cfg.transitions),
            low_cfg.portfolio_settings,
            config.scoring_weights,
            SimulationSettings(**{**settings.dict(), "num_runs_per_scenario": min(500, settings.num_runs_per_scenario)}),
            np.random.default_rng(settings.random_seed),
        )
        high_res, _ = _simulate_scenario(
            base_strategy,
            base.initial_state_id,
            {s.id: s for s in config.career_states},
            {l.id: l for l in config.locations},
            _build_transition_map(high_cfg.transitions),
            high_cfg.portfolio_settings,
            config.scoring_weights,
            SimulationSettings(**{**settings.dict(), "num_runs_per_scenario": min(500, settings.num_runs_per_scenario)}),
            np.random.default_rng(settings.random_seed),
        )
        entries.append(
            {
                "parameter": name,
                "base_utility": base.utility_score,
                "low_utility": low_res.utility_score,
                "high_utility": high_res.utility_score,
            }
        )
    return {"parameters": entries}


def _build_transition_map(transitions: List[Transition]) -> Dict[str, List[Transition]]:
    mapping: Dict[str, List[Transition]] = {}
    for t in transitions:
        mapping.setdefault(t.from_state_id, []).append(t)
    return mapping


def _validate_dag(states: Dict[str, CareerState], transitions_by_from: Dict[str, List[Transition]], settings: SimulationSettings) -> None:
    # Enforce time monotonicity guard (no transitions to earlier-available states).
    for from_id, edges in transitions_by_from.items():
        from_state = states.get(from_id)
        if not from_state:
            continue
        for t in edges:
            to_state = states.get(t.to_state_id)
            if not to_state:
                continue
            if to_state.t_months_min < from_state.t_months_min:
                raise ValueError(
                    f"Transition {t.id} goes to state with earlier t_months_min; violates DAG time ordering"
                )


def _normalize_dict(d: Dict[str, float]) -> Dict[str, float]:
    total = sum(d.values()) or 1.0
    return {k: v / total for k, v in d.items()}
