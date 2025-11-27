// Form-like builder for locations, states, transitions, strategies, and weights.
import React from "react";
import {
  CareerState,
  Compensation,
  ConfigPayload,
  EquityGrant,
  PortfolioSettings,
  Strategy,
  Transition,
} from "../types";

type Props = {
  config: ConfigPayload;
  onChange: (config: ConfigPayload) => void;
};

/**
 * ConfigBuilder renders a form-driven facade over the raw configuration schema. It lets users add,
 * edit, and remove every domain primitive (locations, career states, transitions, strategies, and
 * simulation knobs) while keeping the parent component as the single source of truth. The component
 * favors small, composable helpers to mutate slices of the config immutably so each input stays
 * responsive and React can reconcile updates efficiently.
 */
const ConfigBuilder: React.FC<Props> = ({ config, onChange }) => {
  /**
   * Helper to immutably update a config section while preserving other fields. By fanning updates
   * through this function, every sub-editor (locations, transitions, strategies) can surgically
   * replace its slice without worrying about stale references or shallow copies. This pattern keeps
   * React's change detection happy and prevents accidental state loss when multiple sections update.
   */
  const updateConfig = <K extends keyof ConfigPayload>(key: K, value: ConfigPayload[K]) => {
    onChange({ ...config, [key]: value });
  };

  /**
   * Parse numeric inputs from string fields while providing a deterministic fallback. Text inputs
   * can emit empty strings or non-numeric garbage; coercing through Number and then guarding with
   * Number.isFinite keeps downstream calculations from receiving NaN and derailing the UI. The
   * fallback value is chosen per-call so each field can define a sensible default.
   */
  const numberVal = (val: string, fallback = 0): number => {
    const n = Number(val);
    return Number.isFinite(n) ? n : fallback;
  };

  /**
   * Update portfolio assumptions while preserving other config fields. Portfolio settings drive the
   * simulated investment account and are conceptually independent from the DAG, so updates merge the
   * incoming partial with the existing object to avoid clobbering sibling properties the user did
   * not touch. This keeps the builder resilient to partial edits from multiple inputs at once.
   */
  const setPortfolio = (partial: Partial<PortfolioSettings>) => {
    updateConfig("portfolio_settings", { ...config.portfolio_settings, ...partial });
  };

  /**
   * Update scoring weights with an immutable merge. The scoring weights influence composite utility
   * calculations on the backend; by merging partials we allow each numeric input to update just its
   * own weight while the rest stay intact. This avoids forcing the user to re-enter the full weight
   * vector every time they tweak one dial.
   */
  const setScoring = (partial: Partial<ConfigPayload["scoring_weights"]>) => {
    updateConfig("scoring_weights", { ...config.scoring_weights, ...partial });
  };

  /**
   * Update simulation settings such as horizon lengths, cadence, and sampling. These parameters
   * control the Monte Carlo engine, so the helper merges partial updates to avoid discarding values
   * the user has already tuned. Keeping this function narrow also makes it easy to reason about
   * which inputs will trigger re-runs of the simulation downstream.
   */
  const setSimulation = (partial: Partial<ConfigPayload["simulation_settings"]>) => {
    updateConfig("simulation_settings", { ...config.simulation_settings, ...partial });
  };

  /**
   * Append a new location entry seeded with conservative placeholder values. Locations underpin cost
   * of living, tax considerations, and preference filters in strategies, so creating a new one
   * generates a unique id, a human-readable name, and baseline financial assumptions the user can
   * immediately adjust. The function clones the list to maintain immutability guarantees.
   */
  const addLocation = () => {
    const next = [
      ...config.locations,
      {
        id: `loc-${config.locations.length + 1}`,
        name: "New location",
        col_annual: 85000,
        state_tax_rate: 0.1,
      },
    ];
    updateConfig("locations", next);
  };

  /**
   * Apply partial edits to a specific location by index. The update is index-based rather than
   * id-based because the UI already has the array index at render time, and mapping over the array
   * with a conditional replacement keeps the operation immutable. This prevents unintended ripple
   * effects on other locations while still giving each input full control over its target fields.
   */
  const updateLocation = (idx: number, partial: Partial<ConfigPayload["locations"][number]>) => {
    const next = config.locations.map((loc, i) => (i === idx ? { ...loc, ...partial } : loc));
    updateConfig("locations", next);
  };

  /**
   * Remove a location by index. The function filters out the targeted element and leaves the rest of
   * the list untouched, which makes deletions predictable and reversible through external undo
   * mechanisms. Because strategies and states reference locations by id, downstream callers are
   * responsible for keeping those references in sync after a removal.
   */
  const removeLocation = (idx: number) => {
    const next = config.locations.filter((_, i) => i !== idx);
    updateConfig("locations", next);
  };

  /**
   * Generate a baseline compensation package for new career states. Providing a factory function
   * instead of a constant ensures each state receives a fresh object (avoiding accidental shared
   * references) and encapsulates the default mix of salary, bonus, and equity so changes propagate
   * consistently whenever a new state is added or toggled back to paid.
   */
  const defaultCompensation = (): Compensation => ({
    base_annual: 150000,
    bonus_target_annual: 30000,
    bonus_prob_pay: 0.7,
    equity: [],
    one_times: [],
  });

  /**
   * Append a new career state pre-populated with sensible defaults. The helper chooses the first
   * known location as the starting geography, seeds compensation with the default template, and
   * initializes wellbeing/identity scores to mid-range values so charts and constraints behave
   * immediately. Using a generated id tied to the current length avoids collisions during rapid
   * entry and keeps the array append immutable.
   */
  const addState = () => {
    const firstLocation = config.locations[0]?.id || "home";
    const next: CareerState = {
      id: `state-${config.career_states.length + 1}`,
      label: "New state",
      role_title: "Role title",
      location_id: firstLocation,
      employment_status: "employed",
      compensation: defaultCompensation(),
      wellbeing: 0.5,
      identity_brand: { external_portability: 0.5, internal_stature: 0.5 },
      t_months_min: 0,
      constraints: { paycut_floor_pct: -0.4, must_clear_col_after_tax: true },
      beliefs: {},
    };
    updateConfig("career_states", [...config.career_states, next]);
  };

  /**
   * Apply partial edits to a career state by index. The function copies the array, replaces only the
   * targeted state, and leaves all other entries untouched, ensuring React can reconcile without
   * unnecessary re-renders. This pattern also protects against unintentional wholesale replacement
   * of the states list when only one field needs to change.
   */
  const updateState = (idx: number, partial: Partial<CareerState>) => {
    const next = config.career_states.map((s, i) => (i === idx ? { ...s, ...partial } : s));
    updateConfig("career_states", next);
  };

  /**
   * Remove a career state by index. Filtering by index instead of id aligns with how the UI renders
   * and keeps the function free of lookup logic, but it also means any external references (like
   * transitions pointing at the removed state) must be reconciled elsewhere to avoid dangling ids.
   */
  const removeState = (idx: number) => {
    const next = config.career_states.filter((_, i) => i !== idx);
    updateConfig("career_states", next);
  };

  /**
   * Toggle whether a state has compensation. When disabling pay we null out the compensation object;
   * when enabling we regenerate the default package to avoid sharing references between states. This
   * lets the UI represent unpaid sabbaticals or unemployment explicitly while preserving the ability
   * to reintroduce pay with a clean slate.
   */
  const toggleCompensation = (idx: number) => {
    const state = config.career_states[idx];
    updateState(idx, { compensation: state.compensation ? null : defaultCompensation() });
  };

  /**
   * Apply partial edits to a state's compensation, guarding against unpaid states. Because unpaid
   * states store `compensation` as null, the function early-returns if there is no package to merge.
   * When present, the merge strategy preserves untouched fields and creates a new compensation object
   * so React sees a change and re-renders dependent inputs.
   */
  const updateCompensation = (idx: number, partial: Partial<Compensation>) => {
    const state = config.career_states[idx];
    if (!state.compensation) return;
    const next = { ...state.compensation, ...partial };
    updateState(idx, { compensation: next });
  };

  /**
   * Add a default equity grant to a given state. Grants are appended rather than replaced, and the
   * function exits quietly if the state is unpaid because equity is undefined in that context.
   * Seeding a recognizable RSU template gives users a starting point they can quickly customize for
   * value, vesting schedule, and cliff.
   */
  const addEquity = (stateIdx: number) => {
    const state = config.career_states[stateIdx];
    if (!state.compensation) return;
    const grants: EquityGrant[] = [
      ...state.compensation.equity,
      { type: "RSU", grant_value: 100000, vest_years: 4, cliff_months: 12 },
    ];
    updateCompensation(stateIdx, { equity: grants });
  };

  /**
   * Apply partial edits to a specific equity grant within a state's compensation. The function maps
   * over the grants array, replacing only the targeted index to maintain immutability. If the state
   * has no compensation, it quietly exits to avoid throwing in response to UI toggles that hide pay.
   */
  const updateEquity = (stateIdx: number, grantIdx: number, partial: Partial<EquityGrant>) => {
    const state = config.career_states[stateIdx];
    if (!state.compensation) return;
    const grants = state.compensation.equity.map((g, i) => (i === grantIdx ? { ...g, ...partial } : g));
    updateCompensation(stateIdx, { equity: grants });
  };

  /**
   * Remove a specific equity grant by index. Filtering produces a new array so React can detect the
   * change, and the guard against missing compensation avoids runtime errors when toggling between
   * paid and unpaid states. This keeps the grants list consistent with user actions in the UI.
   */
  const removeEquity = (stateIdx: number, grantIdx: number) => {
    const state = config.career_states[stateIdx];
    if (!state.compensation) return;
    const grants = state.compensation.equity.filter((_, i) => i !== grantIdx);
    updateCompensation(stateIdx, { equity: grants });
  };

  /**
   * Append a new transition edge between career states with sensible defaults. Because transitions
   * require valid endpoints, the function picks the first two states when available and falls back
   * to a self-loop if only one exists. Default hazard parameters (probabilities, lag, guards) are
   * set to conservative values so users can see the edge immediately and refine its behavior.
   */
  const addTransition = () => {
    const from = config.career_states[0]?.id || "state-1";
    const to = config.career_states[1]?.id || from;
    const next: Transition = {
      id: `t${config.transitions.length + 1}`,
      from_state_id: from,
      to_state_id: to,
      type: "move",
      base_annual_prob: 0.1,
      desire_multiplier: 1,
      lag_months: 0,
      delta: { relocation_cost: null },
      guards: { respect_paycut_floor: true, must_clear_col_after_tax: true, requires_terms_accepted: false, custom_conditions: {} },
    };
    updateConfig("transitions", [...config.transitions, next]);
  };

  /**
   * Apply partial edits to a transition. Mapping with a conditional replacement keeps the update
   * immutable and scoped to the intended edge. This approach also preserves extra fields added later
   * by the backend because we merge into the existing object rather than recreating it from scratch.
   */
  const updateTransition = (idx: number, partial: Partial<Transition>) => {
    const next = config.transitions.map((t, i) => (i === idx ? { ...t, ...partial } : t));
    updateConfig("transitions", next);
  };

  /**
   * Update the delta section of a transition, which captures financial and identity shocks tied to
   * the move. Because `delta` itself is optional, the function first ensures there is an object to
   * merge into, then overlays the supplied partial. This keeps null/undefined distinctions intact so
   * the backend can differentiate between "no change" and "explicitly zero".
   */
  const updateTransitionDelta = (idx: number, partial: Partial<NonNullable<Transition["delta"]>>) => {
    const t = config.transitions[idx];
    updateTransition(idx, { delta: { ...(t.delta || {}), ...partial } });
  };

  /**
   * Remove a transition by index. Filtering keeps the operation immutable and ordered, which matters
   * for readability in the UI list. Downstream consumers referencing transition ids should reconcile
   * independently, mirroring how state and location deletions are handled.
   */
  const removeTransition = (idx: number) => {
    const next = config.transitions.filter((_, i) => i !== idx);
    updateConfig("transitions", next);
  };

  /**
   * Append a new strategy, seeding it to start from the first available state. Strategies orchestrate
   * how transitions are chosen during simulations, so the helper gives them a name, description, and
   * permissive defaults for paycut floors and location lists to minimize user friction. By cloning
   * and appending, the operation preserves immutability for predictable React updates.
   */
  const addStrategy = () => {
    const defaultState = config.career_states[0]?.id ? [config.career_states[0].id] : [];
    const next: Strategy = {
      id: `strategy-${config.strategies.length + 1}`,
      name: "New strategy",
      description: "Describe the policy",
      initial_choice_state_ids: defaultState,
      preferred_locations: [],
      disallowed_locations: [],
      paycut_floor_pct: -0.25,
      rules: [],
    };
    updateConfig("strategies", [...config.strategies, next]);
  };

  /**
   * Apply partial edits to a strategy. The merge approach retains any unedited fields (like nested
   * rules or location preferences) and keeps the update scoped to a single strategy record. This
   * prevents collateral changes when multiple inputs inside the same card fire on the same frame.
   */
  const updateStrategy = (idx: number, partial: Partial<Strategy>) => {
    const next = config.strategies.map((s, i) => (i === idx ? { ...s, ...partial } : s));
    updateConfig("strategies", next);
  };

  /**
   * Toggle membership of a string within an array, returning a new array each time. This helper is
   * shared between checkboxes for preferred/disallowed locations and initial state choices, keeping
   * the toggle logic centralized. Returning the same array shape (either filtered or appended) helps
   * React track identity changes and keeps immutability intact.
   */
  const toggleArrayValue = (arr: string[], value: string): string[] => {
    if (!value) return arr;
    return arr.includes(value) ? arr.filter((v) => v !== value) : [...arr, value];
  };

  /**
   * Toggle whether a state is an allowed initial choice for a given strategy. It delegates the array
   * membership logic to `toggleArrayValue` and then writes back the updated list for that strategy.
   * This keeps the UI checkboxes and the underlying config in sync without manual array juggling.
   */
  const toggleInitialChoice = (idx: number, stateId: string) => {
    const strategy = config.strategies[idx];
    updateStrategy(idx, { initial_choice_state_ids: toggleArrayValue(strategy.initial_choice_state_ids, stateId) });
  };

  /**
   * Toggle a location inside either the preferred or disallowed lists on a strategy. By accepting
   * the key as a parameter, the same helper powers both checklists, reducing duplication while still
   * keeping types narrow through `Pick`. The function clones the underlying array to maintain
   * immutability and allow React to spot the change.
   */
  const toggleLocationPref = (idx: number, key: "preferred_locations" | "disallowed_locations", locId: string) => {
    const strategy = config.strategies[idx];
    updateStrategy(idx, { [key]: toggleArrayValue(strategy[key], locId) } as Pick<Strategy, typeof key>);
  };

  return (
    <div style={root}>
      <div style={scrollArea}>
        <p style={{ marginTop: 0, marginBottom: 10, opacity: 0.8 }}>
          Use the builder to add locations, roles, transitions, and strategies. Your changes sync to the JSON preview automatically.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 8, marginBottom: 12 }}>
          <StatPill label="Locations" value={config.locations.length} />
          <StatPill label="States" value={config.career_states.length} />
          <StatPill label="Transitions" value={config.transitions.length} />
          <StatPill label="Strategies" value={config.strategies.length} />
        </div>

      <Section title="Portfolio & weights">
        <div style={gridTwo}>
          <LabeledInput label="Initial liquid" value={config.portfolio_settings.initial_liquid} onChange={(v) => setPortfolio({ initial_liquid: numberVal(v, 0) })} />
          <LabeledInput label="Mean annual return" value={config.portfolio_settings.mean_annual_return} onChange={(v) => setPortfolio({ mean_annual_return: numberVal(v, 0.05) })} step="0.01" />
          <LabeledInput label="Volatility (stdev)" value={config.portfolio_settings.std_annual_return} onChange={(v) => setPortfolio({ std_annual_return: numberVal(v, 0.1) })} step="0.01" />
          <LabeledInput label="Contribution rate" value={config.portfolio_settings.contribution_rate} onChange={(v) => setPortfolio({ contribution_rate: numberVal(v, 0) })} step="0.01" />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(140px,1fr))", gap: 8, marginTop: 10 }}>
          <LabeledInput label="Financial weight" value={config.scoring_weights.financial} onChange={(v) => setScoring({ financial: numberVal(v, 0.65) })} step="0.01" />
          <LabeledInput label="Career capital weight" value={config.scoring_weights.career_capital} onChange={(v) => setScoring({ career_capital: numberVal(v, 0.2) })} step="0.01" />
          <LabeledInput label="Enjoyment/identity weight" value={config.scoring_weights.enjoyment_identity} onChange={(v) => setScoring({ enjoyment_identity: numberVal(v, 0.1) })} step="0.01" />
          <LabeledInput label="Location fit weight" value={config.scoring_weights.location_fit} onChange={(v) => setScoring({ location_fit: numberVal(v, 0.1) })} step="0.01" />
          <LabeledInput label="Legacy weight" value={config.scoring_weights.legacy} onChange={(v) => setScoring({ legacy: numberVal(v, 0.05) })} step="0.01" />
        </div>
      </Section>

      <Section title="Locations">
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {config.locations.map((loc, idx) => (
            <div key={loc.id || idx} style={itemCard}>
              <div style={gridThree}>
                <LabeledInput label="Id" value={loc.id} onChange={(v) => updateLocation(idx, { id: v })} />
                <LabeledInput label="Name" value={loc.name} onChange={(v) => updateLocation(idx, { name: v })} />
                <LabeledInput label="COL annual" value={loc.col_annual} onChange={(v) => updateLocation(idx, { col_annual: numberVal(v, loc.col_annual) })} />
                <LabeledInput label="State tax rate" value={loc.state_tax_rate} onChange={(v) => updateLocation(idx, { state_tax_rate: numberVal(v, loc.state_tax_rate) })} step="0.01" />
              </div>
              <button onClick={() => removeLocation(idx)} style={ghostButton}>Remove location</button>
            </div>
          ))}
          <button onClick={addLocation} style={ctaButton}>Add location</button>
        </div>
      </Section>

      <Section title="Career states">
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {config.career_states.map((state, idx) => (
            <div key={state.id || idx} style={itemCard}>
              <div style={gridThree}>
                <LabeledInput label="Id" value={state.id} onChange={(v) => updateState(idx, { id: v })} />
                <LabeledInput label="Label" value={state.label} onChange={(v) => updateState(idx, { label: v })} />
                <LabeledInput label="Role title" value={state.role_title} onChange={(v) => updateState(idx, { role_title: v })} />
                <SelectInput label="Location" value={state.location_id} onChange={(v) => updateState(idx, { location_id: v })} options={config.locations.map((l) => ({ label: l.name, value: l.id }))} />
                <SelectInput label="Employment status" value={state.employment_status} onChange={(v) => updateState(idx, { employment_status: v })} options={[
                  { label: "Employed", value: "employed" },
                  { label: "Unemployed", value: "unemployed" },
                  { label: "Consulting", value: "consulting" },
                  { label: "Sabbatical", value: "sabbatical" },
                ]} />
                <LabeledInput label="Wellbeing (0-1)" value={state.wellbeing} onChange={(v) => updateState(idx, { wellbeing: numberVal(v, state.wellbeing) })} step="0.05" />
                <LabeledInput label="Identity external" value={state.identity_brand.external_portability} onChange={(v) => updateState(idx, { identity_brand: { ...state.identity_brand, external_portability: numberVal(v, state.identity_brand.external_portability) } })} step="0.05" />
                <LabeledInput label="Identity internal" value={state.identity_brand.internal_stature} onChange={(v) => updateState(idx, { identity_brand: { ...state.identity_brand, internal_stature: numberVal(v, state.identity_brand.internal_stature) } })} step="0.05" />
                <LabeledInput label="Min months in seat" value={state.t_months_min || 0} onChange={(v) => updateState(idx, { t_months_min: numberVal(v, 0) })} />
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 8, gap: 8 }}>
                <div style={{ fontWeight: 600 }}>Compensation</div>
                <button onClick={() => toggleCompensation(idx)} style={ghostButton}>
                  {state.compensation ? "Set as unpaid" : "Add compensation"}
                </button>
              </div>
              {state.compensation ? (
                <>
                  <div style={gridThree}>
                    <LabeledInput label="Base annual" value={state.compensation.base_annual} onChange={(v) => updateCompensation(idx, { base_annual: numberVal(v, state.compensation!.base_annual) })} />
                    <LabeledInput label="Bonus target" value={state.compensation.bonus_target_annual} onChange={(v) => updateCompensation(idx, { bonus_target_annual: numberVal(v, state.compensation!.bonus_target_annual) })} />
                    <LabeledInput label="Bonus prob" value={state.compensation.bonus_prob_pay} onChange={(v) => updateCompensation(idx, { bonus_prob_pay: numberVal(v, state.compensation!.bonus_prob_pay) })} step="0.05" />
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div style={{ fontWeight: 600 }}>Equity grants</div>
                      <button onClick={() => addEquity(idx)} style={ghostButton}>Add grant</button>
                    </div>
                    {state.compensation.equity.length === 0 ? (
                      <div style={{ opacity: 0.7 }}>No grants yet.</div>
                    ) : (
                      state.compensation.equity.map((g, gIdx) => (
                        <div key={`${state.id}-grant-${gIdx}`} style={{ ...gridThree, marginTop: 6 }}>
                          <LabeledInput label="Type" value={g.type} onChange={(v) => updateEquity(idx, gIdx, { type: v })} />
                          <LabeledInput label="Grant value" value={g.grant_value} onChange={(v) => updateEquity(idx, gIdx, { grant_value: numberVal(v, g.grant_value) })} />
                          <LabeledInput label="Vesting years" value={g.vest_years} onChange={(v) => updateEquity(idx, gIdx, { vest_years: numberVal(v, g.vest_years) })} step="0.25" />
                          <LabeledInput label="Cliff months" value={g.cliff_months || 0} onChange={(v) => updateEquity(idx, gIdx, { cliff_months: numberVal(v, g.cliff_months || 0) })} />
                          <button onClick={() => removeEquity(idx, gIdx)} style={{ ...ghostButton, height: 36, alignSelf: "flex-end" }}>Remove</button>
                        </div>
                      ))
                    )}
                  </div>
                </>
              ) : (
                <div style={{ opacity: 0.7 }}>No salary/bonus for this state.</div>
              )}

              <button onClick={() => removeState(idx)} style={{ ...ghostButton, marginTop: 10 }}>Remove state</button>
            </div>
          ))}
          <button onClick={addState} style={ctaButton}>Add career state</button>
        </div>
      </Section>

      <Section title="Transitions">
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {config.transitions.map((t, idx) => (
            <div key={t.id || idx} style={itemCard}>
              <div style={gridThree}>
                <LabeledInput label="Id" value={t.id} onChange={(v) => updateTransition(idx, { id: v })} />
                <SelectInput label="From" value={t.from_state_id} onChange={(v) => updateTransition(idx, { from_state_id: v })} options={config.career_states.map((s) => ({ label: s.label, value: s.id }))} />
                <SelectInput label="To" value={t.to_state_id} onChange={(v) => updateTransition(idx, { to_state_id: v })} options={config.career_states.map((s) => ({ label: s.label, value: s.id }))} />
                <LabeledInput label="Type" value={t.type} onChange={(v) => updateTransition(idx, { type: v })} />
                <LabeledInput label="Base annual prob" value={t.base_annual_prob} onChange={(v) => updateTransition(idx, { base_annual_prob: numberVal(v, t.base_annual_prob) })} step="0.01" />
                <LabeledInput label="Desire multiplier" value={t.desire_multiplier || 1} onChange={(v) => updateTransition(idx, { desire_multiplier: numberVal(v, t.desire_multiplier || 1) })} step="0.1" />
                <LabeledInput label="Lag (months)" value={t.lag_months || 0} onChange={(v) => updateTransition(idx, { lag_months: numberVal(v, 0) })} />
                <LabeledInput label="Relocation cost" value={t.delta?.relocation_cost ?? ""} onChange={(v) => updateTransitionDelta(idx, { relocation_cost: v === "" ? null : numberVal(v, 0) })} />
                <LabeledInput label="Comp adj % (delta)" value={t.delta?.comp_adjustment_pct ?? ""} onChange={(v) => updateTransitionDelta(idx, { comp_adjustment_pct: v === "" ? null : numberVal(v, 0) })} step="0.01" />
              </div>
              <button onClick={() => removeTransition(idx)} style={ghostButton}>Remove transition</button>
            </div>
          ))}
          <button onClick={addTransition} style={ctaButton}>Add transition</button>
        </div>
      </Section>

      <Section title="Strategies">
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {config.strategies.map((s, idx) => (
            <div key={s.id || idx} style={itemCard}>
              <div style={gridThree}>
                <LabeledInput label="Id" value={s.id} onChange={(v) => updateStrategy(idx, { id: v })} />
                <LabeledInput label="Name" value={s.name} onChange={(v) => updateStrategy(idx, { name: v })} />
                <LabeledInput label="Paycut floor %" value={s.paycut_floor_pct ?? ""} onChange={(v) => updateStrategy(idx, { paycut_floor_pct: v === "" ? null : numberVal(v, -0.2) })} step="0.01" />
              </div>
              <LabeledInput label="Description" value={s.description} onChange={(v) => updateStrategy(idx, { description: v })} />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 8 }}>
                <div>
                  <div style={labelText}>Start in state(s)</div>
                  <div style={pillRow}>
                    {config.career_states.map((st) => (
                      <label key={`${s.id}-${st.id}`} style={pillCheckbox}>
                        <input
                          type="checkbox"
                          checked={s.initial_choice_state_ids.includes(st.id)}
                          onChange={() => toggleInitialChoice(idx, st.id)}
                        />
                        <span>{st.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <div>
                  <div style={labelText}>Preferred locations</div>
                  <div style={pillRow}>
                    {config.locations.map((loc) => (
                      <label key={`${s.id}-${loc.id}-pref`} style={pillCheckbox}>
                        <input
                          type="checkbox"
                          checked={s.preferred_locations.includes(loc.id)}
                          onChange={() => toggleLocationPref(idx, "preferred_locations", loc.id)}
                        />
                        <span>{loc.name}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <div>
                  <div style={labelText}>Disallowed locations</div>
                  <div style={pillRow}>
                    {config.locations.map((loc) => (
                      <label key={`${s.id}-${loc.id}-dis`} style={pillCheckbox}>
                        <input
                          type="checkbox"
                          checked={s.disallowed_locations.includes(loc.id)}
                          onChange={() => toggleLocationPref(idx, "disallowed_locations", loc.id)}
                        />
                        <span>{loc.name}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
          <button onClick={addStrategy} style={ctaButton}>Add strategy</button>
        </div>
      </Section>

      <Section title="Simulation settings">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 8 }}>
          <LabeledInput label="Time step (months)" value={config.simulation_settings.time_step_months} onChange={(v) => setSimulation({ time_step_months: numberVal(v, 1) })} />
          <LabeledInput label="Horizon short (years)" value={config.simulation_settings.horizon_years_short} onChange={(v) => setSimulation({ horizon_years_short: numberVal(v, 5) })} />
          <LabeledInput label="Horizon long (years)" value={config.simulation_settings.horizon_years_long} onChange={(v) => setSimulation({ horizon_years_long: numberVal(v, 10) })} />
          <LabeledInput label="Discount rate (real)" value={config.simulation_settings.discount_rate_real} onChange={(v) => setSimulation({ discount_rate_real: numberVal(v, 0.02) })} step="0.005" />
          <LabeledInput label="Risk lambda" value={config.simulation_settings.risk_penalty_lambda} onChange={(v) => setSimulation({ risk_penalty_lambda: numberVal(v, 0.5) })} step="0.05" />
          <LabeledInput label="CVaR alpha" value={config.simulation_settings.cvar_alpha} onChange={(v) => setSimulation({ cvar_alpha: numberVal(v, 0.1) })} step="0.01" />
          <LabeledInput label="Runs per scenario" value={config.simulation_settings.num_runs_per_scenario} onChange={(v) => setSimulation({ num_runs_per_scenario: numberVal(v, 500) })} />
          <LabeledInput label="Random seed" value={config.simulation_settings.random_seed ?? ""} onChange={(v) => setSimulation({ random_seed: v === "" ? null : numberVal(v, 7) })} />
        </div>
      </Section>
      </div>
    </div>
  );
};

/**
 * Section is a thin presentational wrapper that gives each logical chunk (locations, states, etc.)
 * consistent framing. By keeping it here rather than in a shared design system we avoid external
 * dependencies and can quickly tweak spacing or typography across the builder in one place.
 */
const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div style={sectionBox}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
      <h3 style={{ margin: 0, fontSize: 16 }}>{title}</h3>
    </div>
    {children}
  </div>
);

/**
 * LabeledInput standardizes labeled text fields across the builder. It accepts any string/number
 * value and pushes edits upward immediately, leaving validation to the caller. Sharing this small
 * component keeps spacing, typography, and focus styles uniform so the dense form stays legible.
 */
const LabeledInput: React.FC<{
  label: string;
  value: string | number;
  onChange: (val: string) => void;
  step?: string;
}> = ({ label, value, onChange, step }) => (
  <label style={fieldLabel}>
    <span style={labelText}>{label}</span>
    <input
      style={inputStyle}
      type="text"
      step={step}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  </label>
);

/**
 * SelectInput mirrors LabeledInput but wraps a native select element. It expects a list of label/
 * value pairs and surfaces changes through the provided callback, letting the parent handle updates.
 * Centralizing select styling ensures it blends visually with adjacent text inputs inside grid rows.
 */
const SelectInput: React.FC<{
  label: string;
  value: string;
  onChange: (val: string) => void;
  options: { label: string; value: string }[];
}> = ({ label, value, onChange, options }) => (
  <label style={fieldLabel}>
    <span style={labelText}>{label}</span>
    <select style={inputStyle} value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  </label>
);

/**
 * StatPill displays lightweight metrics at the top of the builder (counts of states, locations,
 * transitions, and strategies). It intentionally stays dumb: it renders whatever number it receives
 * without formatting so the caller can decide how to aggregate or localize values.
 */
const StatPill: React.FC<{ label: string; value: number }> = ({ label, value }) => (
  <div style={pill}>
    <div style={{ fontSize: 12, opacity: 0.7 }}>{label}</div>
    <div style={{ fontWeight: 700, fontSize: 20 }}>{value}</div>
  </div>
);

const sectionBox: React.CSSProperties = {
  background: "#0b1220",
  border: "1px solid #1f2937",
  borderRadius: 10,
  padding: 12,
  marginBottom: 12,
};

const itemCard: React.CSSProperties = {
  border: "1px solid #1f2937",
  borderRadius: 10,
  padding: 10,
  background: "#0f172a",
};

const fieldLabel: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  fontSize: 13,
};

const labelText: React.CSSProperties = { opacity: 0.7, fontSize: 12 };
const inputStyle: React.CSSProperties = {
  background: "#0b1220",
  border: "1px solid #1f2937",
  color: "#e2e8f0",
  borderRadius: 8,
  padding: "8px 10px",
};

const gridThree: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))",
  gap: 8,
};

const gridTwo: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))",
  gap: 8,
};

const pill: React.CSSProperties = {
  background: "#0f172a",
  border: "1px solid #1f2937",
  borderRadius: 10,
  padding: "10px 12px",
};

const pillRow: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 8,
  marginTop: 6,
};

const pillCheckbox: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  padding: "6px 8px",
  borderRadius: 8,
  border: "1px solid #1f2937",
};

const ghostButton: React.CSSProperties = {
  background: "transparent",
  color: "#e2e8f0",
  border: "1px solid #334155",
  borderRadius: 8,
  padding: "6px 10px",
  cursor: "pointer",
};

const ctaButton: React.CSSProperties = {
  background: "#38bdf8",
  border: "none",
  color: "#0b1220",
  padding: "10px 14px",
  borderRadius: 8,
  cursor: "pointer",
  fontWeight: 700,
};

const root: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  height: "100%",
};

const scrollArea: React.CSSProperties = {
  flex: 1,
  overflowY: "auto",
  paddingRight: 4,
  display: "flex",
  flexDirection: "column",
  gap: 12,
};

export default ConfigBuilder;
