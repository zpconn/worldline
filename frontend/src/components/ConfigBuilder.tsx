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

const ConfigBuilder: React.FC<Props> = ({ config, onChange }) => {
  const updateConfig = <K extends keyof ConfigPayload>(key: K, value: ConfigPayload[K]) => {
    onChange({ ...config, [key]: value });
  };

  const numberVal = (val: string, fallback = 0): number => {
    const n = Number(val);
    return Number.isFinite(n) ? n : fallback;
  };

  const setPortfolio = (partial: Partial<PortfolioSettings>) => {
    updateConfig("portfolio_settings", { ...config.portfolio_settings, ...partial });
  };

  const setScoring = (partial: Partial<ConfigPayload["scoring_weights"]>) => {
    updateConfig("scoring_weights", { ...config.scoring_weights, ...partial });
  };

  const setSimulation = (partial: Partial<ConfigPayload["simulation_settings"]>) => {
    updateConfig("simulation_settings", { ...config.simulation_settings, ...partial });
  };

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

  const updateLocation = (idx: number, partial: Partial<ConfigPayload["locations"][number]>) => {
    const next = config.locations.map((loc, i) => (i === idx ? { ...loc, ...partial } : loc));
    updateConfig("locations", next);
  };

  const removeLocation = (idx: number) => {
    const next = config.locations.filter((_, i) => i !== idx);
    updateConfig("locations", next);
  };

  const defaultCompensation = (): Compensation => ({
    base_annual: 150000,
    bonus_target_annual: 30000,
    bonus_prob_pay: 0.7,
    equity: [],
    one_times: [],
  });

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

  const updateState = (idx: number, partial: Partial<CareerState>) => {
    const next = config.career_states.map((s, i) => (i === idx ? { ...s, ...partial } : s));
    updateConfig("career_states", next);
  };

  const removeState = (idx: number) => {
    const next = config.career_states.filter((_, i) => i !== idx);
    updateConfig("career_states", next);
  };

  const toggleCompensation = (idx: number) => {
    const state = config.career_states[idx];
    updateState(idx, { compensation: state.compensation ? null : defaultCompensation() });
  };

  const updateCompensation = (idx: number, partial: Partial<Compensation>) => {
    const state = config.career_states[idx];
    if (!state.compensation) return;
    const next = { ...state.compensation, ...partial };
    updateState(idx, { compensation: next });
  };

  const addEquity = (stateIdx: number) => {
    const state = config.career_states[stateIdx];
    if (!state.compensation) return;
    const grants: EquityGrant[] = [
      ...state.compensation.equity,
      { type: "RSU", grant_value: 100000, vest_years: 4, cliff_months: 12 },
    ];
    updateCompensation(stateIdx, { equity: grants });
  };

  const updateEquity = (stateIdx: number, grantIdx: number, partial: Partial<EquityGrant>) => {
    const state = config.career_states[stateIdx];
    if (!state.compensation) return;
    const grants = state.compensation.equity.map((g, i) => (i === grantIdx ? { ...g, ...partial } : g));
    updateCompensation(stateIdx, { equity: grants });
  };

  const removeEquity = (stateIdx: number, grantIdx: number) => {
    const state = config.career_states[stateIdx];
    if (!state.compensation) return;
    const grants = state.compensation.equity.filter((_, i) => i !== grantIdx);
    updateCompensation(stateIdx, { equity: grants });
  };

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

  const updateTransition = (idx: number, partial: Partial<Transition>) => {
    const next = config.transitions.map((t, i) => (i === idx ? { ...t, ...partial } : t));
    updateConfig("transitions", next);
  };

  const updateTransitionDelta = (idx: number, partial: Partial<NonNullable<Transition["delta"]>>) => {
    const t = config.transitions[idx];
    updateTransition(idx, { delta: { ...(t.delta || {}), ...partial } });
  };

  const removeTransition = (idx: number) => {
    const next = config.transitions.filter((_, i) => i !== idx);
    updateConfig("transitions", next);
  };

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

  const updateStrategy = (idx: number, partial: Partial<Strategy>) => {
    const next = config.strategies.map((s, i) => (i === idx ? { ...s, ...partial } : s));
    updateConfig("strategies", next);
  };

  const toggleArrayValue = (arr: string[], value: string): string[] => {
    if (!value) return arr;
    return arr.includes(value) ? arr.filter((v) => v !== value) : [...arr, value];
  };

  const toggleInitialChoice = (idx: number, stateId: string) => {
    const strategy = config.strategies[idx];
    updateStrategy(idx, { initial_choice_state_ids: toggleArrayValue(strategy.initial_choice_state_ids, stateId) });
  };

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

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div style={sectionBox}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
      <h3 style={{ margin: 0, fontSize: 16 }}>{title}</h3>
    </div>
    {children}
  </div>
);

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
