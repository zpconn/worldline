import { CSSProperties, useEffect, useState } from "react";
import axios from "axios";
import ConfigEditor from "./components/ConfigEditor";
import SimulationResults from "./components/SimulationResults";
import DagView from "./components/DagView";
import ConfigBuilder from "./components/ConfigBuilder";
import { ConfigPayload } from "./types";

const API_BASE = "http://localhost:8000";

const sampleConfig: ConfigPayload = {
  locations: [
    { id: "home", name: "Home City", col_annual: 60000, state_tax_rate: 0.05 },
    { id: "hub", name: "Tech Hub", col_annual: 90000, state_tax_rate: 0.1 },
  ],
  portfolio_settings: {
    initial_liquid: 250000,
    mean_annual_return: 0.06,
    std_annual_return: 0.12,
    contribution_rate: 0.1,
  },
  career_states: [
    {
      id: "current",
      label: "Current Role",
      role_title: "Engineering Manager",
      location_id: "home",
      employment_status: "employed",
      compensation: {
        base_annual: 200000,
        bonus_target_annual: 50000,
        bonus_prob_pay: 0.7,
        equity: [],
        one_times: [],
      },
      wellbeing: 0.6,
      identity_brand: { external_portability: 0.6, internal_stature: 0.6 },
    },
    {
      id: "promotion",
      label: "Director",
      role_title: "Director of Engineering",
      location_id: "home",
      employment_status: "employed",
      compensation: {
        base_annual: 260000,
        bonus_target_annual: 70000,
        bonus_prob_pay: 0.7,
        equity: [],
        one_times: [],
      },
      wellbeing: 0.65,
      identity_brand: { external_portability: 0.7, internal_stature: 0.75 },
    },
    {
      id: "startup",
      label: "Startup CTO",
      role_title: "CTO",
      location_id: "hub",
      employment_status: "employed",
      compensation: {
        base_annual: 180000,
        bonus_target_annual: 30000,
        bonus_prob_pay: 0.4,
        equity: [{ type: "RSU", grant_value: 200000, vest_years: 4, cliff_months: 12 }],
        one_times: [],
      },
      wellbeing: 0.75,
      identity_brand: { external_portability: 0.8, internal_stature: 0.7 },
    },
    {
      id: "unemployed",
      label: "Unemployment",
      role_title: "Unemployed",
      location_id: "home",
      employment_status: "unemployed",
      compensation: null,
      wellbeing: 0.35,
      identity_brand: { external_portability: 0.45, internal_stature: 0.3 },
    },
  ],
  transitions: [
    {
      id: "t1",
      from_state_id: "current",
      to_state_id: "promotion",
      type: "promotion",
      base_annual_prob: 0.25,
      desire_multiplier: 1,
      lag_months: 0,
    },
    {
      id: "t2",
      from_state_id: "current",
      to_state_id: "startup",
      type: "external_switch",
      base_annual_prob: 0.12,
      desire_multiplier: 1,
      lag_months: 1,
      delta: { relocation_cost: 15000 },
    },
    {
      id: "t3",
      from_state_id: "current",
      to_state_id: "unemployed",
      type: "layoff",
      base_annual_prob: 0.05,
      desire_multiplier: 1,
    },
    {
      id: "t4",
      from_state_id: "promotion",
      to_state_id: "unemployed",
      type: "layoff",
      base_annual_prob: 0.04,
      desire_multiplier: 1,
    },
    {
      id: "t5",
      from_state_id: "startup",
      to_state_id: "unemployed",
      type: "startup_failure",
      base_annual_prob: 0.18,
      desire_multiplier: 1,
    },
    {
      id: "t6",
      from_state_id: "unemployed",
      to_state_id: "current",
      type: "reentry",
      base_annual_prob: 0.35,
      desire_multiplier: 1,
    },
  ],
  strategies: [
    {
      id: "stability",
      name: "Stability",
      description: "Keep current role and optimize promotion odds",
      initial_choice_state_ids: ["current"],
      preferred_locations: ["home"],
      disallowed_locations: [],
      paycut_floor_pct: -0.2,
      rules: [],
    },
    {
      id: "upswing",
      name: "Upswing",
      description: "Bias toward startup growth moves",
      initial_choice_state_ids: ["current"],
      preferred_locations: ["hub"],
      disallowed_locations: [],
      paycut_floor_pct: -0.35,
      rules: [],
    },
  ],
  scoring_weights: {
    financial: 0.65,
    career_capital: 0.2,
    enjoyment_identity: 0.1,
    location_fit: 0.1,
    legacy: 0.05,
  },
  simulation_settings: {
    time_step_months: 1,
    horizon_years_short: 5,
    horizon_years_long: 10,
    discount_rate_real: 0.02,
    risk_penalty_lambda: 0.5,
    cvar_alpha: 0.1,
    num_runs_per_scenario: 500,
    random_seed: 7,
  },
};

function App() {
  const [config, setConfig] = useState<ConfigPayload>(sampleConfig);
  const [configText, setConfigText] = useState<string>(() => JSON.stringify(sampleConfig, null, 2));
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [saving, setSaving] = useState(false);
  const [simulating, setSimulating] = useState(false);

  useEffect(() => {
    setConfigText(JSON.stringify(config, null, 2));
  }, [config]);

  const applyJsonToBuilder = () => {
    try {
      const parsed = JSON.parse(configText) as ConfigPayload;
      setConfig(parsed);
      setJsonError(null);
      setConfigText(JSON.stringify(parsed, null, 2));
    } catch (err: unknown) {
      setJsonError(err instanceof Error ? err.message : "Unable to parse JSON");
    }
  };

  const saveConfig = async (silent = false) => {
    if (!silent) setSaving(true);
    try {
      await axios.post(`${API_BASE}/config`, config);
    } finally {
      if (!silent) setSaving(false);
    }
  };

  const runSimulation = async () => {
    setSimulating(true);
    try {
      await saveConfig(true);
      const res = await axios.post(`${API_BASE}/simulate`);
      setResult(res.data);
    } finally {
      setSimulating(false);
    }
  };

  const exportJson = async () => {
    const res = await axios.get(`${API_BASE}/export`);
    setResult(res.data);
    const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "simulation-result.json";
    link.click();
  };

  useEffect(() => {
    void saveConfig(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div style={{ fontFamily: "Space Grotesk, 'Segoe UI', sans-serif", background: "linear-gradient(135deg,#0f172a,#1e293b)", minHeight: "100vh", color: "#e2e8f0" }}>
      <header style={{ padding: "20px 32px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 28 }}>Worldline</h1>
          <p style={{ margin: 0, opacity: 0.7 }}>Model your path as a DAG, run Monte Carlo, and compare strategies.</p>
        </div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <button onClick={() => setConfig(sampleConfig)} style={secondaryButtonStyle}>Reset sample</button>
          <button onClick={() => saveConfig()} disabled={saving} style={primaryButtonStyle}>
            {saving ? "Saving..." : "Save Config"}
          </button>
          <button onClick={runSimulation} disabled={simulating} style={accentButtonStyle}>
            {simulating ? "Simulating..." : "Run Simulations"}
          </button>
          <button onClick={exportJson} style={secondaryButtonStyle}>Export JSON</button>
        </div>
      </header>

      <main style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 16, padding: "0 20px 32px" }}>
        <section style={tallCardStyle}>
          <h2 style={sectionTitle}>Interactive Config Builder</h2>
          <ConfigBuilder config={config} onChange={setConfig} />
        </section>
        <section style={tallCardStyle}>
          <h2 style={sectionTitle}>JSON Preview</h2>
          <ConfigEditor
            value={configText}
            onChange={setConfigText}
            onApply={applyJsonToBuilder}
            parseError={jsonError}
            helperText="Tweak JSON directly, then apply it back into the builder."
          />
          <div style={{ marginTop: 10, fontSize: 13, opacity: 0.75 }}>
            Save pushes the current builder/JSON state to the backend. Run Simulations auto-saves first.
          </div>
        </section>
        <section style={{ ...cardStyle, gridColumn: "1 / span 2" }}>
          <h2 style={sectionTitle}>Worldline DAG</h2>
          <DagView config={config} />
        </section>
        <section style={{ ...cardStyle, gridColumn: "1 / span 2" }}>
          <h2 style={sectionTitle}>Simulation & Results</h2>
          <SimulationResults result={result} />
        </section>
        <section style={{ ...cardStyle, gridColumn: "1 / span 2" }}>
          <h2 style={sectionTitle}>How to Run & Glossary</h2>
          <Instructions />
        </section>
      </main>
    </div>
  );
}

const cardStyle: CSSProperties = {
  background: "#0b1220",
  border: "1px solid #1f2937",
  borderRadius: 12,
  padding: 16,
  boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
};

const tallCardStyle: CSSProperties = {
  ...cardStyle,
  height: "72vh",
  minHeight: 520,
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
};

const sectionTitle: CSSProperties = {
  margin: "0 0 12px 0",
  fontSize: 18,
  letterSpacing: 0.2,
};

const primaryButtonStyle: CSSProperties = {
  background: "#38bdf8",
  border: "none",
  color: "#0b1220",
  padding: "10px 16px",
  borderRadius: 8,
  cursor: "pointer",
  fontWeight: 700,
};

const accentButtonStyle: CSSProperties = {
  ...primaryButtonStyle,
  background: "#a855f7",
  color: "#0b1220",
};

const secondaryButtonStyle: CSSProperties = {
  ...primaryButtonStyle,
  background: "transparent",
  color: "#e2e8f0",
  border: "1px solid #334155",
};

export default App;

const Instructions = () => (
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
    <div>
      <h3 style={{ marginTop: 0, marginBottom: 6 }}>Quick run steps (local)</h3>
      <ol style={{ margin: 0, paddingLeft: 20, lineHeight: 1.5 }}>
        <li>Start backend: `cd backend && .\\.venv\\Scripts\\activate` then `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`.</li>
        <li>Start frontend: `cd frontend && npm install && npm run dev` (open http://localhost:5173).</li>
        <li>Use the builder or JSON, click Save Config (or Run Simulations to auto-save) to send it to the API.</li>
        <li>Use Export JSON to download the last simulation result.</li>
      </ol>
    </div>
    <div>
      <h3 style={{ marginTop: 0, marginBottom: 6 }}>Glossary</h3>
      <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.5 }}>
        <li><strong>DAG</strong>: Directed acyclic graph of career states and transitions.</li>
        <li><strong>Strategy</strong>: Policy that filters/weights transitions and defines starting choices.</li>
        <li><strong>Scenario</strong>: A (strategy + initial_state) combo evaluated over a horizon.</li>
        <li><strong>Utility</strong>: EV minus risk penalty plus weighted non-financial scores.</li>
        <li><strong>EV</strong>: Expected value (mean NPV across simulations).</li>
        <li><strong>CVaR</strong>: Conditional value at risk, average of the worst tail outcomes.</li>
        <li><strong>Risk lambda</strong>: Weight on variance; higher = more risk-averse.</li>
        <li><strong>cvar alpha</strong>: Tail fraction for CVaR (e.g., 0.10 = worst 10%).</li>
        <li><strong>COL</strong>: Cost of living for a location.</li>
        <li><strong>Paycut floor</strong>: Minimum allowed compensation change on a move.</li>
        <li><strong>Downside liquidity</strong>: Checks like P(liquid &lt; 1x or 2x COL).</li>
        <li><strong>Tornado</strong>: Sensitivity chart showing utility change when perturbing key inputs.</li>
      </ul>
    </div>
  </div>
);
