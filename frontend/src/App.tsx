import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import ConfigEditor from "./components/ConfigEditor";
import SimulationResults from "./components/SimulationResults";
import DagView from "./components/DagView";

const API_BASE = "http://localhost:8000";

const sampleConfig = {
  locations: [
    { id: "home", name: "Home City", col_annual: 60000, state_tax_rate: 0.05 },
    { id: "hub", name: "Tech Hub", col_annual: 90000, state_tax_rate: 0.1 }
  ],
  portfolio_settings: {
    initial_liquid: 250000,
    mean_annual_return: 0.06,
    std_annual_return: 0.12,
    contribution_rate: 0.1
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
        one_times: []
      },
      wellbeing: 0.6,
      identity_brand: { external_portability: 0.6, internal_stature: 0.6 }
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
        one_times: []
      },
      wellbeing: 0.65,
      identity_brand: { external_portability: 0.7, internal_stature: 0.75 }
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
        one_times: []
      },
      wellbeing: 0.75,
      identity_brand: { external_portability: 0.8, internal_stature: 0.7 }
    },
    {
      id: "unemployed",
      label: "Unemployment",
      role_title: "Unemployed",
      location_id: "home",
      employment_status: "unemployed",
      compensation: null,
      wellbeing: 0.35,
      identity_brand: { external_portability: 0.45, internal_stature: 0.3 }
    }
  ],
  transitions: [
    {
      id: "t1",
      from_state_id: "current",
      to_state_id: "promotion",
      type: "promotion",
      base_annual_prob: 0.25,
      desire_multiplier: 1,
      lag_months: 0
    },
    {
      id: "t2",
      from_state_id: "current",
      to_state_id: "startup",
      type: "external_switch",
      base_annual_prob: 0.12,
      desire_multiplier: 1,
      lag_months: 1,
      delta: { relocation_cost: 15000 }
    },
    {
      id: "t3",
      from_state_id: "current",
      to_state_id: "unemployed",
      type: "layoff",
      base_annual_prob: 0.05,
      desire_multiplier: 1
    },
    {
      id: "t4",
      from_state_id: "promotion",
      to_state_id: "unemployed",
      type: "layoff",
      base_annual_prob: 0.04,
      desire_multiplier: 1
    },
    {
      id: "t5",
      from_state_id: "startup",
      to_state_id: "unemployed",
      type: "startup_failure",
      base_annual_prob: 0.18,
      desire_multiplier: 1
    },
    {
      id: "t6",
      from_state_id: "unemployed",
      to_state_id: "current",
      type: "reentry",
      base_annual_prob: 0.35,
      desire_multiplier: 1
    }
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
      rules: []
    },
    {
      id: "upswing",
      name: "Upswing",
      description: "Bias toward startup growth moves",
      initial_choice_state_ids: ["current"],
      preferred_locations: ["hub"],
      disallowed_locations: [],
      paycut_floor_pct: -0.35,
      rules: []
    }
  ],
  scoring_weights: {
    financial: 0.65,
    career_capital: 0.2,
    enjoyment_identity: 0.1,
    location_fit: 0.1,
    legacy: 0.05
  },
  simulation_settings: {
    time_step_months: 1,
    horizon_years_short: 5,
    horizon_years_long: 10,
    discount_rate_real: 0.02,
    risk_penalty_lambda: 0.5,
    cvar_alpha: 0.1,
    num_runs_per_scenario: 500,
    random_seed: 7
  }
};

function App() {
  const [configText, setConfigText] = useState(JSON.stringify(sampleConfig, null, 2));
  const [result, setResult] = useState<any>(null);
  const [saving, setSaving] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const configObj = useMemo(() => {
    try {
      return JSON.parse(configText);
    } catch {
      return null;
    }
  }, [configText]);

  const saveConfig = async () => {
    if (!configObj) return;
    setSaving(true);
    try {
      await axios.post(`${API_BASE}/config`, configObj);
    } finally {
      setSaving(false);
    }
  };

  const runSimulation = async () => {
    setSimulating(true);
    try {
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
    // eagerly push sample config for first run
    saveConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div style={{ fontFamily: "Space Grotesk, 'Segoe UI', sans-serif", background: "linear-gradient(135deg,#0f172a,#1e293b)", minHeight: "100vh", color: "#e2e8f0" }}>
      <header style={{ padding: "20px 32px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 28 }}>Career-as-DAG Monte Carlo</h1>
          <p style={{ margin: 0, opacity: 0.7 }}>Model transitions, run Monte Carlo, and compare strategies.</p>
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <button onClick={saveConfig} disabled={!configObj || saving} style={primaryButtonStyle}>
            {saving ? "Saving..." : "Save Config"}
          </button>
          <button onClick={runSimulation} disabled={simulating} style={accentButtonStyle}>
            {simulating ? "Simulating..." : "Run Simulations"}
          </button>
          <button onClick={exportJson} style={secondaryButtonStyle}>Export JSON</button>
        </div>
      </header>

      <main style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, padding: "0 20px 32px" }}>
        <section style={cardStyle}>
          <h2 style={sectionTitle}>Configuration</h2>
          <ConfigEditor value={configText} onChange={setConfigText} />
        </section>
        <section style={cardStyle}>
          <h2 style={sectionTitle}>Career DAG</h2>
          <DagView config={configObj} />
        </section>
        <section style={{ ...cardStyle, gridColumn: "1 / span 2" }}>
          <h2 style={sectionTitle}>Simulation & Results</h2>
          <SimulationResults result={result} />
        </section>
      </main>
    </div>
  );
}

const cardStyle: React.CSSProperties = {
  background: "#0b1220",
  border: "1px solid #1f2937",
  borderRadius: 12,
  padding: 16,
  boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
};

const sectionTitle: React.CSSProperties = {
  margin: "0 0 12px 0",
  fontSize: 18,
  letterSpacing: 0.2,
};

const primaryButtonStyle: React.CSSProperties = {
  background: "#38bdf8",
  border: "none",
  color: "#0b1220",
  padding: "10px 16px",
  borderRadius: 8,
  cursor: "pointer",
  fontWeight: 700,
};

const accentButtonStyle: React.CSSProperties = {
  ...primaryButtonStyle,
  background: "#a855f7",
  color: "#0b1220",
};

const secondaryButtonStyle: React.CSSProperties = {
  ...primaryButtonStyle,
  background: "transparent",
  color: "#e2e8f0",
  border: "1px solid #334155",
};

export default App;
