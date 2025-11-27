// Panels and charts to display simulation outputs (best scenarios, downside, sensitivity).
import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from "recharts";

type Props = {
  result: any;
  simulating?: boolean;
};

type BarDatum = {
  name: string;
  EV: number;
  CVaR: number;
  Utility: number;
};

/**
 * SimulationResults renders every visualization tied to a simulation run: headline winners, bar
 * charts for utility/EV/CVaR, downside tables, sensitivity tornado, and a scrolling list of all
 * scenarios. It is intentionally stateless, deriving every chart input from the `result` prop so
 * that rerunning simulations or swapping configs instantly rehydrates the UI. A lightweight
 * `simulating` flag overlays an animated progress panel to keep users informed while waiting.
 */
const SimulationResults: React.FC<Props> = ({ result, simulating = false }) => {
  const showProgress = simulating;

  if (!result) {
    return (
      <div style={{ position: "relative" }}>
        {showProgress && <LiveProgress />}
        <div style={{ opacity: 0.6 }}>Run simulations to see metrics.</div>
      </div>
    );
  }

  /**
   * Slice and reshape the raw result into chart-friendly structures. We separate 10-year and 5-year
   * scenarios to avoid mixing horizons, then build bar data keyed by label so Recharts can map each
   * metric to a stacked column. Domain calculations pad both sides to keep bars away from axes, and
   * tick generation uses a "nice number" strategy so grid lines fall on human-friendly increments
   * regardless of the absolute magnitude of the simulation outputs.
   */
  const scenarios10 = (result.all_scenarios || []).filter((s: any) => s.horizon_years === 10);
  const scenarios5 = (result.all_scenarios || []).filter((s: any) => s.horizon_years === 5);
  const barData: BarDatum[] = scenarios10.map((s: any) => ({
    name: s.label,
    EV: s.ev_npv,
    CVaR: s.cvar_npv,
    Utility: s.utility_score,
  }));
  const leftVals = barData.flatMap((d: BarDatum) => [d.EV || 0, d.CVaR || 0]);
  const rightVals = barData.map((d: BarDatum) => d.Utility || 0);
  const leftDomain = paddedDomain(leftVals);
  const rightDomain = paddedDomain(rightVals);
  const leftTicks = buildTicks(leftDomain);
  const rightTicks = buildTicks(rightDomain);

  const tornado = (result.sensitivity?.parameters || [])
    .map((p: any) => {
      const base = p.base_utility ?? 0;
      const low = p.low_utility ?? 0;
      const high = p.high_utility ?? 0;
      return {
        name: p.parameter,
        low,
        base,
        high,
        lowDelta: low - base,
        highDelta: high - base,
      };
    })
    .sort((a: any, b: any) => Math.max(Math.abs(b.lowDelta), Math.abs(b.highDelta)) - Math.max(Math.abs(a.lowDelta), Math.abs(a.highDelta)));
  const tornadoDomain = paddedSymmetricDomain(tornado.flatMap((d: any) => [d.lowDelta, d.highDelta]));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 16, position: "relative" }}>
      {showProgress && <LiveProgress />}
      <div>
        <div style={pillRow}>
          <SummaryCard title="Best 5Y" scenario={result.best_5y} />
          <SummaryCard title="Best 10Y" scenario={result.best_10y} />
        </div>
        <h3 style={subhead}>Utility (10y)</h3>
        <div style={{ height: 220, background: "#0f172a", borderRadius: 10, border: "1px solid #1f2937", padding: 8 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ top: 16, right: 8, bottom: 16, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="name" hide />
              <YAxis domain={rightDomain} ticks={rightTicks} tickFormatter={formatCompact} tick={{ fill: "#94a3b8" }} />
              <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 4" />
              <Tooltip formatter={(value) => formatCompact(value as number)} />
              <Legend />
              <Bar dataKey="Utility" fill="#38bdf8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <h3 style={subhead}>EV / CVaR (10y)</h3>
        <div style={{ height: 220, background: "#0f172a", borderRadius: 10, border: "1px solid #1f2937", padding: 8 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ top: 16, right: 8, bottom: 16, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="name" hide />
              <YAxis domain={leftDomain} ticks={leftTicks} tickFormatter={formatCompact} tick={{ fill: "#94a3b8" }} />
              <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 4" />
              <Tooltip formatter={(value) => formatCompact(value as number)} />
              <Legend />
              <Bar dataKey="EV" fill="#a855f7" />
              <Bar dataKey="CVaR" fill="#f97316" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <h3 style={subhead}>Downside Dashboard (by strategy)</h3>
        <DownsideTable data={result.downside_dashboard?.by_strategy || {}} />
      </div>
      <div>
        <h3 style={subhead}>Sensitivity (tornado)</h3>
        <div style={{ height: 220, background: "#0f172a", borderRadius: 10, border: "1px solid #1f2937", padding: 8 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={tornado} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <YAxis dataKey="name" type="category" width={70} />
              <XAxis type="number" domain={tornadoDomain} tickFormatter={formatOneDecimal} />
              <ReferenceLine x={0} stroke="#475569" strokeDasharray="4 4" />
              <Tooltip content={TornadoTooltip} />
              <Legend />
              <Bar dataKey="lowDelta" name="Low vs base" fill="#f97316" />
              <Bar dataKey="highDelta" name="High vs base" fill="#22c55e" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <h3 style={subhead}>All Scenarios (5y)</h3>
        <ScenarioList scenarios={scenarios5} />
      </div>
    </div>
  );
};

/**
 * SummaryCard distills a single scenario into a compact snapshot with title, label, and headline
 * metrics. It intentionally applies minimal formatting so numbers elsewhere in the dashboard render
 * identically, and it gracefully falls back to a placeholder card when no scenario is provided.
 */
const SummaryCard: React.FC<{ title: string; scenario: any }> = ({ title, scenario }) => {
  if (!scenario) return <div style={summaryCard}>No data</div>;
  return (
    <div style={summaryCard}>
      <p style={{ margin: 0, opacity: 0.7 }}>{title}</p>
      <h4 style={{ margin: "4px 0", fontSize: 18 }}>{scenario.label}</h4>
      <p style={{ margin: 0 }}>Utility {scenario.utility_score.toFixed(2)}</p>
      <p style={{ margin: 0, opacity: 0.7 }}>EV NPV {scenario.ev_npv.toFixed(0)}</p>
      <p style={{ margin: 0, opacity: 0.7 }}>CVaR {scenario.cvar_npv.toFixed(0)}</p>
    </div>
  );
};

/**
 * DownsideTable surfaces tail risk metrics for each strategy. It expects a record keyed by strategy
 * id and renders probabilities for liquidity shortfalls and unemployment durations, formatting them
 * as percentages for immediate readability. When the dataset is empty, it opts for a gentle message
 * rather than an empty grid, signaling that simulations have not produced downside metrics yet.
 */
const DownsideTable: React.FC<{ data: Record<string, any> }> = ({ data }) => {
  const entries = Object.entries(data);
  if (!entries.length) return <div style={{ opacity: 0.6 }}>No downside metrics yet.</div>;
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
      <thead>
        <tr>
          <th style={th}>Strategy</th>
          <th style={th}>P(liquid&lt;1x)</th>
          <th style={th}>P(liquid&lt;2x)</th>
          <th style={th}>Unemp &gt;=6m</th>
          <th style={th}>Unemp &gt;=12m</th>
          <th style={th}>Lower pay reentry</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([sid, metrics]) => (
          <tr key={sid}>
            <td style={td}>{sid}</td>
            <td style={td}>{pct(metrics.p_liquid_lt_1x_col)}</td>
            <td style={td}>{pct(metrics.p_liquid_lt_2x_col)}</td>
            <td style={td}>{pct(metrics.p_unemp_ge_6m)}</td>
            <td style={td}>{pct(metrics.p_unemp_ge_12m)}</td>
            <td style={td}>{pct(metrics.p_lower_pay_reentry)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

/**
 * ScenarioList provides a scrollable list of every scenario for a given horizon. It deliberately
 * keeps formatting terse--showing only label and headline metrics--so the panel can accommodate many
 * scenarios without overwhelming the user. The container caps its height and enables overflow to
 * avoid pushing other charts off-screen.
 */
const ScenarioList: React.FC<{ scenarios: any[] }> = ({ scenarios }) => {
  return (
    <div style={{ background: "#0f172a", borderRadius: 10, border: "1px solid #1f2937", maxHeight: 250, overflow: "auto" }}>
      {scenarios.map((s) => (
        <div key={`${s.strategy_id}-${s.initial_state_id}`} style={{ padding: 10, borderBottom: "1px solid #1f2937" }}>
          <div style={{ fontWeight: 700 }}>{s.label}</div>
          <div style={{ fontSize: 12, opacity: 0.8 }}>Utility {s.utility_score.toFixed(2)} | EV {s.ev_npv.toFixed(0)} | CVaR {s.cvar_npv.toFixed(0)}</div>
        </div>
      ))}
    </div>
  );
};

/**
 * pct converts a probability expressed as a decimal into a whole-number percentage string. It clamps
 * falsy values to zero so missing metrics do not propagate NaN into the UI, and rounds to the nearest
 * integer to keep the table compact and easy to scan.
 */
const pct = (v: number) => `${Math.round((v || 0) * 100)}%`;

/**
 * formatCompact renders large numbers with single-letter magnitude suffixes so axis labels stay
 * readable at different scales. The thresholds cover thousands, millions, and billions while falling
 * back to whole numbers for smaller values, keeping a consistent display vocabulary across charts.
 */
const formatCompact = (n: number) => {
  const abs = Math.abs(n);
  if (abs >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}b`;
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}m`;
  if (abs >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return n.toFixed(0);
};

/**
 * formatOneDecimal shows a single decimal place, used primarily for axis ticks on symmetric charts
 * like the tornado plot. Keeping precision fixed avoids jitter when domains change between runs.
 */
const formatOneDecimal = (n: number) => n.toFixed(1);

/**
 * formatTwoDecimal offers slightly more precision for tooltip content where space is less constrained
 * and users expect finer-grained utility deltas.
 */
const formatTwoDecimal = (n: number) => n.toFixed(2);

/**
 * formatDelta prefixes a sign and formats to two decimals, making it clear whether a value reflects
 * upside or downside relative to a base case in sensitivity outputs.
 */
const formatDelta = (n: number) => `${n >= 0 ? "+" : ""}${formatTwoDecimal(n)}`;

/**
 * paddedDomain computes a numeric range expanded by a padding ratio so chart elements never sit
 * directly on the axes. When the data is degenerate (all zeros or a single value), it enforces a
 * minimum span to keep Recharts happy and ensures both positive and negative space are represented.
 */
const paddedDomain = (vals: number[], padRatio = 0.08): [number, number] => {
  if (!vals.length) return [0, 1];
  const min = Math.min(...vals, 0);
  const max = Math.max(...vals, 0);
  const span = max - min;
  const pad = span === 0 ? Math.max(Math.abs(max || 1) * padRatio, 1) : span * padRatio;
  return [min - pad, max + pad];
};

/**
 * paddedSymmetricDomain creates a balanced range around zero for charts where negative and positive
 * deltas are equally important (e.g., tornado plots). It calculates the largest absolute magnitude
 * and adds proportional padding so bars do not hug the axis even when sensitivity is small.
 */
const paddedSymmetricDomain = (vals: number[], padRatio = 0.08): [number, number] => {
  const maxAbs = Math.max(...vals.map((v) => Math.abs(v)), 0);
  const pad = maxAbs === 0 ? 1 : maxAbs * padRatio;
  const bound = maxAbs + pad;
  return [-bound, bound];
};

/**
 * buildTicks generates a "nice" set of axis ticks given a numeric domain. It computes an initial step
 * size from the span, snaps that step to human-friendly intervals via `niceStep`, then anchors ticks
 * at multiples of the step while ensuring zero is always included. Safety caps prevent runaway loops
 * when domains are pathological.
 */
const buildTicks = ([min, max]: [number, number], target = 6): number[] => {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0];
  if (min === max) {
    const base = min || 1;
    return [base - 1, 0, base + 1].sort((a, b) => a - b);
  }
  const span = max - min;
  const rawStep = span / Math.max(1, target - 1);
  const step = niceStep(rawStep);
  let start = Math.floor(min / step) * step;
  let end = Math.ceil(max / step) * step;
  start = Math.min(start, 0);
  end = Math.max(end, 0);
  const ticks: number[] = [];
  for (let v = start; v <= end + 1e-9; v += step) {
    ticks.push(Number(v.toFixed(10)));
    if (ticks.length > 50) break; // safety cap
  }
  if (!ticks.includes(0)) ticks.push(0);
  ticks.sort((a, b) => a - b);
  return ticks;
};

/**
 * niceStep rounds a raw tick interval to a base-10 friendly value (1, 2, 5, or 10 times a power of
 * ten). This mirrors the "nice numbers" approach used in charting libraries to keep grid lines on
 * intuitive boundaries regardless of the magnitude of the data.
 */
const niceStep = (raw: number): number => {
  const exp = Math.floor(Math.log10(Math.max(raw, 1e-12)));
  const f = raw / Math.pow(10, exp);
  let nf: number;
  if (f < 1.5) nf = 1;
  else if (f < 3) nf = 2;
  else if (f < 7) nf = 5;
  else nf = 10;
  return nf * Math.pow(10, exp);
};

/**
 * TornadoTooltip customizes the sensitivity chart hover content. It spells out the base utility and
 * the absolute utilities at the low/high perturbations, pairing each with a signed delta so users
 * can quickly see both direction and magnitude of impact without decoding the bar lengths alone.
 */
const TornadoTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload;
  return (
    <div style={{ background: "#0b1220", border: "1px solid #1f2937", borderRadius: 8, padding: 8 }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 12, opacity: 0.8 }}>Base utility: {formatTwoDecimal(row.base)}</div>
      <div style={{ fontSize: 12, color: "#f97316" }}>
        Low: {formatTwoDecimal(row.low)} ({formatDelta(row.lowDelta)})
      </div>
      <div style={{ fontSize: 12, color: "#22c55e" }}>
        High: {formatTwoDecimal(row.high)} ({formatDelta(row.highDelta)})
      </div>
    </div>
  );
};

const subhead: React.CSSProperties = { marginBottom: 6, marginTop: 16 };
const pillRow: React.CSSProperties = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 };
const summaryCard: React.CSSProperties = {
  background: "#0f172a",
  border: "1px solid #1f2937",
  borderRadius: 10,
  padding: 10,
  minHeight: 110,
};
const th: React.CSSProperties = { textAlign: "left", padding: "6px 4px", borderBottom: "1px solid #1f2937" };
const td: React.CSSProperties = { padding: "6px 4px", borderBottom: "1px solid #1f2937" };

/**
 * LiveProgress overlays an animated status panel while simulations run. It uses pure CSS animation
 * (embedded in a style tag) to avoid pulling in additional dependencies, and it intentionally covers
 * the result grid with a blurred backdrop to signal that data is temporarily stale during the run.
 */
const LiveProgress: React.FC = () => (
  <>
    <style>{progressCss}</style>
    <div className="sim-progress-shell">
      <div className="sim-progress-content">
        <div className="sim-progress-header">
          <div className="sim-progress-pip" />
          <span>Simulation running</span>
        </div>
        <div className="sim-progress-sub">Monte Carlo sweeps in flight across all cores</div>
        <div className="sim-progress-bar">
          <div className="sim-progress-fill" />
          <div className="sim-progress-glow" />
        </div>
        <div className="sim-progress-spark one" />
        <div className="sim-progress-spark two" />
        <div className="sim-progress-spark three" />
      </div>
    </div>
  </>
);

const progressCss = `
.sim-progress-shell {
  position: absolute;
  inset: -8px;
  padding: 8px;
  border-radius: 14px;
  backdrop-filter: blur(6px);
  background: radial-gradient(circle at 20% 30%, rgba(56,189,248,0.08), transparent 45%), radial-gradient(circle at 80% 70%, rgba(168,85,247,0.08), transparent 45%);
  border: 1px solid rgba(148,163,184,0.2);
  z-index: 5;
}
.sim-progress-content {
  position: relative;
  background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(17,24,39,0.9));
  border: 1px solid rgba(148,163,184,0.25);
  border-radius: 10px;
  padding: 12px 14px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.35);
  overflow: hidden;
}
.sim-progress-header {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  letter-spacing: 0.3px;
}
.sim-progress-sub {
  font-size: 12px;
  opacity: 0.75;
  margin-top: 2px;
  margin-bottom: 8px;
}
.sim-progress-pip {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: linear-gradient(135deg, #38bdf8, #22c55e);
  box-shadow: 0 0 12px rgba(34,197,94,0.8), 0 0 20px rgba(56,189,248,0.6);
  animation: pulse 1.4s ease-in-out infinite;
}
.sim-progress-bar {
  position: relative;
  height: 12px;
  border-radius: 999px;
  background: rgba(148,163,184,0.15);
  overflow: hidden;
}
.sim-progress-fill {
  position: absolute;
  inset: 0;
  width: 60%;
  background: linear-gradient(90deg, #0ea5e9, #a855f7, #22c55e, #0ea5e9);
  background-size: 200% 100%;
  animation: wave 1.8s linear infinite;
  filter: drop-shadow(0 0 12px rgba(56,189,248,0.5));
}
.sim-progress-glow {
  position: absolute;
  inset: -18px -6px;
  background: radial-gradient(circle at 20% 50%, rgba(14,165,233,0.35), transparent 40%),
              radial-gradient(circle at 70% 50%, rgba(168,85,247,0.3), transparent 40%);
  filter: blur(20px);
  pointer-events: none;
}
.sim-progress-spark {
  position: absolute;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: rgba(56,189,248,0.9);
  filter: blur(1px);
  animation: float 1.8s ease-in-out infinite;
}
.sim-progress-spark.one { top: 8px; left: 20%; animation-delay: 0s; }
.sim-progress-spark.two { top: 10px; left: 55%; animation-delay: 0.25s; background: rgba(168,85,247,0.9); }
.sim-progress-spark.three { top: 6px; left: 80%; animation-delay: 0.5s; background: rgba(34,197,94,0.9); }
@keyframes wave {
  0% { transform: translateX(-40%); background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { transform: translateX(40%); background-position: 0% 50%; }
}
@keyframes pulse {
  0% { transform: scale(0.98); opacity: 0.8; }
  50% { transform: scale(1.08); opacity: 1; }
  100% { transform: scale(0.98); opacity: 0.8; }
}
@keyframes float {
  0% { transform: translateY(0); opacity: 0.9; }
  50% { transform: translateY(-6px); opacity: 0.6; }
  100% { transform: translateY(0); opacity: 0.9; }
}
`;

export default SimulationResults;
