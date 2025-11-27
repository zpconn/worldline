// JSON editor/preview with apply hook back into the builder state.
import React from "react";

type Props = {
  value: string;
  onChange: (val: string) => void;
  onApply?: () => void;
  parseError?: string | null;
  helperText?: string;
};

/**
 * ConfigEditor is a focused JSON textarea that mirrors the structured builder. It keeps no internal
 * state of its own; instead, it streams all changes up through `onChange` so the parent can keep a
 * canonical string copy and validate it. When `onApply` is provided, the component renders a button
 * that triggers the parent's parse-and-hydrate flow, allowing users to experiment with raw JSON and
 * then synchronize their edits back into the normalized config with a single action.
 */
const ConfigEditor: React.FC<Props> = ({ value, onChange, onApply, parseError, helperText }) => {
  return (
    <div style={editorRoot}>
      <p style={{ marginTop: 0, marginBottom: 8, opacity: 0.75 }}>
        {helperText || "Edit JSON directly. Use Apply to sync your edits back into the builder."}
      </p>
      {onApply ? (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 6 }}>
          <button onClick={onApply} style={applyButton}>
            Apply JSON to builder
          </button>
        </div>
      ) : null}
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%",
          height: "100%",
          minHeight: 260,
          flex: 1,
          fontFamily: "ui-monospace, SFMono-Regular",
          background: "#0f172a",
          color: "#e2e8f0",
          border: "1px solid #1f2937",
          borderRadius: 8,
          padding: 10,
        }}
      />
      <div style={{ marginTop: 6, fontSize: 12, opacity: parseError ? 1 : 0.7, color: parseError ? "#f97316" : "#e2e8f0" }}>
        {parseError || "JSON parses correctly."}
      </div>
    </div>
  );
};

export default ConfigEditor;

const applyButton: React.CSSProperties = {
  background: "#38bdf8",
  border: "none",
  color: "#0b1220",
  padding: "8px 12px",
  borderRadius: 8,
  cursor: "pointer",
  fontWeight: 700,
};

const editorRoot: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  height: "100%",
};
