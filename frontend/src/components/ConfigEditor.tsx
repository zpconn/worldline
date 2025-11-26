import React from "react";

type Props = {
  value: string;
  onChange: (val: string) => void;
};

const ConfigEditor: React.FC<Props> = ({ value, onChange }) => {
  return (
    <div>
      <p style={{ marginTop: 0, marginBottom: 8, opacity: 0.75 }}>
        Edit JSON directly. Include locations, career states, transitions, strategies, and settings.
      </p>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%",
          height: 400,
          fontFamily: "ui-monospace, SFMono-Regular",
          background: "#0f172a",
          color: "#e2e8f0",
          border: "1px solid #1f2937",
          borderRadius: 8,
          padding: 10,
        }}
      />
    </div>
  );
};

export default ConfigEditor;
