import React, { useMemo } from "react";
import ReactFlow, { Background, Controls, MiniMap, MarkerType, Edge } from "reactflow";
import "reactflow/dist/style.css";
import { ConfigPayload } from "../types";

type Props = {
  config: ConfigPayload | null;
};

const DagView: React.FC<Props> = ({ config }) => {
  const { nodes, edges } = useMemo(() => {
    if (!config) return { nodes: [], edges: [] };
    const nodes = (config.career_states || []).map((s: any, idx: number) => ({
      id: s.id,
      data: { label: `${s.label} (${s.location_id})` },
      position: { x: (idx % 3) * 220, y: Math.floor(idx / 3) * 160 },
      style: {
        border: "1px solid #1f2937",
        background: s.employment_status === "unemployed" ? "#b91c1c" : "#0ea5e9",
        color: "#0b1220",
        borderRadius: 10,
        padding: 8,
      },
    }));
    const edges: Edge[] = (config.transitions || []).map((t: any) => ({
      id: t.id,
      source: t.from_state_id,
      target: t.to_state_id,
      label: t.type,
      animated: true,
      style: { stroke: "#c084fc" },
      markerEnd: { type: MarkerType.ArrowClosed },
    }));
    return { nodes, edges };
  }, [config]);

  if (!config) {
    return <div style={{ opacity: 0.6 }}>Provide a valid config to view the DAG.</div>;
  }

  return (
    <div style={{ height: 420, background: "#0f172a", borderRadius: 8, border: "1px solid #1f2937" }}>
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background />
        <MiniMap />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default DagView;
