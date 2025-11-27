// React Flow DAG visualization of career states and transitions.
import React, { useMemo } from "react";
import ReactFlow, { Background, Controls, MiniMap, MarkerType, Edge } from "reactflow";
import "reactflow/dist/style.css";
import { ConfigPayload } from "../types";

type Props = {
  config: ConfigPayload | null;
};

/**
 * DagView visualizes the career graph using React Flow. It converts the structured config into node
 * and edge objects on the fly so edits in the builder immediately reshape the diagram. The component
 * deliberately stays presentation-only: it does no editing or validation, focusing solely on taking
 * the domain model and expressing it as a legible DAG with minimal styling.
 */
const DagView: React.FC<Props> = ({ config }) => {
  /**
   * Build the graph model for React Flow. Nodes are laid out in a simple grid to avoid overlap
   * without running a heavy layout engine; edge styling communicates transition type and direction.
   * Because the config can change frequently, the computation is memoized on the full config object
   * to avoid rehydrating React Flow state on unrelated renders.
   */
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

  /**
   * Compute the container height based on the lowest node so the flow viewport never clips the
   * diagram. The calculation deliberately pads the bottom so the last row has breathing room and
   * ensures a sensible minimum height when no nodes exist, preserving layout stability for the page.
   */
  const containerHeight = useMemo(() => {
    if (nodes.length === 0) return 420;
    const maxY = Math.max(...nodes.map((n) => n.position.y));
    return Math.max(420, maxY + 220);
  }, [nodes]);

  if (!config) {
    return <div style={{ opacity: 0.6 }}>Provide a valid config to view the DAG.</div>;
  }

  return (
    <div
      style={{
        height: containerHeight,
        background: "#0f172a",
        borderRadius: 8,
        border: "1px solid #1f2937",
        transition: "height 160ms ease",
      }}
    >
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background />
        <MiniMap />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default DagView;
