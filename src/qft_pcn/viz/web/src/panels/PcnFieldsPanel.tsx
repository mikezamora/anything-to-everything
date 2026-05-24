/**
 * PCN-Fields panel: stacks Φ / E / Π for every PCN layer.
 *
 * Each layer card renders a small 2D heatmap (no R3F — keep it cheap for
 * runs with 4+ layers). Clicking a card expands it to full panel width.
 * Toggle which field (Φ, E, Π) is plotted.
 */

import { useState } from 'react';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelToolbar } from './PanelToolbar';
import { PanelReadouts } from './PanelReadouts';
import { diverging, normGrid, as2DGrid, type PhiLike } from './common';
import { FrameInterpreter } from '../components/FrameInterpreter';

type Grid = number[][];

// `snapshot_pcn_fields` emits phi/E/Pi as 3D `(channels, Nx, Ny)` (Field.values
// is shape (C, Nx, Ny) and `_grid` .tolist()s it). The heatmap needs 2D, so
// the layer loop calls `as2DGrid` before handing it to <Heatmap>.
interface LayerState { phi?: PhiLike; E?: PhiLike; Pi?: PhiLike; channels?: number; }
interface PcnFieldsState { layers?: LayerState[]; step?: number; }

function Heatmap({ grid, w = 90, h = 90 }: { grid: Grid; w?: number; h?: number }) {
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  if (rows === 0 || cols === 0) return null;
  const { norm } = normGrid(grid);
  const cellW = w / cols, cellH = h / rows;
  const cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      cells.push(
        <rect key={`${r}-${c}`} x={c * cellW} y={r * cellH}
              width={cellW} height={cellH}
              fill={diverging(norm[r][c])} />);
    }
  }
  return <svg width={w} height={h}>{cells}</svg>;
}

function norm2(g?: PhiLike) {
  const grid = as2DGrid(g);
  if (!grid) return 0;
  let s = 0;
  for (const row of grid) for (const v of row)
    if (Number.isFinite(v)) s += v * v;
  return Math.sqrt(s);
}

export function PcnFieldsPanel({ frame, baselineFrame: _baselineFrame }: {
  frame: Frame; baselineFrame?: Frame;
}) {
  const st = (frame.layer_states['pcn-fields'] ?? {}) as PcnFieldsState;
  const layers = st.layers ?? [];
  const [field, setField] = useState<'phi' | 'E' | 'Pi'>('phi');
  const [expanded, setExpanded] = useState<number | null>(null);

  const hasData = layers.length > 0;
  const totalPhi = layers.reduce((a, l) => a + norm2(l.phi), 0);
  const totalE = layers.reduce((a, l) => a + norm2(l.E), 0);
  const meanPi = layers.length
    ? layers.reduce((a, l) => {
        const g = as2DGrid(l.Pi) ?? [];
        const flat = g.flat();
        return a + (flat.length ? flat.reduce((p, q) => p + q, 0) / flat.length : 0);
      }, 0) / layers.length
    : 0;

  return (
    <PanelShell
      title="PCN Fields — Φ / E / Π stack"
      step={st.step ?? frame.step}
      hasData={hasData}
      toolbar={
        <PanelToolbar items={(['phi', 'E', 'Pi'] as const).map((k) => ({
          key: k, label: k, active: field === k,
          onToggle: () => setField(k),
        }))} />
      }
      readouts={<PanelReadouts cells={[
        { label: 'depth', value: layers.length },
        { label: '‖Φ‖₂', value: totalPhi.toFixed(3) },
        { label: '‖E‖₂', value: totalE.toFixed(3) },
        { label: 'mean Π', value: meanPi.toFixed(3) },
      ]} />}
    >
      <FrameInterpreter layer="pcn-fields" />
      <div style={{ padding: 8, overflowY: 'auto' }}>
        {layers.map((layer, i) => {
          const raw = field === 'phi' ? layer.phi
                    : field === 'E' ? layer.E : layer.Pi;
          const g = as2DGrid(raw);
          const isExpanded = expanded === i;
          return (
            <div
              key={i}
              onClick={() => setExpanded(isExpanded ? null : i)}
              style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: 6, cursor: 'pointer',
                background: isExpanded ? '#16203a' : 'transparent',
                borderRadius: 4, marginBottom: 4,
              }}
            >
              <span style={{ width: 60, color: '#8c97b3' }}>
                layer {i}
              </span>
              {g
                ? <Heatmap grid={g}
                           w={isExpanded ? 320 : 90}
                           h={isExpanded ? 320 : 90} />
                : <span>—</span>}
              <span style={{ fontSize: 11, color: '#7e8aa3' }}>
                {layer.channels ?? 1} ch · ‖·‖₂ {norm2(g).toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </PanelShell>
  );
}
