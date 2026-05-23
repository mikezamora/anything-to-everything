/**
 * MERA imag-time relaxation panel — the §10.10 induction-theorem live
 * demo. Reads `frame.layer_states.mera_relax` (shape: `snapshot_mera_relax`).
 *
 * Surfaces:
 *   - total energy readout + time series (should monotonically decay
 *     under imag-time);
 *   - per-term residuals as a coloured table (one row per term);
 *   - the AST text round-tripped from the live MERA leaves;
 *   - the indices of the Forall-protected leaves (clamped by the trotter
 *     step's `frozen_leaves=` argument — the load-bearing §1.1 / §10.10
 *     invariant).
 */

import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';

interface MeraRelaxResidual {
  rule_id: string;
  site: number;
  value: number;
}

interface MeraRelaxState {
  total_energy?: number | null;
  residuals?: MeraRelaxResidual[] | null;
  n_leaves?: number | null;
  layer_bond_dims?: number[] | null;
  forall_protected_leaves?: number[] | null;
  ast_text?: string | null;
  step?: number | null;
}

/** Linear-magnitude colour: pale (low residual = satisfied) → saturated
 * orange (high residual = active). Mirrors the LogicPanel intensity
 * convention so the two relaxation surfaces read the same way. */
function residualColor(magnitude: number, peak: number): string {
  if (!Number.isFinite(magnitude) || peak <= 0) return '#1b2336';
  const t = Math.min(1, Math.abs(magnitude) / peak);
  // Mix between a dim grey and a saturated orange.
  const r = Math.round(27 + (208 - 27) * t);
  const g = Math.round(35 + (160 - 35) * t);
  const b = Math.round(54 + (95 - 54) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function ResidualTable({ rows }: { rows: MeraRelaxResidual[] }) {
  if (rows.length === 0) return null;
  let peak = 0;
  for (const r of rows) {
    const a = Math.abs(r.value);
    if (Number.isFinite(a) && a > peak) peak = a;
  }
  return (
    <table
      data-testid="mera-relax-residuals"
      style={{
        fontSize: 11,
        borderCollapse: 'collapse',
        color: '#9aa6c8',
        width: '100%',
      }}
    >
      <thead>
        <tr style={{ color: '#7f8bb0' }}>
          <th style={{ textAlign: 'left', padding: '2px 6px' }}>rule_id</th>
          <th style={{ textAlign: 'right', padding: '2px 6px' }}>site</th>
          <th style={{ textAlign: 'right', padding: '2px 6px' }}>residual</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => {
          const bg = residualColor(r.value, peak);
          const hot = peak > 0 && Math.abs(r.value) / peak > 0.5;
          return (
            <tr key={`${r.rule_id}@${r.site}@${i}`} style={{ background: bg }}>
              <td
                style={{
                  padding: '2px 6px',
                  color: hot ? '#0b0e14' : '#c8d0e0',
                  fontFamily: 'monospace',
                }}
              >
                {r.rule_id}
              </td>
              <td
                style={{
                  padding: '2px 6px',
                  textAlign: 'right',
                  color: hot ? '#0b0e14' : '#c8d0e0',
                }}
              >
                {r.site}
              </td>
              <td
                style={{
                  padding: '2px 6px',
                  textAlign: 'right',
                  color: hot ? '#0b0e14' : '#c8d0e0',
                  fontFamily: 'monospace',
                }}
                title={r.value.toString()}
              >
                {r.value.toExponential(2)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function MeraRelaxPanel({
  frame,
  baselineFrame: _baselineFrame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.mera_relax ?? {}) as MeraRelaxState;
  const energy = st.total_energy ?? null;
  const nLeaves = st.n_leaves ?? null;
  const residuals = (st.residuals ?? []) as MeraRelaxResidual[];
  const protectedLeaves = st.forall_protected_leaves ?? [];
  const astText = st.ast_text ?? null;
  const layerBondDims = st.layer_bond_dims ?? [];

  const hasData =
    nLeaves != null && nLeaves > 0;

  // Sort residuals by descending |value| so the active rules float to the
  // top; cap the rendered list so the panel stays compact under the 360-
  // term enumeration MeraEvalHamiltonian produces.
  const sortedResiduals = [...residuals]
    .filter((r) => Number.isFinite(r.value) && Math.abs(r.value) > 1e-12)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 32);

  const readouts = (
    <PanelReadouts
      cells={[
        {
          label: 'total energy',
          value: energy != null ? energy.toExponential(3) : '—',
          highlightId: 'total_energy',
        },
        { label: 'n_leaves', value: nLeaves ?? '—' },
        {
          label: '∀-protected',
          value: protectedLeaves.length,
        },
        {
          label: 'layers',
          value: layerBondDims.length,
        },
      ]}
    />
  );

  const metricsStrip = (
    <MetricsStrip
      layer="mera_relax"
      metrics={[
        {
          key: 'total_energy',
          label: '⟨H⟩',
          color: '#9aedc1',
          select: (ls) => {
            const v = ls.total_energy as number | null | undefined;
            return typeof v === 'number' && Number.isFinite(v) ? v : null;
          },
        },
      ]}
    />
  );

  return (
    <PanelShell
      title="MERA imag-time relax — §10.10 induction theorem"
      step={frame.step}
      meta={
        nLeaves != null
          ? `${nLeaves} leaves · ${protectedLeaves.length} ∀-protected`
          : undefined
      }
      hasData={hasData}
      emptyMessage="No mera_relax substrate active — start a run with the 'mera_relax' layer."
      readouts={readouts}
      metricsStrip={metricsStrip}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          gap: 8,
          padding: 6,
          overflow: 'auto',
        }}
      >
        <div data-testid="mera-relax-forall-leaves" style={{ fontSize: 11 }}>
          <span style={{ color: '#7f8bb0', marginRight: 6 }}>
            ∀-protected leaves ({protectedLeaves.length}):
          </span>
          <code style={{ color: '#9aedc1' }}>
            {protectedLeaves.length > 0 ? protectedLeaves.join(', ') : '—'}
          </code>
        </div>
        {astText !== null && (
          <pre
            data-testid="mera-relax-ast-text"
            style={{
              fontSize: 11,
              color: '#c8d0e0',
              background: '#0b0e14',
              border: '1px solid #2f3a55',
              padding: 6,
              margin: 0,
              maxHeight: 96,
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {astText}
          </pre>
        )}
        <div style={{ minHeight: 0, overflow: 'auto' }}>
          <ResidualTable rows={sortedResiduals} />
        </div>
      </div>
    </PanelShell>
  );
}
