/**
 * PCN-Coupling panel: visualizes the bidirectional QFT <-> PCN bridge.
 *
 * Two arrows whose widths animate with live magnitudes:
 *   - top arrow (PCN -> QFT): mean |stress-energy| from the error field
 *   - bottom arrow (QFT -> PCN): the QPCN's variational energy
 *     (proxy for "QFT pulling the PCN's observation targets")
 *
 * Readouts: kappa_R coupling constant, mean |stress-energy|, mean |Ricci|.
 */

import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';
import { FrameInterpreter } from '../components/FrameInterpreter';
import { smallNumberFormat } from './common';

interface PcnCouplingState {
  kappa_R?: number | null;
  mean_abs_stress_energy?: number | null;
  mean_abs_ricci?: number | null;
  qpcn_observable_energy?: number | null;
  step?: number;
}

function arrowWidth(magnitude: number | null | undefined): number {
  if (magnitude == null || !Number.isFinite(magnitude)) return 4;
  const clipped = Math.min(Math.abs(magnitude), 1);
  return 4 + clipped * 18;
}

export function PcnCouplingPanel({ frame, baselineFrame: _baselineFrame }: {
  frame: Frame; baselineFrame?: Frame;
}) {
  const st = (frame.layer_states['pcn-coupling'] ?? {}) as PcnCouplingState;
  const hasData = st.kappa_R != null;
  const upWidth = arrowWidth(st.mean_abs_stress_energy);
  const downWidth = arrowWidth(st.qpcn_observable_energy);

  return (
    <PanelShell
      title="PCN ↔ QFT — bidirectional coupling"
      step={st.step ?? frame.step}
      hasData={hasData}
      readouts={<PanelReadouts cells={[
        { label: 'κ_R', value: st.kappa_R != null ? st.kappa_R.toFixed(4) : '—' },
        { label: 'mean |T|', value: st.mean_abs_stress_energy != null
            ? st.mean_abs_stress_energy.toExponential(2) : '—',
          highlightId: 'mean_abs_stress_energy' },
        { label: 'mean |R|', value: smallNumberFormat(st.mean_abs_ricci) },
        { label: '⟨H⟩', value: st.qpcn_observable_energy != null
            ? st.qpcn_observable_energy.toFixed(3) : '—' },
      ]} />}
      metricsStrip={<MetricsStrip layer="pcn-coupling" metrics={[
        { key: 'T', label: 'mean |T|', color: '#6cd0ff',
          select: (ls) => ls.mean_abs_stress_energy as number | null | undefined },
        { key: 'R', label: 'mean |R|', color: '#fbc66a',
          select: (ls) => ls.mean_abs_ricci as number | null | undefined },
      ]} />}
    >
      <FrameInterpreter layer="pcn-coupling" />
      <svg viewBox="0 0 400 220" width="100%" height="220">
        {/* PCN box */}
        <rect x={20} y={70} width={120} height={80}
              fill="#16203a" stroke="#2a3450" />
        <text x={80} y={115} textAnchor="middle" fill="#e0e6f3"
              fontSize="14">PCN</text>

        {/* QFT box */}
        <rect x={260} y={70} width={120} height={80}
              fill="#16203a" stroke="#2a3450" />
        <text x={320} y={115} textAnchor="middle" fill="#e0e6f3"
              fontSize="14">QFT</text>

        {/* PCN -> QFT (top arrow): stress-energy */}
        <line data-testid="arrow-pcn-to-qft"
              x1={140} y1={90} x2={260} y2={90}
              stroke="#6cd0ff" strokeWidth={upWidth}
              markerEnd="url(#arrowhead-up)" />
        <text x={200} y={68} textAnchor="middle"
              fill="#9aa3bb" fontSize="11">T_μν</text>
        <text data-testid="arrow-pcn-to-qft-magnitude"
              x={200} y={82} textAnchor="middle"
              fill="#6cd0ff" fontSize="10" fontFamily="monospace">
          mean|T| = {smallNumberFormat(st.mean_abs_stress_energy)}
        </text>

        {/* QFT -> PCN (bottom arrow): variational energy ⟨H⟩ (proxy for
            per-observable feedback; see explainer for the §3.4 caveat). */}
        <line data-testid="arrow-qft-to-pcn"
              x1={260} y1={130} x2={140} y2={130}
              stroke="#fbc66a" strokeWidth={downWidth}
              markerEnd="url(#arrowhead-down)" />
        <text x={200} y={146} textAnchor="middle"
              fill="#9aa3bb" fontSize="11">⟨H⟩</text>
        <text data-testid="arrow-qft-to-pcn-magnitude"
              x={200} y={160} textAnchor="middle"
              fill="#fbc66a" fontSize="10" fontFamily="monospace">
          ⟨H⟩ = {smallNumberFormat(st.qpcn_observable_energy)}
        </text>

        {/* Arrow-width legend: thin = 0, thick ≈ 1.0. */}
        <g data-testid="pcn-coupling-arrow-legend">
          <text x={10} y={196} fill="#7f8bb0" fontSize="9">
            arrow width
          </text>
          <line x1={75} y1={193} x2={120} y2={193}
                stroke="#5f6b86" strokeWidth={4} />
          <text x={125} y={196} fill="#7f8bb0" fontSize="9">
            |·|≈0
          </text>
          <line x1={170} y1={193} x2={215} y2={193}
                stroke="#5f6b86" strokeWidth={20} />
          <text x={222} y={196} fill="#7f8bb0" fontSize="9">
            |·|≥1
          </text>
        </g>

        <defs>
          <marker id="arrowhead-up" markerWidth="8" markerHeight="8"
                  refX="6" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6cd0ff" />
          </marker>
          <marker id="arrowhead-down" markerWidth="8" markerHeight="8"
                  refX="6" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#fbc66a" />
          </marker>
        </defs>
      </svg>
    </PanelShell>
  );
}
