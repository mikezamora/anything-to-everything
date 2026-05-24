/**
 * PCN-Dynamics panel: total free energy F, per-layer F contributions,
 * per-layer KL, ‖E‖₂ and mean Π. Time-series strip on total F.
 *
 * Per-layer KL is the entropy-like portion of F (excludes the
 * kappa_R * R geometric regulariser); see QFTPCNLayer.kl_divergence.
 */

import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';
import { FrameInterpreter } from '../components/FrameInterpreter';
import { smallNumberFormat } from './common';

interface PcnDynamicsState {
  total_free_energy?: number | null;
  per_layer_free_energy?: (number | null)[];
  per_layer_kl?: (number | null)[];
  per_layer_e_norm?: (number | null)[];
  per_layer_pi_mean?: (number | null)[];
  n_layers?: number;
  step?: number;
}

export function PcnDynamicsPanel({ frame, baselineFrame: _baselineFrame }: {
  frame: Frame; baselineFrame?: Frame;
}) {
  const st = (frame.layer_states['pcn-dynamics'] ?? {}) as PcnDynamicsState;
  const totalF = st.total_free_energy;
  const perF = st.per_layer_free_energy ?? [];
  const perKL = st.per_layer_kl ?? [];
  const perE = st.per_layer_e_norm ?? [];
  const perPi = st.per_layer_pi_mean ?? [];
  const hasKL = perKL.some((v) => v != null);
  const hasData = perF.length > 0;

  return (
    <PanelShell
      title="PCN Dynamics — free energy + per-layer trajectories"
      step={st.step ?? frame.step}
      hasData={hasData}
      readouts={<PanelReadouts cells={[
        { label: 'depth', value: st.n_layers ?? perF.length },
        { label: 'total F', value: totalF != null ? totalF.toFixed(3) : '—',
          highlightId: 'total_free_energy' },
      ]} />}
      metricsStrip={<MetricsStrip layer="pcn-dynamics" metrics={[{
        key: 'F', label: 'total F', color: '#fbc66a',
        select: (ls) => ls.total_free_energy as number | null | undefined,
      }]} />}
    >
      <FrameInterpreter layer="pcn-dynamics" />
      <table className="pcn-dynamics-table">
        <thead><tr>
          <th>layer</th><th>F</th>{hasKL && <th>KL</th>}<th>‖E‖₂</th><th>mean Π</th>
        </tr></thead>
        <tbody>
          {perF.map((f, i) => (
            <tr key={i}>
              <td>{i}</td>
              <td>{smallNumberFormat(f)}</td>
              {hasKL && <td>{smallNumberFormat(perKL[i])}</td>}
              <td>{smallNumberFormat(perE[i])}</td>
              <td>{smallNumberFormat(perPi[i])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </PanelShell>
  );
}
