/**
 * MiniPanel — embedded mini-viz for Learn articles (`miniViz` sections).
 *
 * Resolves the `learn-heavy-miniviz` deferred extension
 * (src/qft_pcn/viz/EXTENSIONS.md#learn-heavy-miniviz).
 *
 * Two render modes:
 *   - Cheap panels (`pcn-dynamics`, `pcn-coupling`): live-render the real
 *     panel component inside a `transform: scale(0.5)` wrapper, fed a
 *     minimal synthetic Frame whose `layer_states[layer]` is just enough
 *     for the panel's empty-state branch to fall through to a render.
 *   - Heavy panels (everything else — manifold, multifield, mps,
 *     hamiltonian, qpcn, mera, vqc, logic, mera_relax, bridge,
 *     pcn-fields): render a static SVG placeholder with the layer name
 *     and a hint pointing readers at the Viz route.
 */

import type { Frame } from '../lib/types';
import { PcnDynamicsPanel } from '../panels/PcnDynamicsPanel';
import { PcnCouplingPanel } from '../panels/PcnCouplingPanel';

const CHEAP_LAYERS = new Set<string>(['pcn-dynamics', 'pcn-coupling']);

interface Props {
  layer: string;
  fixtureFrameId?: string;
}

/** Build a minimal Frame whose `layer_states[layer]` is populated enough
 *  for the matching cheap panel to render its data path (not the empty
 *  state). The exact shape mirrors the snapshot the live runner emits. */
function syntheticFrame(layer: string): Frame {
  if (layer === 'pcn-dynamics') {
    return {
      step: 0,
      layer_states: {
        'pcn-dynamics': {
          total_free_energy: 0.42,
          per_layer_free_energy: [0.15, 0.12, 0.09, 0.06],
          per_layer_e_norm: [0.30, 0.22, 0.14, 0.08],
          per_layer_pi_mean: [1.0, 0.9, 0.8, 0.7],
          n_layers: 4,
          step: 0,
        },
      },
    };
  }
  if (layer === 'pcn-coupling') {
    return {
      step: 0,
      layer_states: {
        'pcn-coupling': {
          kappa_R: 0.05,
          mean_abs_stress_energy: 0.12,
          mean_abs_ricci: 0.08,
          qpcn_observable_energy: 0.31,
          step: 0,
        },
      },
    };
  }
  // Unused by heavy panels — placeholder.
  return { step: 0, layer_states: { [layer]: {} } };
}

function MiniPlaceholder({ layer }: { layer: string }) {
  return (
    <div className="mini-panel mini-panel-placeholder" data-testid="mini-panel-placeholder">
      <svg viewBox="0 0 320 120" width="100%" height="120" role="img" aria-label={`mini-viz placeholder for ${layer}`}>
        <rect x={1} y={1} width={318} height={118} fill="#0b0e14" stroke="#2f3a55" />
        <text x={160} y={50} textAnchor="middle" fill="#c8d0e0" fontSize="14" fontFamily="monospace">
          {layer}
        </text>
        <text x={160} y={80} textAnchor="middle" fill="#7f8bb0" fontSize="10">
          Open the Viz route for live rendering
        </text>
      </svg>
    </div>
  );
}

export function MiniPanel({ layer, fixtureFrameId: _fixtureFrameId }: Props) {
  if (CHEAP_LAYERS.has(layer)) {
    const frame = syntheticFrame(layer);
    const Inner = layer === 'pcn-dynamics' ? PcnDynamicsPanel : PcnCouplingPanel;
    return (
      <div className="mini-panel mini-panel-live" data-testid="mini-panel-live">
        <div
          className="mini-panel-scale"
          style={{ transform: 'scale(0.5)', transformOrigin: 'top left' }}
        >
          <Inner frame={frame} />
        </div>
      </div>
    );
  }
  return <MiniPlaceholder layer={layer} />;
}
