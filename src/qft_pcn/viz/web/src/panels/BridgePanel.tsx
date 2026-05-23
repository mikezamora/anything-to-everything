/**
 * Bridge panel — inspects a single `RunResult` from the bridge runtime
 * (`run_problem`). Reads `frame.layer_states.bridge`
 * (shape: `snapshot_run_result`).
 *
 * Surfaces:
 *   - trotter_steps + final energy + converged flag as readouts;
 *   - solved_ast_text as a <pre> block (None until a MERA-based runner
 *     populates `result.solved_ast`);
 *   - miniatures of the embedded MPS (final ground state) + Hamiltonian
 *     snapshots, rendered as compact readouts so the panel composes
 *     cleanly without re-mounting the full Mps/Hamiltonian panels.
 */

import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';

interface MpsMiniature {
  bond_dims?: number[] | null;
  entropies?: Array<number | null> | null;
  n_sites?: number | null;
  d_local?: number | null;
}

interface HamiltonianMiniature {
  n_sites?: number | null;
  d_local?: number | null;
  species?: string[] | null;
}

interface BridgeState {
  mps?: MpsMiniature | null;
  hamiltonian?: HamiltonianMiniature | null;
  trotter_steps?: number | null;
  energy?: number | null;
  converged?: boolean | null;
  solved_ast_text?: string | null;
  meta_n_leaves?: number | null;
}

export function BridgePanel({
  frame,
  baselineFrame: _baselineFrame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.bridge ?? {}) as BridgeState;
  const mps = st.mps ?? null;
  const ham = st.hamiltonian ?? null;
  const trotter = st.trotter_steps ?? null;
  const energy = st.energy ?? null;
  const converged = st.converged ?? null;
  const solvedAst = st.solved_ast_text ?? null;
  const metaNLeaves = st.meta_n_leaves ?? null;

  const hasData =
    trotter != null ||
    energy != null ||
    mps != null ||
    ham != null;

  const chiMax =
    mps && mps.bond_dims && mps.bond_dims.length > 0
      ? Math.max(...mps.bond_dims)
      : null;

  const readouts = (
    <PanelReadouts
      cells={[
        { label: 'trotter steps', value: trotter ?? '—' },
        {
          label: '⟨H⟩',
          value:
            energy != null && Number.isFinite(energy)
              ? energy.toExponential(3)
              : '—',
        },
        {
          label: 'converged',
          value: converged == null ? '—' : converged ? 'yes' : 'no',
        },
        { label: 'meta leaves', value: metaNLeaves ?? '—' },
      ]}
    />
  );

  return (
    <PanelShell
      title="Bridge — RunResult inspector"
      step={frame.step}
      meta={
        trotter != null
          ? `${trotter} Trotter steps${
              energy != null && Number.isFinite(energy)
                ? ` · ⟨H⟩ = ${energy.toExponential(2)}`
                : ''
            }`
          : undefined
      }
      hasData={hasData}
      emptyMessage="No bridge result available — start a run with the 'bridge' layer."
      readouts={readouts}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          gap: 10,
          padding: 6,
          overflow: 'auto',
        }}
      >
        <div data-testid="bridge-solved-ast">
          <div style={{ fontSize: 10, color: '#7f8bb0', marginBottom: 2 }}>
            solved_ast (None until a MERA-based runner populates it):
          </div>
          <pre
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
            {solvedAst ?? '— (not populated)'}
          </pre>
        </div>

        {mps && (
          <div data-testid="bridge-mps-miniature">
            <div style={{ fontSize: 10, color: '#7f8bb0', marginBottom: 2 }}>
              ground-state MPS:
            </div>
            <PanelReadouts
              cells={[
                { label: 'N', value: mps.n_sites ?? '—' },
                { label: 'd_local', value: mps.d_local ?? '—' },
                { label: 'χ_max', value: chiMax ?? '—' },
                {
                  label: 'bonds',
                  value: (mps.bond_dims ?? []).length,
                },
              ]}
            />
            {mps.bond_dims && mps.bond_dims.length > 0 && (
              <div
                style={{
                  fontSize: 10,
                  color: '#9aa6c8',
                  fontFamily: 'monospace',
                  marginTop: 2,
                }}
              >
                χ: [{mps.bond_dims.join(', ')}]
              </div>
            )}
          </div>
        )}

        {ham && (
          <div data-testid="bridge-hamiltonian-miniature">
            <div style={{ fontSize: 10, color: '#7f8bb0', marginBottom: 2 }}>
              Hamiltonian:
            </div>
            <PanelReadouts
              cells={[
                { label: 'N', value: ham.n_sites ?? '—' },
                { label: 'd_local', value: ham.d_local ?? '—' },
                {
                  label: 'species',
                  value: (ham.species ?? []).length,
                },
              ]}
            />
            {ham.species && ham.species.length > 0 && (
              <div
                style={{
                  fontSize: 10,
                  color: '#9aa6c8',
                  fontFamily: 'monospace',
                  marginTop: 2,
                }}
              >
                species: [{ham.species.join(', ')}]
              </div>
            )}
          </div>
        )}
      </div>
    </PanelShell>
  );
}
