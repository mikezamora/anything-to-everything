/**
 * Hamiltonian panel — surfaces the QPCN's generative model per §3.3.4:
 *   - a per-species table of one-site coefficients
 *     (bare_mass ω, kinetic t, quartic μ, source J)
 *   - a species×species coupling matrix for g_{ab} (density-density)
 *     and λ_{ab} (Yukawa-like field coupling), with a toolbar toggle
 *   - a 1D curvature strip R(x_k) along the MPS site axis (the substrate
 *     exposes H.curvature as 1D per §3.3.4, not as a 2D matrix; the
 *     previous panel mistakenly treated it as the latter and rendered
 *     nothing real, see D-2)
 *   - a KaTeX block with the canonical-form decomposition.
 *
 * Reads `frame.layer_states.hamiltonian` (shape: `snapshot_hamiltonian`).
 */

import { useEffect, useRef, useState } from 'react';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';
import { PanelToolbar, type ToolbarItem } from './PanelToolbar';
import { tex, diverging } from './common';
import { FrameInterpreter } from '../components/FrameInterpreter';

interface HamiltonianTerm {
  kind: string;
  species: string;
  site: string;
  coeff: number;
}

interface HamiltonianState {
  n_sites?: number | null;
  d_local?: number | null;
  species_dims?: number[] | null;
  species?: string[] | null;
  per_species?: Record<
    string,
    { bare_mass?: number; kinetic?: number; quartic?: number; source?: number }
  > | null;
  density_couplings?: Record<string, number> | null;
  yukawa_couplings?: Record<string, number> | null;
  curvature_xi?: number | null;
  /** 1D per-site R(x_k); may legacy-render as 2D if older recordings exist. */
  curvature?: number[] | number[][] | number | null;
  terms?: HamiltonianTerm[] | null;
}

/** 1D strip painting one cell per site, coloured by R(x_k) on the
 * diverging ramp. Aligned to the MPS site axis so the user can
 * visually register it against the MPS / Logic site chain panels below. */
function CurvatureStrip({ values }: { values: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const H = 18;
  const W = Math.max(64, values.length * 18);

  useEffect(() => {
    const cnv = canvasRef.current;
    if (!cnv) return;
    const ctx = cnv.getContext('2d');
    if (!ctx) return;
    const n = values.length;
    if (n === 0) {
      ctx.clearRect(0, 0, W, H);
      return;
    }
    let peak = 0;
    for (const v of values) {
      const a = Math.abs(v);
      if (Number.isFinite(a) && a > peak) peak = a;
    }
    const scale = peak > 0 ? peak : 1;
    const cellW = W / n;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = diverging(values[i] / scale);
      ctx.fillRect(Math.floor(i * cellW), 0, Math.ceil(cellW), H);
    }
  }, [values, W]);

  return (
    <div data-testid="hamiltonian-curvature-strip">
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        aria-label="curvature strip R(x_k)"
        style={{
          width: '100%',
          height: H,
          imageRendering: 'pixelated',
          border: '1px solid #2f3a55',
        }}
      />
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          color: '#7f8bb0',
          fontSize: 9,
          marginTop: 2,
        }}
      >
        <span>site 0</span>
        <span>R(x_k) · diverging ramp</span>
        <span>site {values.length - 1}</span>
      </div>
    </div>
  );
}

/** Species×species coupling matrix. Rendered as a small HTML table so
 * jsdom-based tests can assert cell values without a canvas surface. */
function CouplingMatrix({
  species,
  couplings,
  label,
}: {
  species: string[];
  couplings: Record<string, number>;
  label: string;
}) {
  if (species.length === 0) return null;
  // pair-key parser matches snapshot_hamiltonian: "a|b"
  const get = (a: string, b: string): number => {
    const k1 = `${a}|${b}`;
    const k2 = `${b}|${a}`;
    return couplings[k1] ?? couplings[k2] ?? 0;
  };
  let peak = 0;
  for (const v of Object.values(couplings)) {
    const a = Math.abs(v);
    if (Number.isFinite(a) && a > peak) peak = a;
  }
  const scale = peak > 0 ? peak : 1;
  return (
    <table
      data-testid={`coupling-matrix-${label}`}
      style={{
        fontSize: 10,
        borderCollapse: 'collapse',
        color: '#9aa6c8',
      }}
    >
      <thead>
        <tr>
          <th
            style={{ padding: '2px 6px', color: '#7f8bb0', textAlign: 'left' }}
          >
            {label}
          </th>
          {species.map((s) => (
            <th
              key={s}
              style={{ padding: '2px 6px', color: '#7f8bb0' }}
            >
              {s}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {species.map((a) => (
          <tr key={a}>
            <td style={{ padding: '2px 6px', color: '#7f8bb0' }}>{a}</td>
            {species.map((b) => {
              const v = get(a, b);
              return (
                <td
                  key={b}
                  style={{
                    padding: '2px 6px',
                    textAlign: 'right',
                    background: diverging(v / scale),
                    color: Math.abs(v) / scale > 0.5 ? '#0b0e14' : '#9aa6c8',
                    minWidth: 36,
                  }}
                  title={`${a}↔${b}: ${v.toFixed(4)}`}
                >
                  {v === 0 ? '·' : v.toFixed(2)}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function HamiltonianPanel({
  frame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.hamiltonian ?? {}) as HamiltonianState;
  const speciesNames = st.species ?? [];
  const perSpecies = st.per_species ?? {};
  const density = st.density_couplings ?? {};
  const yukawa = st.yukawa_couplings ?? {};

  // The 1D curvature path. Defensively unwrap a legacy 2D array by taking
  // the diagonal (so older recordings still render something honest).
  let curvature1D: number[] = [];
  const curv = st.curvature;
  if (Array.isArray(curv)) {
    if (curv.length > 0 && Array.isArray(curv[0])) {
      const m = curv as number[][];
      curvature1D = m.map((row, i) => row[i] ?? 0);
    } else {
      curvature1D = (curv as number[]).filter((v) => typeof v === 'number');
    }
  }

  const hasData =
    speciesNames.length > 0 ||
    curvature1D.length > 0 ||
    Object.keys(density).length > 0 ||
    Object.keys(yukawa).length > 0;

  const hasDensity = Object.keys(density).length > 0;
  const hasYukawa = Object.keys(yukawa).length > 0;
  const hasAnyCoupling = hasDensity || hasYukawa;
  const showToggle = hasDensity && hasYukawa;

  const [couplingView, setCouplingView] = useState<'density' | 'yukawa'>(
    'density',
  );
  // Effective view: if only one dict is populated, force that one regardless
  // of stored toggle state so users never see an all-zero matrix that's
  // really just "wrong tab selected".
  const effectiveCouplingView: 'density' | 'yukawa' = showToggle
    ? couplingView
    : hasYukawa && !hasDensity
      ? 'yukawa'
      : 'density';

  const toolbarItems: ToolbarItem[] = showToggle
    ? [
        {
          key: 'density',
          label: 'g_{ab} (density)',
          active: couplingView === 'density',
          onToggle: () => setCouplingView('density'),
        },
        {
          key: 'yukawa',
          label: 'λ_{ab} (Yukawa)',
          active: couplingView === 'yukawa',
          onToggle: () => setCouplingView('yukawa'),
        },
      ]
    : [];

  const readouts = (
    <PanelReadouts
      cells={[
        { label: 'N', value: st.n_sites ?? '—' },
        { label: 'd_local', value: st.d_local ?? '—' },
        { label: 'species', value: speciesNames.length },
        {
          label: 'ξ (curv. coup.)',
          value:
            st.curvature_xi != null ? st.curvature_xi.toFixed(3) : '—',
        },
      ]}
    />
  );

  return (
    <PanelShell
      title="Hamiltonian — generative-model coefficients (§3.3.4)"
      step={frame.step}
      meta={
        st.n_sites != null
          ? `${st.n_sites} sites · d=${st.d_local ?? '?'}`
          : undefined
      }
      hasData={hasData}
      emptyMessage="No Hamiltonian active in this frame."
      readouts={readouts}
      toolbar={toolbarItems.length > 0 ? <PanelToolbar items={toolbarItems} /> : undefined}
    >
      <FrameInterpreter layer="hamiltonian" />
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          gap: 8,
          padding: 4,
          overflow: 'auto',
        }}
      >
        {/* Per-species coefficient table */}
        {speciesNames.length > 0 && (
          <table
            data-testid="hamiltonian-per-species"
            style={{
              fontSize: 11,
              borderCollapse: 'collapse',
              color: '#9aa6c8',
            }}
          >
            <thead>
              <tr style={{ color: '#7f8bb0' }}>
                <th style={{ textAlign: 'left', padding: '2px 8px' }}>
                  species
                </th>
                <th style={{ textAlign: 'right', padding: '2px 8px' }}>d</th>
                <th style={{ textAlign: 'right', padding: '2px 8px' }}>
                  ω (mass)
                </th>
                <th style={{ textAlign: 'right', padding: '2px 8px' }}>
                  t (kinetic)
                </th>
                <th style={{ textAlign: 'right', padding: '2px 8px' }}>
                  μ (quartic)
                </th>
                <th style={{ textAlign: 'right', padding: '2px 8px' }}>
                  J (source)
                </th>
              </tr>
            </thead>
            <tbody>
              {speciesNames.map((s, i) => {
                const ps = perSpecies[s] ?? {};
                return (
                  <tr key={s}>
                    <td style={{ padding: '2px 8px' }}>{s}</td>
                    <td style={{ textAlign: 'right', padding: '2px 8px' }}>
                      {st.species_dims?.[i] ?? '?'}
                    </td>
                    <td style={{ textAlign: 'right', padding: '2px 8px' }}>
                      {ps.bare_mass != null ? ps.bare_mass.toFixed(3) : '—'}
                    </td>
                    <td style={{ textAlign: 'right', padding: '2px 8px' }}>
                      {ps.kinetic != null ? ps.kinetic.toFixed(3) : '—'}
                    </td>
                    <td style={{ textAlign: 'right', padding: '2px 8px' }}>
                      {ps.quartic != null ? ps.quartic.toFixed(3) : '—'}
                    </td>
                    <td style={{ textAlign: 'right', padding: '2px 8px' }}>
                      {ps.source != null ? ps.source.toFixed(3) : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}

        {/* Coupling matrix (toggle between density g_ab and Yukawa λ_ab).
         * If neither dict has couplings, hide the matrix entirely and surface
         * an honest note instead of rendering an all-zero grid that users
         * misread as "toggle is broken". */}
        {speciesNames.length > 0 && hasAnyCoupling && (
          <div>
            {effectiveCouplingView === 'density' ? (
              <CouplingMatrix
                species={speciesNames}
                couplings={density}
                label="g_ab"
              />
            ) : (
              <CouplingMatrix
                species={speciesNames}
                couplings={yukawa}
                label="λ_ab"
              />
            )}
          </div>
        )}
        {speciesNames.length > 0 && !hasAnyCoupling && (
          <div
            data-testid="hamiltonian-no-couplings-note"
            style={{
              color: '#7f8bb0',
              fontSize: 11,
              fontStyle: 'italic',
              padding: '4px 6px',
              border: '1px dashed #2f3a55',
              borderRadius: 3,
            }}
          >
            No multi-species couplings declared in this preset. Try a preset
            with at least 2 species + a density/Yukawa term to see the
            coupling matrix.
          </div>
        )}

        {/* Active-terms enumeration (Hamiltonian.terms metadata). */}
        {st.terms && st.terms.length > 0 && (
          <table
            data-testid="hamiltonian-active-terms"
            style={{
              fontSize: 10,
              borderCollapse: 'collapse',
              color: '#9aa6c8',
            }}
          >
            <thead>
              <tr style={{ color: '#7f8bb0' }}>
                <th style={{ textAlign: 'left', padding: '2px 6px' }}>
                  Active terms
                </th>
                <th style={{ textAlign: 'left', padding: '2px 6px' }}>site</th>
                <th style={{ textAlign: 'left', padding: '2px 6px' }}>
                  species
                </th>
                <th style={{ textAlign: 'right', padding: '2px 6px' }}>
                  coeff
                </th>
              </tr>
            </thead>
            <tbody>
              {st.terms.map((t, i) => (
                <tr key={i}>
                  <td style={{ padding: '2px 6px' }}>{t.kind}</td>
                  <td style={{ padding: '2px 6px' }}>{t.site}</td>
                  <td style={{ padding: '2px 6px' }}>{t.species}</td>
                  <td
                    style={{ padding: '2px 6px', textAlign: 'right' }}
                    title={String(t.coeff)}
                  >
                    {t.coeff.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* 1D per-site curvature strip aligned to the site axis */}
        {curvature1D.length > 0 && <CurvatureStrip values={curvature1D} />}

        {/* Canonical-form decomposition */}
        <div
          data-testid="hamiltonian-katex"
          style={{ color: '#c8d0e0', fontSize: 12, marginTop: 4 }}
          dangerouslySetInnerHTML={{
            __html: tex(
              'H = \\sum_i \\left[ \\omega_i n_i + J_i \\phi_i + \\mu_i n_i^2 + \\sum_{a<b} g_{ab} n_a n_b + \\sum_{a<b} \\lambda_{ab} \\phi_a \\phi_b \\right] - \\sum_{\\langle i,j \\rangle} t (a_i^\\dagger a_j + \\text{h.c.})',
            ),
          }}
        />
      </div>
    </PanelShell>
  );
}
