/**
 * Logic panel — honest rendering of the EvalHamiltonian (§8 / §10.1).
 *
 * Layout:
 *   - Site chain along the bottom.
 *   - One node per EvalTerm anchored at its `site`, colour-coded by rule
 *     family, and coloured *intensity* scaled by per-term residual energy
 *     (relaxation progress). Arity is shown as a thin connector to the
 *     reach site.
 *   - The real "binder-entanglement" signal §1.1 demands is rendered as
 *     a separate inline-SVG per-bond entropy curve (`bond_entropies`).
 *     This reads from the logic-encoded MPS state directly, so it ACTUALLY
 *     measures the variable-binding bonds the architecture's soul invariant
 *     describes. The λ_β / λ_arith / λ_if scalars are kept as a small
 *     legend badge (they describe term weights, not binder geometry).
 *
 * What was removed in deviation D-4:
 *   The previous panel drew evenly spaced "binder-entanglement arcs" whose
 *   opacity was a function of the three global λ weights. Those arcs had
 *   no relationship to any binder pair, use→declaration path, or
 *   entanglement entropy. They are gone.
 *
 * Reads `frame.layer_states.logic` (shape: `snapshot_logic`).
 */

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { useSize } from './common';
import { FrameInterpreter } from '../components/FrameInterpreter';

interface LogicTerm {
  rule_id: string;
  site: number;
  arity: number;
}

interface LogicState {
  n_sites?: number | null;
  term_count?: number | null;
  terms?: LogicTerm[] | null;
  lambda_beta?: number | null;
  lambda_arith?: number | null;
  lambda_if?: number | null;
  residuals?: number[] | null;
  total_energy?: number | null;
  /** Per-bond von Neumann entropy on the logic-encoded MPS state — the
   * REAL binder-entanglement signal per §1.1 / §8 / §10.1. One float (or
   * null) per internal bond. */
  bond_entropies?: Array<number | null> | null;
}

// Stable colour per rule family, so the same rule reads the same everywhere.
function ruleColor(ruleId: string): string {
  const r = ruleId.toLowerCase();
  if (r.includes('beta')) return '#5fd0c8';
  if (r.includes('arith')) return '#d0a05f';
  if (r.includes('cmp')) return '#d05f8f';
  if (r.includes('if')) return '#a05fd0';
  return '#5f8fd0';
}

function LogicDiagram({ st }: { st: LogicState }) {
  const [ref, size] = useSize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const nSites = st.n_sites ?? 0;
    const termCount = st.term_count ?? 0;
    if (nSites <= 0) return;
    const { width, height } = size;
    const mx = 36;
    const siteY = height - 48;
    const xOf = (i: number) =>
      mx + (i / Math.max(1, nSites - 1)) * (width - 2 * mx);

    const terms = (st.terms ?? []).filter(
      (t) => t && Number.isFinite(t.site) && t.site >= 0 && t.site < nSites,
    );
    const residuals = st.residuals ?? null;
    // Normalise residuals into [0, 1] for intensity colouring; 0 ⇒ pale,
    // 1 ⇒ saturated, so a relaxed term reads as "satisfied" and a high-
    // residual term reads as "still contributing energy".
    let maxR = 0;
    if (residuals) for (const r of residuals) if (Math.abs(r) > maxR) maxR = Math.abs(r);
    const residualOf = (i: number): number => {
      if (!residuals || residuals[i] == null || maxR === 0) return 0.5;
      return Math.min(1, Math.abs(residuals[i]) / maxR);
    };

    const topY = 24;
    const astBottom = siteY - 60;

    if (terms.length > 0) {
      const perSite = new Map<number, number>();
      terms.forEach((t, idx) => {
        const slot = perSite.get(t.site) ?? 0;
        perSite.set(t.site, slot + 1);
        const x = xOf(t.site);
        const y = astBottom - slot * 30;
        const color = ruleColor(t.rule_id);
        const intensity = residualOf(idx);

        // Arity reach indicator: a thin chord to `site + arity - 1`.
        const reach = Math.min(nSites - 1, t.site + Math.max(1, t.arity) - 1);
        const xr = xOf(reach);
        svg
          .append('path')
          .attr(
            'd',
            `M${x},${y} Q${(x + xr) / 2},${y - 18} ${xr},${siteY}`,
          )
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 1)
          .attr('opacity', 0.25);
        // connector down to the term's anchor site.
        svg
          .append('line')
          .attr('x1', x)
          .attr('y1', y)
          .attr('x2', x)
          .attr('y2', siteY)
          .attr('stroke', color)
          .attr('stroke-width', 1)
          .attr('opacity', 0.35);

        // Outer ring = rule colour; inner fill intensity = residual energy.
        svg
          .append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', 8)
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 1.5);
        svg
          .append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', 6)
          .attr('fill', color)
          .attr('opacity', 0.2 + 0.8 * intensity)
          .append('title')
          .text(
            `${t.rule_id} @ site ${t.site} (arity ${t.arity})` +
              (residuals && residuals[idx] != null
                ? ` · residual ${residuals[idx].toExponential(2)}`
                : ''),
          );
        svg
          .append('text')
          .attr('x', x + 11)
          .attr('y', y + 3)
          .attr('fill', '#9aa6c8')
          .attr('font-size', 9)
          .attr('text-anchor', 'start')
          .text(t.rule_id);
      });
    } else {
      // --- schematic fallback: balanced binary tree from term_count ------
      const leaves = Math.max(1, termCount);
      const depth = Math.ceil(Math.log2(leaves + 1));
      for (let d = 0; d <= depth; d++) {
        const count = Math.min(leaves, 2 ** d);
        const y = topY + (d / depth) * (astBottom - topY);
        for (let i = 0; i < count; i++) {
          const x = mx + ((i + 0.5) / count) * (width - 2 * mx);
          svg
            .append('circle')
            .attr('cx', x)
            .attr('cy', y)
            .attr('r', d === depth ? 4 : 6)
            .attr('fill', d === depth ? '#d0a05f' : '#5f8fd0')
            .attr('stroke', '#0b0e14');
          if (d > 0) {
            const pCount = Math.min(leaves, 2 ** (d - 1));
            const pi = Math.floor(i / 2);
            const px = mx + ((pi + 0.5) / pCount) * (width - 2 * mx);
            const py = topY + ((d - 1) / depth) * (astBottom - topY);
            svg
              .append('line')
              .attr('x1', px)
              .attr('y1', py)
              .attr('x2', x)
              .attr('y2', y)
              .attr('stroke', '#2f3a55');
          }
        }
      }
    }

    // --- the MPS site chain ---------------------------------------------
    svg
      .append('line')
      .attr('x1', xOf(0))
      .attr('y1', siteY)
      .attr('x2', xOf(nSites - 1))
      .attr('y2', siteY)
      .attr('stroke', '#3a4660');
    for (let i = 0; i < nSites; i++) {
      svg
        .append('circle')
        .attr('cx', xOf(i))
        .attr('cy', siteY)
        .attr('r', 7)
        .attr('fill', '#1b2336')
        .attr('stroke', '#5fd0c8')
        .attr('stroke-width', 2);
      svg
        .append('text')
        .attr('x', xOf(i))
        .attr('y', siteY + 22)
        .attr('fill', '#7f8bb0')
        .attr('font-size', 9)
        .attr('text-anchor', 'middle')
        .text(i);
    }

    // λ legend — small badge (term weights, NOT binder geometry).
    const lams = [
      { name: 'β', v: st.lambda_beta ?? 0, color: '#5fd0c8' },
      { name: 'arith', v: st.lambda_arith ?? 0, color: '#d0a05f' },
      { name: 'if', v: st.lambda_if ?? 0, color: '#a05fd0' },
    ];
    lams.forEach((lam, k) => {
      svg
        .append('text')
        .attr('x', width - 12)
        .attr('y', 16 + k * 14)
        .attr('fill', lam.color)
        .attr('font-size', 10)
        .attr('text-anchor', 'end')
        .text(`λ${lam.name} = ${lam.v.toFixed(2)}`);
    });
  }, [st, size]);

  return (
    <div ref={ref} style={{ width: '100%', height: '100%' }}>
      <svg ref={svgRef} width={size.width} height={size.height} />
    </div>
  );
}

/** Real per-bond von Neumann entropy on the logic-encoded MPS state.
 * §1.1 / §8: variable binding is realized as bond entanglement on the
 * use→declaration path, so this chart is the panel's load-bearing display
 * of the binder-as-entanglement invariant. */
function BondEntropyChart({ entropies }: { entropies: Array<number | null> }) {
  const vals = entropies.map((v) => (v == null ? 0 : v));
  if (vals.length === 0) return null;
  const W = 240;
  const H = 60;
  const pad = 6;
  const maxV = Math.max(1e-9, ...vals);
  const n = vals.length;
  const xAt = (i: number) =>
    pad + (n === 1 ? (W - 2 * pad) / 2 : (i / (n - 1)) * (W - 2 * pad));
  const yAt = (v: number) => H - pad - (v / maxV) * (H - 2 * pad);
  const d = vals
    .map((v, i) =>
      `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(2)} ${yAt(v).toFixed(2)}`,
    )
    .join(' ');
  return (
    <svg
      data-testid="logic-bond-entropy"
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="binder bond entropy per cut"
      style={{ display: 'block' }}
    >
      <path d={d} fill="none" stroke="#5fd0c8" strokeWidth={1.5} />
      <text x={pad} y={10} fill="#7f8bb0" fontSize={9}>
        binder bond S(cut) — real entanglement on logic MPS
      </text>
    </svg>
  );
}

export function LogicPanel({
  frame,
  baselineFrame: _baselineFrame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.logic ?? {}) as LogicState;
  const hasData = (st.n_sites ?? 0) > 0;
  const nTerms = st.terms?.length ?? 0;
  const energy = st.total_energy;
  const bondEntropies = st.bond_entropies ?? null;

  return (
    <PanelShell
      title="Logic — terms over site chain + binder bond entropy"
      step={frame.step}
      meta={
        hasData
          ? `${st.n_sites} sites · ${
              nTerms > 0 ? nTerms : st.term_count ?? 0
            } terms${
              energy != null && Number.isFinite(energy)
                ? ` · ⟨H⟩ = ${energy.toExponential(2)}`
                : ''
            }`
          : undefined
      }
      hasData={hasData}
      emptyMessage="No logic substrate active — start a run with the 'logic' layer."
    >
      <FrameInterpreter layer="logic" />
      {hasData && (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
          }}
        >
          <div style={{ flex: '1 1 auto', minHeight: 0 }}>
            <LogicDiagram st={st} />
          </div>
          {bondEntropies && bondEntropies.length > 0 && (
            <div style={{ flex: '0 0 auto', padding: '4px 8px' }}>
              <BondEntropyChart entropies={bondEntropies} />
            </div>
          )}
        </div>
      )}
    </PanelShell>
  );
}
