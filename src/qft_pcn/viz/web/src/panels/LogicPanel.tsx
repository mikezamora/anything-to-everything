/**
 * Logic panel — a D3 schematic of the logic encoder: the `n_sites` MPS site
 * chain along the bottom, the AST/rule terms drawn above it, and
 * binder-entanglement arcs joining sites whose opacity tracks the encoder's
 * lambda weights (beta / arith / if).
 *
 * When `snapshot_logic` supplies a real `terms` list (one `{rule_id, site,
 * arity}` per `EvalTerm`), each term is drawn as a node anchored at its
 * `site` along the chain, labelled by `rule_id`, with an arity-wide arc.
 * When `terms` is empty/absent the panel falls back to a schematic balanced
 * binary tree built from `term_count`.
 *
 * Reads `frame.layer_states.logic` (shape: `snapshot_logic`).
 */

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { useSize } from './common';

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

    // --- binder-entanglement arcs over the site chain --------------------
    const lams = [
      { name: 'β', v: st.lambda_beta ?? 0, color: '#5fd0c8' },
      { name: 'arith', v: st.lambda_arith ?? 0, color: '#d0a05f' },
      { name: 'if', v: st.lambda_if ?? 0, color: '#a05fd0' },
    ];
    const maxLam = Math.max(1e-6, ...lams.map((l) => Math.abs(l.v)));
    lams.forEach((lam, k) => {
      // span pairs of sites; opacity ~ relative lambda weight.
      const span = k + 1;
      for (let i = 0; i + span < nSites; i += span + 1) {
        const x1 = xOf(i);
        const x2 = xOf(i + span);
        const lift = 20 + span * 18;
        svg
          .append('path')
          .attr(
            'd',
            `M${x1},${siteY} Q${(x1 + x2) / 2},${siteY - lift} ${x2},${siteY}`,
          )
          .attr('fill', 'none')
          .attr('stroke', lam.color)
          .attr('stroke-width', 2)
          .attr('opacity', 0.15 + 0.7 * (Math.abs(lam.v) / maxLam));
      }
    });

    const terms = (st.terms ?? []).filter(
      (t) => t && Number.isFinite(t.site) && t.site >= 0 && t.site < nSites,
    );
    const topY = 24;
    const astBottom = siteY - 90;

    if (terms.length > 0) {
      // --- real AST/rule terms: one node per term, anchored at its site --
      // Stack terms that share a site so labels do not collide.
      const perSite = new Map<number, number>();
      terms.forEach((t) => {
        const slot = perSite.get(t.site) ?? 0;
        perSite.set(t.site, slot + 1);
        const x = xOf(t.site);
        const y = astBottom - slot * 30;
        const color = ruleColor(t.rule_id);

        // arity arc: spans `site .. site + arity - 1` along the chain.
        const reach = Math.min(nSites - 1, t.site + Math.max(1, t.arity) - 1);
        const xr = xOf(reach);
        svg
          .append('path')
          .attr(
            'd',
            `M${x},${y} Q${(x + xr) / 2},${y - 24} ${xr},${siteY}`,
          )
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 1.5)
          .attr('opacity', 0.5);
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

        svg
          .append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', 7)
          .attr('fill', color)
          .attr('stroke', '#0b0e14')
          .append('title')
          .text(`${t.rule_id} @ site ${t.site} (arity ${t.arity})`);
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

    // legend
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

  return (
    <PanelShell
      title="Logic — AST over site chain + binder arcs"
      step={frame.step}
      meta={
        hasData
          ? `${st.n_sites} sites · ${
              nTerms > 0 ? nTerms : st.term_count ?? 0
            } terms`
          : undefined
      }
      hasData={hasData}
      isExtension={!hasData}
      extensionAnchor="#logic-live-relaxation-panel"
      emptyMessage="logic is currently fixture-only — see EXTENSIONS.md."
    >
      {hasData && <LogicDiagram st={st} />}
    </PanelShell>
  );
}
