/**
 * MPS panel — top: a D3 tensor-network diagram, one node per site, where bond
 * line width is proportional to the bond dimension; bottom: a Plotly curve of
 * per-bond entanglement entropy with the log(chi) ceiling drawn in.
 *
 * Reads `frame.layer_states.mps` (shape: `snapshot_mps`).
 */

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import Plotly from 'plotly.js-dist-min';
import type { Data as PlotData, Layout as PlotLayout } from 'plotly.js-dist-min';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';
import { useSize } from './common';
import { FrameInterpreter } from '../components/FrameInterpreter';

interface MpsState {
  bond_dims?: number[] | null;
  entropies?: Array<number | null> | null;
  n_sites?: number | null;
  d_local?: number | null;
}

/** D3 chain: site circles joined by bonds whose width tracks bond dimension. */
function TensorChain({
  bondDims,
  nSites: nSitesProp,
}: {
  bondDims: number[];
  nSites?: number | null;
}) {
  const [ref, size] = useSize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const { width, height } = size;
    // bond_dims is the interior-bond list (length n_sites - 1). Prefer the
    // explicit n_sites field; fall back to bondDims.length + 1.
    const nSites = Math.max(
      0,
      nSitesProp != null && nSitesProp > 0 ? nSitesProp : bondDims.length + 1,
    );
    if (nSites <= 0) return;

    const margin = 40;
    const y = height / 2;
    const xs = d3
      .scaleLinear()
      .domain([0, Math.max(1, nSites - 1)])
      .range([margin, width - margin]);
    const maxBond = d3.max(bondDims) ?? 1;
    const wScale = d3
      .scaleLinear()
      .domain([1, Math.max(1, maxBond)])
      .range([1, 16]);

    // Interior bonds connect site i to i+1, dim = bondDims[i].
    for (let i = 0; i < nSites - 1; i++) {
      const dim = bondDims[i] ?? 1;
      svg
        .append('line')
        .attr('x1', xs(i))
        .attr('y1', y)
        .attr('x2', xs(i + 1))
        .attr('y2', y)
        .attr('stroke', '#4f7fd0')
        .attr('stroke-width', wScale(dim));
      svg
        .append('text')
        .attr('x', (xs(i) + xs(i + 1)) / 2)
        .attr('y', y - 16)
        .attr('fill', '#7f8bb0')
        .attr('font-size', 10)
        .attr('text-anchor', 'middle')
        .text(`χ=${dim}`);
    }

    for (let i = 0; i < nSites; i++) {
      const g = svg.append('g').attr('transform', `translate(${xs(i)},${y})`);
      g.append('circle')
        .attr('r', 13)
        .attr('fill', '#1b2336')
        .attr('stroke', '#5fd0c8')
        .attr('stroke-width', 2);
      // physical leg
      g.append('line')
        .attr('x1', 0)
        .attr('y1', 13)
        .attr('x2', 0)
        .attr('y2', 34)
        .attr('stroke', '#5f6b86');
      g.append('text')
        .attr('fill', '#c8d0e0')
        .attr('font-size', 9)
        .attr('text-anchor', 'middle')
        .attr('dy', 3)
        .text(i);
    }
  }, [bondDims, nSitesProp, size]);

  return (
    <div ref={ref} style={{ width: '100%', height: '100%' }}>
      <svg ref={svgRef} width={size.width} height={size.height} />
    </div>
  );
}

/** Plotly entropy curve with the log(chi) ceiling overlaid. */
function EntropyCurve({
  entropies,
  bondDims,
}: {
  entropies: Array<number | null>;
  bondDims: number[];
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const x = entropies.map((_, i) => i);
    const s = entropies.map((v) => (v == null ? null : v));
    // entropy[i] cuts the i-th interior bond, whose dimension is bondDims[i].
    const ceil = entropies.map((_, i) => Math.log(Math.max(1, bondDims[i] ?? 1)));

    Plotly.react(
      el,
      [
        {
          x,
          y: ceil,
          name: 'log χ ceiling',
          mode: 'lines',
          line: { color: '#5f6b86', dash: 'dot' },
        },
        {
          x,
          y: s,
          name: 'entropy',
          mode: 'lines+markers',
          line: { color: '#5fd0c8' },
          marker: { color: '#5fd0c8' },
        },
      ] as PlotData[],
      {
        paper_bgcolor: '#0b0e14',
        plot_bgcolor: '#0b0e14',
        font: { color: '#8fa8d8', size: 10 },
        margin: { l: 40, r: 10, t: 24, b: 30 },
        title: { text: 'Entanglement entropy per bond', font: { size: 11 } },
        xaxis: { title: { text: 'bond' }, gridcolor: '#1c2230' },
        yaxis: { title: { text: 'S' }, gridcolor: '#1c2230' },
        showlegend: true,
        legend: { x: 0, y: 1, font: { size: 9 } },
      } as Partial<PlotLayout>,
      { displayModeBar: false, responsive: true },
    );
    return () => {
      Plotly.purge(el);
    };
  }, [entropies, bondDims]);

  return <div ref={ref} style={{ width: '100%', height: '100%' }} />;
}

export function MpsPanel({
  frame,
  baselineFrame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.mps ?? {}) as MpsState;
  const bst = (baselineFrame?.layer_states.mps ?? {}) as MpsState;
  const bondDims = st.bond_dims ?? [];
  const entropies = st.entropies ?? [];
  const hasData = bondDims.length > 0;

  const totalS = (st.entropies ?? []).reduce(
    (a: number, v: number | null) => a + (v ?? 0),
    0,
  );
  const baselineTotalS = baselineFrame
    ? (bst.entropies ?? []).reduce(
        (a: number, v: number | null) => a + (v ?? 0),
        0,
      )
    : null;
  const chiMax = bondDims.length > 0 ? Math.max(...bondDims) : null;

  const readouts = (
    <PanelReadouts
      cells={[
        { label: 'N', value: st.n_sites ?? '—' },
        { label: 'd_local', value: st.d_local ?? '—' },
        {
          label: 'total S',
          value: totalS.toFixed(3),
          baselineValue: baselineTotalS,
        },
        { label: 'χ_max', value: chiMax ?? '—' },
      ]}
    />
  );

  const metricsStrip = (
    <MetricsStrip
      layer="mps"
      metrics={[
        {
          key: 'totalS',
          label: 'total S',
          color: '#9aedc1',
          select: (ls) => {
            const ents = (ls.entropies ?? []) as (number | null)[];
            return ents.reduce(
              (a: number, v: number | null) => a + (v ?? 0),
              0,
            );
          },
        },
      ]}
    />
  );

  return (
    <PanelShell
      title="MPS — tensor chain + entanglement"
      step={frame.step}
      meta={
        st.n_sites != null
          ? `${st.n_sites} sites · d=${st.d_local ?? '?'}`
          : undefined
      }
      hasData={hasData}
      readouts={readouts}
      metricsStrip={metricsStrip}
    >
      <FrameInterpreter layer="mps" />
      <div
        style={{ display: 'flex', flexDirection: 'column', height: '100%' }}
      >
        <div style={{ flex: '0 0 45%', minHeight: 0 }}>
          <TensorChain bondDims={bondDims} nSites={st.n_sites} />
        </div>
        <div style={{ flex: '1 1 55%', minHeight: 0 }}>
          {entropies.length > 0 ? (
            <EntropyCurve entropies={entropies} bondDims={bondDims} />
          ) : (
            <div className="viz-panel__empty">No entropy data.</div>
          )}
        </div>
      </div>
    </PanelShell>
  );
}
