/**
 * Hamiltonian panel — a D3 term-coupling heatmap. The `curvature` matrix is a
 * site-by-site coupling map; each cell is coloured by its (curvature-weighted)
 * per-term energy contribution, with site species labels along the axes.
 *
 * Reads `frame.layer_states.hamiltonian` (shape: `snapshot_hamiltonian`).
 */

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { useSize } from './common';

interface HamiltonianState {
  n_sites?: number | null;
  d_local?: number | null;
  species_dims?: number[] | null;
  species?: string[] | null;
  curvature?: number[][] | number | null;
}

function Heatmap({
  matrix,
  species,
}: {
  matrix: number[][];
  species: string[];
}) {
  const [ref, size] = useSize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const n = matrix.length;
    if (n === 0) return;
    const { width, height } = size;
    const margin = { top: 28, right: 16, bottom: 16, left: 56 };
    const cell = Math.max(
      8,
      Math.min(
        (width - margin.left - margin.right) / n,
        (height - margin.top - margin.bottom) / n,
      ),
    );

    const flat = matrix.flat();
    const extent = d3.max(flat.map(Math.abs)) ?? 1;
    const color = d3
      .scaleSequential(d3.interpolateInferno)
      .domain([0, extent || 1]);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = matrix[i][j] ?? 0;
        g.append('rect')
          .attr('x', j * cell)
          .attr('y', i * cell)
          .attr('width', cell - 1)
          .attr('height', cell - 1)
          .attr('fill', color(Math.abs(v)))
          .append('title')
          .text(`(${i},${j}) energy ${v.toFixed(4)}`);
        if (cell > 26) {
          g.append('text')
            .attr('x', j * cell + cell / 2)
            .attr('y', i * cell + cell / 2)
            .attr('fill', Math.abs(v) / (extent || 1) > 0.5 ? '#000' : '#9aa6c8')
            .attr('font-size', 9)
            .attr('text-anchor', 'middle')
            .attr('dy', 3)
            .text(v.toFixed(2));
        }
      }
    }

    // axis labels — site index + species name.
    for (let i = 0; i < n; i++) {
      const label = species[i] ? `${i}·${species[i].slice(0, 3)}` : `${i}`;
      g.append('text')
        .attr('x', -6)
        .attr('y', i * cell + cell / 2)
        .attr('fill', '#7f8bb0')
        .attr('font-size', 9)
        .attr('text-anchor', 'end')
        .attr('dy', 3)
        .text(label);
      g.append('text')
        .attr('x', i * cell + cell / 2)
        .attr('y', -8)
        .attr('fill', '#7f8bb0')
        .attr('font-size', 9)
        .attr('text-anchor', 'middle')
        .text(i);
    }
  }, [matrix, species, size]);

  return (
    <div ref={ref} style={{ width: '100%', height: '100%' }}>
      <svg ref={svgRef} width={size.width} height={size.height} />
    </div>
  );
}

export function HamiltonianPanel({ frame }: { frame: Frame }) {
  const st = (frame.layer_states.hamiltonian ?? {}) as HamiltonianState;
  const curv = st.curvature;
  const matrix = Array.isArray(curv) && Array.isArray(curv[0])
    ? (curv as number[][])
    : null;
  const hasData = !!matrix && matrix.length > 0;

  return (
    <PanelShell
      title="Hamiltonian — term-energy heatmap"
      step={frame.step}
      meta={
        st.n_sites != null
          ? `${st.n_sites} sites · d=${st.d_local ?? '?'}`
          : undefined
      }
      hasData={hasData}
      emptyMessage="No curvature / term matrix for this Hamiltonian."
    >
      {hasData && <Heatmap matrix={matrix!} species={st.species ?? []} />}
    </PanelShell>
  );
}
