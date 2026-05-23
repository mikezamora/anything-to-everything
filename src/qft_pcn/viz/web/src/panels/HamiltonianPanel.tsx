/**
 * Hamiltonian panel — a D3 term-coupling heatmap. The `curvature` matrix is a
 * site-by-site coupling map; each cell is coloured by its (curvature-weighted)
 * per-term energy contribution. Axes are labelled by site index. `species` is
 * the Hamiltonian's *field-species* list (one entry per species, NOT per
 * site), shown as a separate legend.
 *
 * Uplift: PanelReadouts (N, d_local, species count), a KaTeX-rendered
 * canonical-form block `H = sum h_i + sum h_{ij}`, and a small 64x64
 * curvature mini-map painted via the `diverging()` ramp.
 *
 * Reads `frame.layer_states.hamiltonian` (shape: `snapshot_hamiltonian`).
 */

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';
import { useSize, tex, diverging } from './common';

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

    // axis labels — site index only (rows + columns are sites).
    for (let i = 0; i < n; i++) {
      g.append('text')
        .attr('x', -6)
        .attr('y', i * cell + cell / 2)
        .attr('fill', '#7f8bb0')
        .attr('font-size', 9)
        .attr('text-anchor', 'end')
        .attr('dy', 3)
        .text(i);
      g.append('text')
        .attr('x', i * cell + cell / 2)
        .attr('y', -8)
        .attr('fill', '#7f8bb0')
        .attr('font-size', 9)
        .attr('text-anchor', 'middle')
        .text(i);
    }

    // species legend — the field-species list (one entry per species).
    if (species.length > 0) {
      svg
        .append('text')
        .attr('x', width - 8)
        .attr('y', 14)
        .attr('fill', '#7f8bb0')
        .attr('font-size', 9)
        .attr('text-anchor', 'end')
        .text(`species: ${species.join(', ')}`);
    }
  }, [matrix, species, size]);

  return (
    <div ref={ref} style={{ width: '100%', height: '100%' }}>
      <svg ref={svgRef} width={size.width} height={size.height} />
    </div>
  );
}

/**
 * 64x64 curvature mini-map. Each canvas pixel block is filled by the
 * `diverging()` ramp applied to the matrix entry nearest to it (nearest-
 * neighbour upscaling for small N, downsampling otherwise).
 */
function CurvatureMiniMap({ matrix }: { matrix: number[][] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const SIZE = 64;

  useEffect(() => {
    const cnv = canvasRef.current;
    if (!cnv) return;
    const ctx = cnv.getContext('2d');
    if (!ctx) return;
    const n = matrix.length;
    if (n === 0) {
      ctx.clearRect(0, 0, SIZE, SIZE);
      return;
    }
    // Normalize by peak absolute value -> [-1, 1].
    let peak = 0;
    for (const row of matrix) {
      for (const v of row) {
        const a = Math.abs(v);
        if (Number.isFinite(a) && a > peak) peak = a;
      }
    }
    const scale = peak > 0 ? peak : 1;
    const cell = SIZE / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = matrix[i]?.[j] ?? 0;
        ctx.fillStyle = diverging(v / scale);
        ctx.fillRect(
          Math.floor(j * cell),
          Math.floor(i * cell),
          Math.ceil(cell),
          Math.ceil(cell),
        );
      }
    }
  }, [matrix]);

  return (
    <canvas
      ref={canvasRef}
      width={SIZE}
      height={SIZE}
      aria-label="curvature mini-map"
      style={{
        width: SIZE,
        height: SIZE,
        imageRendering: 'pixelated',
        border: '1px solid #2f3a55',
      }}
    />
  );
}

export function HamiltonianPanel({
  frame,
  baselineFrame: _baselineFrame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.hamiltonian ?? {}) as HamiltonianState;
  const curv = st.curvature;
  const matrix = Array.isArray(curv) && Array.isArray(curv[0])
    ? (curv as number[][])
    : null;
  const hasData = !!matrix && matrix.length > 0;
  const speciesNames = (st.species as string[] | null) ?? [];

  const readouts = (
    <PanelReadouts
      cells={[
        { label: 'N', value: st.n_sites ?? '—' },
        { label: 'd_local', value: st.d_local ?? '—' },
        { label: 'species', value: speciesNames.length },
      ]}
    />
  );

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
      readouts={readouts}
    >
      <div
        style={{ display: 'flex', flexDirection: 'column', height: '100%' }}
      >
        <div style={{ flex: '1 1 auto', minHeight: 0 }}>
          {hasData && <Heatmap matrix={matrix!} species={speciesNames} />}
        </div>
        <div
          style={{
            flex: '0 0 auto',
            display: 'flex',
            gap: 12,
            alignItems: 'center',
            padding: '8px 4px 0',
            borderTop: '1px solid #1c2230',
            marginTop: 6,
          }}
        >
          {hasData && <CurvatureMiniMap matrix={matrix!} />}
          <div style={{ flex: 1, minWidth: 0 }}>
            {speciesNames.length > 0 && (
              <table
                style={{
                  fontSize: 10,
                  color: '#9aa6c8',
                  borderCollapse: 'collapse',
                  marginBottom: 6,
                }}
              >
                <tbody>
                  {speciesNames.map((s, i) => (
                    <tr key={s}>
                      <td style={{ paddingRight: 8, color: '#7f8bb0' }}>{s}</td>
                      <td>d={st.species_dims?.[i] ?? '?'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <div
              data-testid="hamiltonian-katex"
              style={{ color: '#c8d0e0', fontSize: 12 }}
              dangerouslySetInnerHTML={{
                __html: tex(
                  'H = \\sum_i h_i + \\sum_{\\langle i,j \\rangle} h_{ij}',
                ),
              }}
            />
          </div>
        </div>
      </div>
    </PanelShell>
  );
}
