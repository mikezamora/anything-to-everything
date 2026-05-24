/**
 * Multifield panel — left: stacked Three.js surfaces, one per field species,
 * coloured by the field's Phi values; right: a D3 force-directed coupling
 * graph where each node is a species and each edge's width / colour encodes
 * the coupling strength g_ij.
 *
 * Reads `frame.layer_states.multifield` (shape: `snapshot_multifield`).
 */

import { useEffect, useMemo, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import * as d3 from 'd3';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';
import { useSize, normGrid, speciesColor, diverging } from './common';
import { FrameInterpreter } from '../components/FrameInterpreter';

type Grid = number[][];

const norm2 = (g?: number[][] | null) =>
  !g ? 0 : Math.sqrt(g.flat().reduce((a, v) => a + (v || 0) ** 2, 0));

interface FieldEntry {
  phi?: Grid;
  E?: Grid;
  Pi?: Grid;
}

interface MultifieldState {
  fields?: Record<string, FieldEntry> | null;
  couplings?: Record<string, number> | null;
  mean_abs_coupling?: number | null;
  step?: number | null;
}

function FieldSurface({
  grid,
  y,
  color,
}: {
  grid: Grid;
  y: number;
  color: string;
}) {
  const geo = useMemo(() => {
    const rows = grid.length;
    const cols = grid[0]?.length ?? 0;
    const g = new THREE.PlaneGeometry(3, 3, cols - 1, rows - 1);
    const pos = g.attributes.position as THREE.BufferAttribute;
    const { norm } = normGrid(grid);
    for (let i = 0; i < pos.count; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      pos.setZ(i, (norm[r]?.[c] ?? 0) * 0.8);
    }
    g.computeVertexNormals();
    return g;
  }, [grid]);
  // Dispose the old GPU buffer when a new frame replaces this geometry.
  useEffect(() => () => geo.dispose(), [geo]);
  return (
    <mesh geometry={geo} rotation={[-Math.PI / 2, 0, 0]} position={[0, y, 0]}>
      <meshStandardMaterial
        color={color}
        transparent
        opacity={0.85}
        side={THREE.DoubleSide}
        flatShading
      />
    </mesh>
  );
}

/** D3 force graph of species coupled by g_ij. */
function CouplingGraph({
  names,
  couplings,
}: {
  names: string[];
  couplings: Record<string, number>;
}) {
  const [ref, size] = useSize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  interface Node extends d3.SimulationNodeDatum {
    id: string;
  }

  // Force layout: run the 200-tick simulation ONLY when the graph topology
  // (names / couplings) changes. Positions are computed in a unit-square
  // [0,1]^2 frame so they can be cheaply rescaled on resize without re-running.
  const layout = useMemo(() => {
    const nodes: Node[] = names.map((id) => ({ id }));
    const links = Object.entries(couplings)
      .map(([key, g]) => {
        const [a, b] = key.split('|');
        return { source: a, target: b, g };
      })
      .filter((l) => names.includes(l.source) && names.includes(l.target));
    if (!names.length) return { nodes, links };

    const sim = d3
      .forceSimulation<Node>(nodes)
      .force(
        'link',
        d3
          .forceLink<Node, (typeof links)[number]>(links)
          .id((d) => d.id)
          .distance(0.25),
      )
      .force('charge', d3.forceManyBody().strength(-0.6))
      .force('center', d3.forceCenter(0.5, 0.5))
      .stop();
    for (let i = 0; i < 200; i++) sim.tick();
    sim.stop();
    return { nodes, links };
  }, [names, couplings]);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (!names.length) return;
    const { width, height } = size;
    const { nodes, links } = layout;

    // Rescale the unit-square layout into the current pixel viewport.
    const pad = 28;
    const px = (u: number | undefined) =>
      pad + (u ?? 0.5) * (width - 2 * pad);
    const py = (u: number | undefined) =>
      pad + (u ?? 0.5) * (height - 2 * pad);

    const maxG = d3.max(links, (l) => Math.abs(l.g)) ?? 1;

    const link = svg
      .append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (l) => diverging(l.g / maxG))
      .attr('stroke-width', (l) => 1 + (Math.abs(l.g) / maxG) * 6)
      .attr('x1', (l) => px((l.source as unknown as Node).x))
      .attr('y1', (l) => py((l.source as unknown as Node).y))
      .attr('x2', (l) => px((l.target as unknown as Node).x))
      .attr('y2', (l) => py((l.target as unknown as Node).y));
    void link;

    svg
      .append('g')
      .selectAll('text.edge')
      .data(links)
      .join('text')
      .attr('class', 'edge')
      .attr('fill', '#7f8bb0')
      .attr('font-size', 10)
      .attr('text-anchor', 'middle')
      .attr(
        'x',
        (l) =>
          (px((l.source as unknown as Node).x) +
            px((l.target as unknown as Node).x)) /
          2,
      )
      .attr(
        'y',
        (l) =>
          (py((l.source as unknown as Node).y) +
            py((l.target as unknown as Node).y)) /
          2,
      )
      .text((l) => l.g.toFixed(2));

    const node = svg
      .append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('transform', (d) => `translate(${px(d.x)},${py(d.y)})`);
    node
      .append('circle')
      .attr('r', 14)
      .attr('fill', (d) => speciesColor(d.id, names))
      .attr('stroke', '#0b0e14')
      .attr('stroke-width', 2);
    node
      .append('text')
      .attr('fill', '#0b0e14')
      .attr('font-size', 9)
      .attr('font-weight', 700)
      .attr('text-anchor', 'middle')
      .attr('dy', 3)
      .text((d) => d.id.slice(0, 4));
  }, [names, layout, size]);

  return (
    <div ref={ref} style={{ width: '100%', height: '100%' }}>
      <svg ref={svgRef} width={size.width} height={size.height} />
    </div>
  );
}

export function MultifieldPanel({
  frame,
  baselineFrame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.multifield ?? {}) as MultifieldState;
  const bst = (baselineFrame?.layer_states.multifield ?? {}) as MultifieldState;
  const fields = st.fields ?? {};
  const baseFields = bst.fields ?? {};
  const names = Object.keys(fields);
  const hasData = names.length > 0;

  // Per-pair signed coupling cells + time-series traces. Architectural
  // diagnostic §2.2 / §4.5: correlated pairs grow their coupling;
  // uncorrelated pairs stay near zero. mean|g| alone hides per-pair
  // structure and sign (D-5), so surface every (a,b) entry directly.
  const couplings = st.couplings ?? {};
  const couplingPairKeys = Object.keys(couplings);
  const pairCells = couplingPairKeys.map((pk) => {
    const v = couplings[pk];
    const signed = `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;
    return {
      label: `g[${pk}]`,
      value: signed,
      baselineValue: bst.couplings?.[pk] ?? null,
    };
  });

  const readouts = (
    <PanelReadouts
      cells={[
        {
          label: 'mean |g|',
          value: st.mean_abs_coupling?.toFixed(3),
          baselineValue: bst.mean_abs_coupling ?? null,
          highlightId: 'mean_abs_coupling',
        },
        ...pairCells,
        ...names.map((name) => {
          const baseNorm = baseFields[name]?.phi
            ? norm2(baseFields[name]!.phi)
            : null;
          return {
            label: `‖${name}.Φ‖₂`,
            value: norm2(fields[name]?.phi).toFixed(3),
            baselineValue: baseNorm,
          };
        }),
      ]}
    />
  );

  // Distinct colours per pair so each trace reads clearly against mean|g|.
  const pairPalette = ['#9aedc1', '#fbc66a', '#ef9090', '#6cd0ff', '#d291ff', '#5fd0c8'];
  const pairMetrics = couplingPairKeys.map((pk, i) => ({
    key: `g:${pk}`,
    label: `g[${pk}]`,
    color: pairPalette[i % pairPalette.length],
    select: (ls: Record<string, unknown>) =>
      (ls.couplings as Record<string, number> | undefined)?.[pk],
  }));

  const metricsStrip = (
    <MetricsStrip
      layer="multifield"
      metrics={[
        {
          key: 'mag',
          label: 'mean|g|',
          color: '#fbc66a',
          select: (ls) => ls.mean_abs_coupling as number,
        },
        ...pairMetrics,
      ]}
    />
  );

  return (
    <PanelShell
      title="Multifield — species surfaces + couplings"
      step={st.step ?? frame.step}
      meta={`${names.length} fields`}
      hasData={hasData}
      readouts={readouts}
      metricsStrip={metricsStrip}
    >
      <FrameInterpreter layer="multifield" />
      <div className="viz-panel__split" style={{ height: '100%' }}>
        <div style={{ position: 'relative' }}>
          {hasData && (
            <Canvas camera={{ position: [3, 3, 4], fov: 50 }}>
              <color attach="background" args={['#0b0e14']} />
              <ambientLight intensity={0.7} />
              <directionalLight position={[4, 6, 3]} intensity={0.7} />
              {names.map((name, i) => {
                const phi = fields[name]?.phi;
                if (!phi) return null;
                return (
                  <FieldSurface
                    key={name}
                    grid={phi}
                    y={(i - (names.length - 1) / 2) * 1.3}
                    color={speciesColor(name, names)}
                  />
                );
              })}
              <OrbitControls enablePan={false} />
            </Canvas>
          )}
        </div>
        <CouplingGraph names={names} couplings={st.couplings ?? {}} />
      </div>
    </PanelShell>
  );
}
