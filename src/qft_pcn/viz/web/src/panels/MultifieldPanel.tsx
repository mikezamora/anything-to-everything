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
import { useSize, normGrid, sequential, diverging } from './common';

type Grid = number[][];

interface FieldEntry {
  phi?: Grid;
  E?: Grid;
  Pi?: Grid;
}

interface MultifieldState {
  fields?: Record<string, FieldEntry> | null;
  couplings?: Record<string, number> | null;
  step?: number | null;
}

const SPECIES_HUE = ['#5fd0c8', '#d0a05f', '#a05fd0', '#5f8fd0', '#d05f8f'];

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

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    if (!names.length) return;
    const { width, height } = size;

    interface Node extends d3.SimulationNodeDatum {
      id: string;
    }
    const nodes: Node[] = names.map((id) => ({ id }));
    const links = Object.entries(couplings)
      .map(([key, g]) => {
        const [a, b] = key.split('|');
        return { source: a, target: b, g };
      })
      .filter((l) => names.includes(l.source) && names.includes(l.target));

    const maxG = d3.max(links, (l) => Math.abs(l.g)) ?? 1;
    const sim = d3
      .forceSimulation<Node>(nodes)
      .force(
        'link',
        d3
          .forceLink<Node, (typeof links)[number]>(links)
          .id((d) => d.id)
          .distance(90),
      )
      .force('charge', d3.forceManyBody().strength(-220))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .stop();
    for (let i = 0; i < 200; i++) sim.tick();

    const link = svg
      .append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (l) => diverging(l.g / maxG))
      .attr('stroke-width', (l) => 1 + (Math.abs(l.g) / maxG) * 6)
      .attr('x1', (l) => (l.source as unknown as Node).x ?? 0)
      .attr('y1', (l) => (l.source as unknown as Node).y ?? 0)
      .attr('x2', (l) => (l.target as unknown as Node).x ?? 0)
      .attr('y2', (l) => (l.target as unknown as Node).y ?? 0);
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
          (((l.source as unknown as Node).x ?? 0) +
            ((l.target as unknown as Node).x ?? 0)) /
          2,
      )
      .attr(
        'y',
        (l) =>
          (((l.source as unknown as Node).y ?? 0) +
            ((l.target as unknown as Node).y ?? 0)) /
          2,
      )
      .text((l) => l.g.toFixed(2));

    const node = svg
      .append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    node
      .append('circle')
      .attr('r', 14)
      .attr('fill', (_d, i) => SPECIES_HUE[i % SPECIES_HUE.length])
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
  }, [names, couplings, size]);

  return (
    <div ref={ref} style={{ width: '100%', height: '100%' }}>
      <svg ref={svgRef} width={size.width} height={size.height} />
    </div>
  );
}

export function MultifieldPanel({ frame }: { frame: Frame }) {
  const st = (frame.layer_states.multifield ?? {}) as MultifieldState;
  const fields = st.fields ?? {};
  const names = Object.keys(fields);
  const hasData = names.length > 0;

  return (
    <PanelShell
      title="Multifield — species surfaces + couplings"
      step={st.step ?? frame.step}
      meta={`${names.length} fields`}
      hasData={hasData}
    >
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
                    color={sequential(i / Math.max(1, names.length - 1))}
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
