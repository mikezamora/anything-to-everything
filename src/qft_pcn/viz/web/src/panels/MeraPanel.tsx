/**
 * MERA panel — a binary MERA tree laid out on the Poincaré disk (Three.js,
 * orthographic). Leaves sit on the disk boundary; coarse-graining layers march
 * inward. Isometry nodes are drawn as triangles, disentanglers as squares, and
 * each tree edge's width tracks the bond dimension of its layer.
 *
 * Reads `frame.layer_states.mera` (shape: `snapshot_mera`).
 */

import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';

interface MeraState {
  n_leaves?: number | null;
  layer_dims?: number[] | null;
  bond_dims?: number[] | null;
  entropies?: Array<number | null> | null;
}

interface MNode {
  x: number;
  y: number;
  depth: number;
  kind: 'leaf' | 'isometry' | 'disentangler';
  bond: number;
}
interface MEdge {
  a: MNode;
  b: MNode;
  width: number;
}

/**
 * Place nodes on concentric circles: the boundary circle holds the leaves,
 * each coarser layer sits on a smaller-radius circle. Within a layer the nodes
 * alternate isometry / disentangler glyphs.
 */
function buildTree(
  nLeaves: number,
  layerDims: number[],
  bondDims: number[],
): { nodes: MNode[]; edges: MEdge[] } {
  const nodes: MNode[] = [];
  const edges: MEdge[] = [];
  if (nLeaves <= 0) return { nodes, edges };

  const maxBond = Math.max(1, ...bondDims, 1);
  const nLayers = Math.max(layerDims.length, 1);

  // Layer 0 = leaves on the boundary (radius ~1).
  let prev: MNode[] = [];
  const ring = (
    count: number,
    radius: number,
    depth: number,
    kind: MNode['kind'],
    bond: number,
  ): MNode[] => {
    const arr: MNode[] = [];
    for (let i = 0; i < count; i++) {
      const ang = (i / count) * Math.PI * 2 - Math.PI / 2;
      const n: MNode = {
        x: Math.cos(ang) * radius,
        y: Math.sin(ang) * radius,
        depth,
        kind,
        bond,
      };
      arr.push(n);
      nodes.push(n);
    }
    return arr;
  };

  prev = ring(nLeaves, 1.0, 0, 'leaf', 1);

  let count = nLeaves;
  for (let layer = 0; layer < nLayers && count > 1; layer++) {
    count = Math.max(1, Math.floor(count / 2));
    // Poincaré-style inward compression: radius shrinks toward the centre.
    const radius = 1.0 * Math.pow(0.55, layer + 1);
    const bond = bondDims[layer] ?? 1;
    const kind: MNode['kind'] =
      layer % 2 === 0 ? 'isometry' : 'disentangler';
    const cur = ring(count, radius, layer + 1, kind, bond);
    // connect each coarse node to two finer nodes.
    cur.forEach((node, i) => {
      const w = 0.6 + (bond / maxBond) * 4;
      const c1 = prev[(2 * i) % prev.length];
      const c2 = prev[(2 * i + 1) % prev.length];
      if (c1) edges.push({ a: node, b: c1, width: w });
      if (c2) edges.push({ a: node, b: c2, width: w });
    });
    prev = cur;
  }

  return { nodes, edges };
}

function MeraScene({ nLeaves, layerDims, bondDims }: {
  nLeaves: number;
  layerDims: number[];
  bondDims: number[];
}) {
  const { nodes, edges } = useMemo(
    () => buildTree(nLeaves, layerDims, bondDims),
    [nLeaves, layerDims, bondDims],
  );

  return (
    <Canvas
      orthographic
      camera={{ position: [0, 0, 10], zoom: 150, near: 0.1, far: 100 }}
    >
      <color attach="background" args={['#0b0e14']} />
      {/* Poincaré disk boundary */}
      <mesh>
        <ringGeometry args={[1.0, 1.01, 64]} />
        <meshBasicMaterial color="#2a3450" />
      </mesh>
      {edges.map((e, i) => (
        <Line
          key={`e${i}`}
          points={[
            [e.a.x, e.a.y, 0],
            [e.b.x, e.b.y, 0],
          ]}
          color="#4f7fd0"
          lineWidth={e.width}
        />
      ))}
      {nodes.map((n, i) => {
        const color =
          n.kind === 'leaf'
            ? '#5fd0c8'
            : n.kind === 'isometry'
              ? '#d0a05f'
              : '#a05fd0';
        // isometry -> triangle, disentangler -> square, leaf -> small circle.
        const r = n.kind === 'leaf' ? 0.025 : 0.05;
        return (
          <mesh key={`n${i}`} position={[n.x, n.y, 0.1]}>
            {n.kind === 'isometry' ? (
              <circleGeometry args={[r, 3]} />
            ) : n.kind === 'disentangler' ? (
              <circleGeometry args={[r, 4]} />
            ) : (
              <circleGeometry args={[r, 16]} />
            )}
            <meshBasicMaterial color={color} />
          </mesh>
        );
      })}
    </Canvas>
  );
}

export function MeraPanel({ frame }: { frame: Frame }) {
  const st = (frame.layer_states.mera ?? {}) as MeraState;
  const nLeaves = st.n_leaves ?? 0;
  const hasData = nLeaves > 0;

  return (
    <PanelShell
      title="MERA — Poincaré-disk tree"
      step={frame.step}
      meta={
        hasData
          ? `${nLeaves} leaves · ${(st.layer_dims ?? []).length} layers`
          : undefined
      }
      hasData={hasData}
    >
      {hasData && (
        <MeraScene
          nLeaves={nLeaves}
          layerDims={st.layer_dims ?? []}
          bondDims={st.bond_dims ?? []}
        />
      )}
    </PanelShell>
  );
}
