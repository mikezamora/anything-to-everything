/**
 * MERA panel — a binary MERA tree laid out on the Poincaré disk (Three.js,
 * orthographic). Leaves sit on the disk boundary; one coarse-graining node
 * is drawn per layer ring inward. Each layer's RG step contains BOTH
 * disentanglers (intra-pair `u` + inter-pair `u_inter`) AND a 2->1 isometry
 * `w` (Vidal 2008, §11.4); the substrate `snapshot_mera` field surfaces the
 * isometry tensors only, so we draw one isometry-glyph per coarse node and
 * do NOT invent alternating "disentangler" layers (deviation D-6). Each
 * tree edge's width tracks the bond dimension of its layer.
 *
 * Below the disk: a readout strip (leaves, layers, max χ) and an inline-SVG
 * entropy-vs-cut line over `st.entropies`.
 *
 * Reads `frame.layer_states.mera` (shape: `snapshot_mera`).
 */

import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';

interface MeraState {
  n_leaves?: number | null;
  layer_dims?: number[] | null;
  bond_dims?: number[] | null;
  entropies?: Array<number | null> | null;
  iso_residuals?: number[] | null;
}

/** Tiny per-layer sparkline of isometry-violation residuals (jsdom-safe SVG). */
function IsoResidualSparkline({ values }: { values: number[] }) {
  if (values.length === 0) return null;
  const W = 120;
  const H = 28;
  const pad = 3;
  const maxV = Math.max(1e-30, ...values);
  const n = values.length;
  const xAt = (i: number) =>
    pad + (n === 1 ? (W - 2 * pad) / 2 : (i / (n - 1)) * (W - 2 * pad));
  const yAt = (v: number) => H - pad - (v / maxV) * (H - 2 * pad);
  const d = values
    .map((v, i) =>
      `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(2)} ${yAt(v).toFixed(2)}`,
    )
    .join(' ');
  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="iso residuals per layer"
      style={{ display: 'inline-block', verticalAlign: 'middle' }}
    >
      <path d={d} fill="none" stroke="#ef9090" strokeWidth={1.5} />
    </svg>
  );
}

export interface MNode {
  x: number;
  y: number;
  depth: number;
  /** Every coarse-graining node is a 2->1 isometry (substrate exposes
   * `isometries` per layer; intra- and inter-pair disentanglers exist on
   * the substrate side but are not surfaced as separate nodes here — see
   * D-6 in visualizer-DEVIATIONS.md for why the prior alternating
   * "isometry / disentangler" layer scheme was incorrect). */
  kind: 'leaf' | 'isometry';
  bond: number;
}
export interface MEdge {
  a: MNode;
  b: MNode;
  width: number;
}

/**
 * Place nodes on concentric circles: the boundary circle holds the leaves,
 * each coarser layer sits on a smaller-radius circle. Every coarse node is
 * a 2->1 isometry (D-6: the substrate exposes `isometries` per layer; the
 * prior alternating "isometry / disentangler" scheme by layer parity was
 * not present in the substrate and misrepresented MERA topology).
 */
export function buildTree(
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
    // Every coarse-graining node is a 2->1 isometry. The architecture's
    // disentanglers exist within the same RG step on the substrate but
    // are not drawn as separate nodes here (D-6).
    const cur = ring(count, radius, layer + 1, 'isometry', bond);
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
        // Leaf = small disc; coarse-graining isometry = triangle glyph.
        // No "disentangler" glyph — see D-6.
        const color = n.kind === 'leaf' ? '#5fd0c8' : '#d0a05f';
        const r = n.kind === 'leaf' ? 0.025 : 0.05;
        return (
          <mesh key={`n${i}`} position={[n.x, n.y, 0.1]}>
            {n.kind === 'isometry' ? (
              <circleGeometry args={[r, 3]} />
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

/** Tiny inline-SVG line plot of entropy-vs-cut. Pure SVG so it works in jsdom. */
function EntropyCutLine({ entropies }: { entropies: Array<number | null> }) {
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
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i).toFixed(2)} ${yAt(v).toFixed(2)}`)
    .join(' ');

  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="entropy vs cut"
      style={{ display: 'block' }}
    >
      <path d={d} fill="none" stroke="#5fd0c8" strokeWidth={1.5} />
      <text x={pad} y={10} fill="#7f8bb0" fontSize={9}>
        S vs cut
      </text>
    </svg>
  );
}

export function MeraPanel({
  frame,
  baselineFrame,
}: {
  frame: Frame;
  /**
   * Optional baseline frame. When present the panel renders a 2-up side-by-
   * side layout: active disk on the left, baseline on the right; each shrunk
   * to half-width. 3D structural views don't admit a pixel-wise diff, so the
   * comparison is visual rather than computed.
   */
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.mera ?? {}) as MeraState;
  const bst = (baselineFrame?.layer_states.mera ?? {}) as MeraState;
  const nLeaves = st.n_leaves ?? 0;
  const layerDims = (st.layer_dims as number[] | null) ?? [];
  const bondDims = (st.bond_dims as number[] | null) ?? [];
  const entropies = (st.entropies ?? []) as Array<number | null>;
  const isoResiduals = (st.iso_residuals ?? []) as number[];
  const isoMax =
    isoResiduals.length > 0 ? Math.max(...isoResiduals) : null;
  const hasData = nLeaves > 0;
  const showCompare = !!baselineFrame && (bst.n_leaves ?? 0) > 0;
  const baseLayerDims = (bst.layer_dims as number[] | null) ?? [];
  const baseBondDims = (bst.bond_dims as number[] | null) ?? [];
  const baseEntropies = (bst.entropies ?? []) as Array<number | null>;

  const readouts = (
    <PanelReadouts
      cells={[
        {
          label: 'leaves',
          value: st.n_leaves ?? '—',
          baselineValue: bst.n_leaves ?? null,
        },
        {
          label: 'layers',
          value: layerDims.length,
          baselineValue: baselineFrame ? baseLayerDims.length : null,
        },
        {
          label: 'max χ',
          value: bondDims.length > 0 ? Math.max(...bondDims) : '—',
          baselineValue: baseBondDims.length > 0 ? Math.max(...baseBondDims) : null,
        },
        {
          label: 'iso err (max)',
          value: isoMax == null ? '—' : isoMax.toExponential(2),
        },
      ]}
    />
  );

  return (
    <PanelShell
      title="MERA — Poincaré-disk tree"
      step={frame.step}
      meta={
        hasData
          ? `${nLeaves} leaves · ${layerDims.length} layers`
          : undefined
      }
      hasData={hasData}
      readouts={readouts}
    >
      {hasData && (
        <div
          data-testid="mera-layout"
          data-compare={showCompare ? 'side-by-side' : 'single'}
          style={{ display: 'flex', flexDirection: 'column', height: '100%' }}
        >
          <div
            style={{
              flex: '1 1 auto',
              minHeight: 0,
              display: 'flex',
              flexDirection: 'row',
            }}
          >
            <div
              data-testid="mera-disk-active"
              style={{
                flex: showCompare ? '1 1 50%' : '1 1 100%',
                minWidth: 0,
                minHeight: 0,
              }}
            >
              <MeraScene
                nLeaves={nLeaves}
                layerDims={layerDims}
                bondDims={bondDims}
              />
            </div>
            {showCompare && (
              <div
                data-testid="mera-disk-baseline"
                style={{ flex: '1 1 50%', minWidth: 0, minHeight: 0 }}
              >
                <MeraScene
                  nLeaves={bst.n_leaves ?? 0}
                  layerDims={baseLayerDims}
                  bondDims={baseBondDims}
                />
              </div>
            )}
          </div>
          {(entropies.length > 0 || (showCompare && baseEntropies.length > 0)) && (
            <div
              style={{
                flex: '0 0 auto',
                padding: '4px 8px',
                display: 'flex',
                gap: 12,
              }}
            >
              {entropies.length > 0 && (
                <div
                  data-testid="mera-entropy-active"
                  style={{
                    flex: showCompare ? '1 1 50%' : '1 1 100%',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                  }}
                >
                  <EntropyCutLine entropies={entropies} />
                  {isoResiduals.length > 0 && (
                    <span
                      data-testid="mera-iso-sparkline"
                      title="‖W†W − I‖ per MERA layer"
                    >
                      <IsoResidualSparkline values={isoResiduals} />
                    </span>
                  )}
                </div>
              )}
              {showCompare && baseEntropies.length > 0 && (
                <div
                  data-testid="mera-entropy-baseline"
                  style={{ flex: '1 1 50%' }}
                >
                  <EntropyCutLine entropies={baseEntropies} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </PanelShell>
  );
}
