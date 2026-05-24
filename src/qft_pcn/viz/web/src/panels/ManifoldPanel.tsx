/**
 * Manifold panel — a Three.js warped grid mesh whose vertex height and colour
 * come from the metric / Ricci-curvature fields, with toggleable Phi / E / Pi
 * height-surfaces from the first layer's fields, and a Plotly inset showing
 * the mean-absolute-curvature scalar over the run.
 *
 * Reads `frame.layer_states.manifold` (shape: `snapshot_network`).
 */

import { useEffect, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelToolbar, type ToolbarItem } from './PanelToolbar';
import { PanelReadouts } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';
import {
  diverging,
  normGrid,
  as2DGrid,
  smallNumberFormat,
  type PhiLike,
} from './common';
import { FrameInterpreter } from '../components/FrameInterpreter';
import { ColorRampLegend } from './scales';

type Grid = number[][];

interface ManifoldState {
  metric_h?: { h_xx?: Grid; h_xy?: Grid; h_yy?: Grid } | null;
  ricci?: Grid | null;
  // `snapshot_network` emits per-layer phi/E/Pi as 3D `(channels, Nx, Ny)`
  // (Field.values is shape (C, Nx, Ny) and `_grid` just .tolist()s it). The
  // overlay surface needs 2D, so `as2DGrid(field.phi, channel)` collapses
  // the channel axis at the consumer site.
  fields?: Array<{ phi?: PhiLike; E?: PhiLike; Pi?: PhiLike; channels?: number }> | null;
  mean_abs_ricci?: number | null;
  step?: number | null;
}

/** Build a BufferGeometry from a height grid, colouring vertices by `colorGrid`. */
function gridGeometry(height: Grid, colorGrid: Grid): THREE.BufferGeometry {
  const rows = height.length;
  const cols = height[0]?.length ?? 0;
  const geo = new THREE.PlaneGeometry(4, 4, cols - 1, rows - 1);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const { norm: hN } = normGrid(height);
  const { norm: cN } = normGrid(colorGrid);
  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const r = Math.floor(i / cols);
    const c = i % cols;
    pos.setZ(i, (hN[r]?.[c] ?? 0) * 1.4);
    const col = new THREE.Color(diverging(cN[r]?.[c] ?? 0));
    colors[i * 3] = col.r;
    colors[i * 3 + 1] = col.g;
    colors[i * 3 + 2] = col.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  return geo;
}

function Surface({
  height,
  colorGrid,
  wireframe,
  opacity,
}: {
  height: Grid;
  colorGrid: Grid;
  wireframe?: boolean;
  opacity?: number;
}) {
  const geo = useMemo(
    () => gridGeometry(height, colorGrid),
    [height, colorGrid],
  );
  // Dispose the old GPU buffer when a new frame replaces this geometry.
  useEffect(() => () => geo.dispose(), [geo]);
  return (
    <mesh geometry={geo} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial
        vertexColors
        wireframe={wireframe}
        transparent={opacity !== undefined && opacity < 1}
        opacity={opacity ?? 1}
        side={THREE.DoubleSide}
        flatShading
      />
    </mesh>
  );
}

export function ManifoldPanel({
  frame,
  baselineFrame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.manifold ?? {}) as ManifoldState;
  const bst = (baselineFrame?.layer_states.manifold ?? {}) as ManifoldState;
  const [overlay, setOverlay] = useState<'phi' | 'E' | 'Pi' | null>(null);
  const [channel, setChannel] = useState(0);
  // Height channel for the warped surface. h_μν is a *tensor*; rendering
  // only h_xx hides the off-diagonal shear h_xy and the orthogonal h_yy
  // (§3.2). Allow the user to switch which component drives surface height,
  // or to view the mean curvature proxy `tr(h) = h_xx + h_yy`.
  const [heightChannel, setHeightChannel] = useState<
    'h_xx' | 'h_xy' | 'h_yy' | 'tr_h'
  >('h_xx');

  // Compose the requested height grid from the metric perturbation tensor.
  function pickHeight(mh: ManifoldState['metric_h']): Grid | null {
    if (!mh) return null;
    if (heightChannel === 'h_xx') return mh.h_xx ?? null;
    if (heightChannel === 'h_xy') return mh.h_xy ?? null;
    if (heightChannel === 'h_yy') return mh.h_yy ?? null;
    // tr(h) = h_xx + h_yy (mean-curvature proxy).
    const hxx = mh.h_xx;
    const hyy = mh.h_yy;
    if (!hxx || !hyy) return hxx ?? hyy ?? null;
    const rows = Math.min(hxx.length, hyy.length);
    if (rows === 0) return null;
    const out: number[][] = [];
    for (let i = 0; i < rows; i++) {
      const a = hxx[i] ?? [];
      const b = hyy[i] ?? [];
      const cols = Math.min(a.length, b.length);
      const row: number[] = [];
      for (let j = 0; j < cols; j++) row.push((a[j] ?? 0) + (b[j] ?? 0));
      out.push(row);
    }
    return out;
  }

  // Base warped grid: height = chosen tensor component, colour = ricci.
  // When a baseline frame is present and its ricci grid shares the same
  // shape as the current one, colour by (current - baseline) so the panel
  // renders a diff heatmap. Height stays from current — subtracting
  // heights would lose the warped-surface readability.
  const baseHeight = pickHeight(st.metric_h) ?? st.ricci ?? null;
  const curRicci = st.ricci ?? st.metric_h?.h_xx ?? null;
  const baseRicci = bst.ricci ?? bst.metric_h?.h_xx ?? null;
  const baseColor = useMemo<Grid | null>(() => {
    if (!curRicci) return null;
    if (
      !baselineFrame ||
      !baseRicci ||
      baseRicci.length !== curRicci.length ||
      baseRicci[0]?.length !== curRicci[0]?.length
    ) {
      return curRicci;
    }
    return curRicci.map((row, i) =>
      row.map((v, j) => v - (baseRicci[i]?.[j] ?? 0)),
    );
  }, [curRicci, baseRicci, baselineFrame]);
  const field = st.fields?.[0];
  const nChannels = field?.channels ?? 1;
  // Wire shape is 3D (C, Nx, Ny); collapse to the selected channel before
  // handing to the Three.js Surface (which builds a PlaneGeometry from the
  // 2D grid extents — a degenerate 1xN strip otherwise).
  const overlayRaw =
    overlay === 'phi'
      ? field?.phi
      : overlay === 'E'
        ? field?.E
        : overlay === 'Pi'
          ? field?.Pi
          : null;
  const overlayGrid = as2DGrid(overlayRaw, channel);

  const hasData = !!baseHeight && !!baseColor;

  const toolbarItems: ToolbarItem[] = (['phi', 'E', 'Pi'] as const).map((k) => ({
    key: k,
    label: k,
    active: overlay === k,
    onToggle: () => setOverlay((cur) => (cur === k ? null : k)),
    disabled: !field,
  }));

  const heightItems: ToolbarItem[] = (
    ['h_xx', 'h_xy', 'h_yy', 'tr_h'] as const
  ).map((k) => ({
    key: k,
    label: k === 'tr_h' ? 'tr(h)' : k,
    active: heightChannel === k,
    onToggle: () => setHeightChannel(k),
  }));

  const toolbar = (
    <>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
        <span style={{ fontSize: 11, color: '#7f8bb0' }}>height</span>
        <PanelToolbar items={heightItems} />
      </span>
      <PanelToolbar items={toolbarItems} />
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
        channel
        <select
          aria-label="channel"
          value={channel}
          onChange={(e) => setChannel(Number(e.target.value))}
        >
          {Array.from({ length: nChannels }, (_, i) => (
            <option key={i} value={i}>
              {i}
            </option>
          ))}
        </select>
      </label>
    </>
  );

  const gridSize = st.metric_h?.h_xx?.length;
  const readouts = (
    <PanelReadouts
      cells={[
        {
          label: 'mean |R|',
          value: smallNumberFormat(st.mean_abs_ricci),
          baselineValue: bst.mean_abs_ricci ?? null,
          highlightId: 'mean_abs_ricci',
        },
        { label: 'layers', value: st.fields?.length ?? 0 },
        { label: 'grid', value: gridSize ? `${gridSize}²` : '—' },
      ]}
    />
  );

  const metricsStrip = (
    <MetricsStrip
      layer="manifold"
      metrics={[
        {
          key: 'mar',
          label: 'mean|R|',
          color: '#6cd0ff',
          select: (ls) => ls.mean_abs_ricci as number,
        },
      ]}
    />
  );

  return (
    <PanelShell
      title="Manifold — warped metric grid"
      step={st.step ?? frame.step}
      meta={
        st.mean_abs_ricci != null
          ? `mean|Ricci| ${st.mean_abs_ricci.toFixed(3)}`
          : undefined
      }
      hasData={hasData}
      toolbar={toolbar}
      readouts={readouts}
      metricsStrip={metricsStrip}
    >
      <FrameInterpreter layer="manifold" />
      {/* Axis HUD + Ricci color-ramp legend over the R3F canvas. */}
      <div
        data-testid="manifold-axes-hud"
        style={{
          position: 'absolute',
          left: 8,
          top: 32,
          fontSize: 10,
          color: '#7f8bb0',
          pointerEvents: 'none',
          lineHeight: 1.4,
          zIndex: 2,
        }}
      >
        <div>x · y → grid site</div>
        <div>z = {heightChannel === 'tr_h' ? 'tr(h)' : heightChannel}</div>
        <div>colour = Ricci R</div>
      </div>
      {baseColor && (() => {
        let lo = Infinity;
        let hi = -Infinity;
        for (const row of baseColor) {
          for (const v of row) {
            if (Number.isFinite(v)) {
              if (v < lo) lo = v;
              if (v > hi) hi = v;
            }
          }
        }
        if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
        return (
          <div
            data-testid="manifold-ricci-legend"
            style={{
              position: 'absolute',
              right: 8,
              bottom: 8,
              pointerEvents: 'none',
              zIndex: 2,
              background: 'rgba(11, 14, 20, 0.7)',
              padding: 4,
              borderRadius: 3,
            }}
          >
            <ColorRampLegend
              min={lo}
              max={hi}
              ramp="diverging"
              label="Ricci R"
            />
          </div>
        );
      })()}
      <div style={{ position: 'absolute', inset: 0 }}>
        {hasData && (
          <Canvas camera={{ position: [3.5, 3.5, 3.5], fov: 50 }}>
            <color attach="background" args={['#0b0e14']} />
            <ambientLight intensity={0.6} />
            <directionalLight position={[5, 8, 3]} intensity={0.8} />
            <Surface height={baseHeight!} colorGrid={baseColor!} />
            {overlayGrid && (
              <group position={[0, 1.6, 0]}>
                <Surface
                  height={overlayGrid}
                  colorGrid={overlayGrid}
                  wireframe
                  opacity={0.7}
                />
              </group>
            )}
            <OrbitControls enablePan={false} />
          </Canvas>
        )}
      </div>
    </PanelShell>
  );
}
