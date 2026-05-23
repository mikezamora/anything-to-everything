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
import { diverging, normGrid } from './common';

type Grid = number[][];

interface ManifoldState {
  metric_h?: { h_xx?: Grid; h_xy?: Grid; h_yy?: Grid } | null;
  ricci?: Grid | null;
  fields?: Array<{ phi?: Grid; E?: Grid; Pi?: Grid; channels?: number }> | null;
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
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.manifold ?? {}) as ManifoldState;
  const [overlay, setOverlay] = useState<'phi' | 'E' | 'Pi' | null>(null);
  const [channel, setChannel] = useState(0);

  // Base warped grid: height = h_xx, colour = ricci.
  const baseHeight = st.metric_h?.h_xx ?? st.ricci ?? null;
  const baseColor = st.ricci ?? st.metric_h?.h_xx ?? null;
  const field = st.fields?.[0];
  const nChannels = field?.channels ?? 1;
  const overlayGrid =
    overlay === 'phi'
      ? field?.phi
      : overlay === 'E'
        ? field?.E
        : overlay === 'Pi'
          ? field?.Pi
          : null;

  const hasData = !!baseHeight && !!baseColor;

  const toolbarItems: ToolbarItem[] = (['phi', 'E', 'Pi'] as const).map((k) => ({
    key: k,
    label: k,
    active: overlay === k,
    onToggle: () => setOverlay((cur) => (cur === k ? null : k)),
    disabled: !field,
  }));

  const toolbar = (
    <>
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
          value:
            st.mean_abs_ricci != null ? st.mean_abs_ricci.toFixed(3) : null,
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
