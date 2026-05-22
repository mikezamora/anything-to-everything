/**
 * Manifold panel — a Three.js warped grid mesh whose vertex height and colour
 * come from the metric / Ricci-curvature fields, with toggleable Phi / E / Pi
 * height-surfaces from the first layer's fields, and a Plotly inset showing
 * the mean-absolute-curvature scalar over the run.
 *
 * Reads `frame.layer_states.manifold` (shape: `snapshot_network`).
 */

import { useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
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

export function ManifoldPanel({ frame }: { frame: Frame }) {
  const st = (frame.layer_states.manifold ?? {}) as ManifoldState;
  const [overlay, setOverlay] = useState<'phi' | 'E' | 'Pi' | null>(null);

  // Base warped grid: height = h_xx, colour = ricci.
  const baseHeight = st.metric_h?.h_xx ?? st.ricci ?? null;
  const baseColor = st.ricci ?? st.metric_h?.h_xx ?? null;
  const field = st.fields?.[0];
  const overlayGrid =
    overlay === 'phi'
      ? field?.phi
      : overlay === 'E'
        ? field?.E
        : overlay === 'Pi'
          ? field?.Pi
          : null;

  const hasData = !!baseHeight && !!baseColor;

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
    >
      <div style={{ position: 'absolute', inset: 0 }}>
        <div
          style={{
            position: 'absolute',
            zIndex: 2,
            top: 8,
            left: 8,
            display: 'flex',
            gap: 4,
          }}
        >
          {(['phi', 'E', 'Pi'] as const).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setOverlay((cur) => (cur === k ? null : k))}
              style={{
                background: overlay === k ? '#2b3a6b' : '#161b29',
                color: '#c8d0e0',
                border: '1px solid #2a3450',
                borderRadius: 4,
                padding: '2px 8px',
                fontSize: 11,
                cursor: 'pointer',
              }}
              disabled={!field}
            >
              {k}
            </button>
          ))}
        </div>
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
