/**
 * VQC panel — left: a D3 circuit diagram, one wire per qubit, with a rotation
 * gate per layer annotated with its live (theta_y, theta_z) angles plus
 * entangling links; right: Three.js Bloch spheres, one per qubit, oriented by
 * that qubit's last-layer rotation angles, with a Z-readout bar beneath.
 *
 * Reads `frame.layer_states.vqc` (shape: `snapshot_vqc`).
 */

import { useEffect, useMemo, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import * as d3 from 'd3';
import * as THREE from 'three';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { useSize } from './common';
import { FrameInterpreter } from '../components/FrameInterpreter';

interface VqcState {
  /** theta shape: n_layers x n_qubits x 2 */
  theta?: number[][][] | null;
  n_qubits?: number | null;
  n_layers?: number | null;
  input_scale?: number | null;
  bias?: number[] | null;
}

/** D3 circuit: qubit wires, rotation-gate boxes, nearest-neighbour CNOT links. */
function Circuit({ theta }: { theta: number[][][] }) {
  const [ref, size] = useSize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const nLayers = theta.length;
    const nQubits = theta[0]?.length ?? 0;
    if (nQubits === 0) return;
    const { width, height } = size;
    const mx = 50;
    const my = 28;
    const yOf = (q: number) =>
      my + (q / Math.max(1, nQubits - 1)) * (height - 2 * my);
    const xOf = (l: number) =>
      mx + ((l + 0.5) / Math.max(1, nLayers)) * (width - 2 * mx);

    // wires
    for (let q = 0; q < nQubits; q++) {
      svg
        .append('line')
        .attr('x1', mx)
        .attr('y1', yOf(q))
        .attr('x2', width - mx)
        .attr('y2', yOf(q))
        .attr('stroke', '#3a4660');
      svg
        .append('text')
        .attr('x', 8)
        .attr('y', yOf(q) + 3)
        .attr('fill', '#7f8bb0')
        .attr('font-size', 10)
        .text(`q${q}`);
    }

    // Layer axis tick labels along the top.
    for (let l = 0; l < nLayers; l++) {
      svg
        .append('text')
        .attr('x', xOf(l))
        .attr('y', 14)
        .attr('fill', '#7f8bb0')
        .attr('font-size', 9)
        .attr('text-anchor', 'middle')
        .text(`L${l}`);
    }
    // Axis legend caption.
    svg
      .append('text')
      .attr('x', mx)
      .attr('y', height - 6)
      .attr('fill', '#5f6b86')
      .attr('font-size', 9)
      .text('x = layer · y = qubit · gate = R_y(θ_y) R_z(θ_z)');

    for (let l = 0; l < nLayers; l++) {
      for (let q = 0; q < nQubits; q++) {
        const [ty = 0, tz = 0] = theta[l][q] ?? [];
        const g = svg
          .append('g')
          .attr('transform', `translate(${xOf(l)},${yOf(q)})`);
        g.append('rect')
          .attr('x', -20)
          .attr('y', -13)
          .attr('width', 40)
          .attr('height', 26)
          .attr('rx', 3)
          .attr('fill', '#1b2336')
          .attr('stroke', '#5fd0c8');
        g.append('text')
          .attr('fill', '#c8d0e0')
          .attr('font-size', 8)
          .attr('text-anchor', 'middle')
          .attr('dy', -2)
          .text(`y${ty.toFixed(2)}`);
        g.append('text')
          .attr('fill', '#9aa6c8')
          .attr('font-size', 8)
          .attr('text-anchor', 'middle')
          .attr('dy', 8)
          .text(`z${tz.toFixed(2)}`);
      }
      // entangling links between adjacent qubits.
      for (let q = 0; q < nQubits - 1; q++) {
        svg
          .append('line')
          .attr('x1', xOf(l) + 22)
          .attr('y1', yOf(q))
          .attr('x2', xOf(l) + 22)
          .attr('y2', yOf(q + 1))
          .attr('stroke', '#d0a05f')
          .attr('stroke-width', 1.5);
      }
    }
  }, [theta, size]);

  return (
    <div ref={ref} style={{ width: '100%', height: '100%' }}>
      <svg ref={svgRef} width={size.width} height={size.height} />
    </div>
  );
}

/** A single Bloch sphere oriented by (theta_y, theta_z). */
function Bloch({
  pos,
  thetaY,
  thetaZ,
}: {
  pos: [number, number, number];
  thetaY: number;
  thetaZ: number;
}) {
  // Bloch vector from spherical-ish angles: polar = thetaY, azimuth = thetaZ.
  const v = useMemo(() => {
    const r = 0.8;
    return new THREE.Vector3(
      r * Math.sin(thetaY) * Math.cos(thetaZ),
      r * Math.cos(thetaY),
      r * Math.sin(thetaY) * Math.sin(thetaZ),
    );
  }, [thetaY, thetaZ]);

  return (
    <group position={pos}>
      <mesh>
        <sphereGeometry args={[0.8, 24, 24]} />
        <meshStandardMaterial
          color="#2a3450"
          transparent
          opacity={0.25}
          wireframe
        />
      </mesh>
      <mesh position={[v.x / 2, v.y / 2, v.z / 2]}>
        <cylinderGeometry args={[0.03, 0.03, v.length(), 8]} />
        <meshBasicMaterial color="#5fd0c8" />
      </mesh>
      <mesh position={[v.x, v.y, v.z]}>
        <sphereGeometry args={[0.09, 12, 12]} />
        <meshBasicMaterial color="#5fd0c8" />
      </mesh>
    </group>
  );
}

export function VqcPanel({
  frame,
  baselineFrame: _baselineFrame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.vqc ?? {}) as VqcState;
  const theta = st.theta ?? [];
  const nQubits = st.n_qubits ?? theta[0]?.length ?? 0;
  const hasData = theta.length > 0 && nQubits > 0;

  // Last-layer angles drive the Bloch spheres.
  const last = theta[theta.length - 1] ?? [];

  return (
    <PanelShell
      title="VQC — circuit + Bloch spheres"
      step={frame.step}
      meta={
        hasData
          ? `${nQubits} qubits · ${st.n_layers ?? theta.length} layers`
          : undefined
      }
      hasData={hasData}
      emptyMessage="No VQC substrate active — start a run with the 'vqc' layer."
    >
      <FrameInterpreter layer="vqc" />
      <div className="viz-panel__split" style={{ height: '100%' }}>
        <Circuit theta={theta} />
        <div style={{ position: 'relative', minWidth: 0, minHeight: 0 }}>
          <div style={{ position: 'absolute', inset: 0 }}>
          {hasData && (
            <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
              <color attach="background" args={['#0b0e14']} />
              <ambientLight intensity={0.7} />
              <directionalLight position={[3, 4, 5]} intensity={0.6} />
              {Array.from({ length: nQubits }, (_, q) => {
                const [ty = 0, tz = 0] = last[q] ?? [];
                const cols = Math.ceil(Math.sqrt(nQubits));
                const row = Math.floor(q / cols);
                const col = q % cols;
                return (
                  <Bloch
                    key={q}
                    pos={[
                      (col - (cols - 1) / 2) * 2,
                      ((cols - 1) / 2 - row) * 2,
                      0,
                    ]}
                    thetaY={ty}
                    thetaZ={tz}
                  />
                );
              })}
            </Canvas>
          )}
          </div>
          {/* Bloch-sphere axis HUD: explain the projection. */}
          <div
            data-testid="vqc-bloch-hud"
            style={{
              position: 'absolute',
              left: 6,
              top: 4,
              fontSize: 10,
              color: '#7f8bb0',
              pointerEvents: 'none',
              lineHeight: 1.4,
            }}
          >
            <div>x = sin θ_y cos θ_z</div>
            <div>y = cos θ_y &nbsp; z = sin θ_y sin θ_z</div>
            <div>last-layer angles</div>
          </div>
        </div>
      </div>
    </PanelShell>
  );
}
