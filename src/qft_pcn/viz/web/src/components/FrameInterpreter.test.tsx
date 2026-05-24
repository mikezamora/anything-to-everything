// src/qft_pcn/viz/web/src/components/FrameInterpreter.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FrameInterpreter } from './FrameInterpreter';
import { useVizStore } from '../store';
import { INTERPRETERS } from '../lib/interpreters';

beforeEach(() => useVizStore.getState().resetAll());

describe('FrameInterpreter', () => {
  it('renders nothing when there is no active run', () => {
    const { container } = render(<FrameInterpreter layer="manifold" />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when the interpreter returns null', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.setActiveRun('A');
    s.pushFrame('A', { step: 0,
                       layer_states: { manifold: { /* no mean_abs_ricci */ } } });
    const { container } = render(<FrameInterpreter layer="manifold" />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the interpretation text when interpreter returns output', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.setActiveRun('A');
    s.pushFrame('A', { step: 12,
      layer_states: { manifold: { mean_abs_ricci: 0.55 } } });
    render(<FrameInterpreter layer="manifold" />);
    expect(screen.getByText(/curvature is concentrating/i))
      .toBeInTheDocument();
    expect(screen.getByText(/Step 12/)).toBeInTheDocument();
  });

  it('exposes all registered interpreters as callable functions', () => {
    for (const [, fn] of Object.entries(INTERPRETERS)) {
      expect(typeof fn).toBe('function');
      // Each must handle empty state without throwing:
      expect(fn({}, 0)).toBeNull();
    }
  });

  it('every interpreter returns null for an empty layer_state without throwing', () => {
    for (const [key, fn] of Object.entries(INTERPRETERS)) {
      expect(fn({}, 0), `${key} should return null for empty state`).toBeNull();
    }
  });

  it('every interpreter has a citation pointing to a learn-route article id', () => {
    // Citations are exercised in T9 article-coverage; here we just ensure
    // every interpreter that returns output sets a citation.
    for (const [key, fn] of Object.entries(INTERPRETERS)) {
      // Stress with a generously populated state — most return non-null.
      const out = fn({
        mean_abs_ricci: 0.3, mean_abs_coupling: 0.3,
        bond_dims: [4], entropies: [0.5],
        n_sites: 4, d_local: 2, species: ['A'],
        energy: -1.5, n_leaves: 4, layer_dims: [2, 2],
        n_qubits: 3, n_layers: 2,
        total_energy: 0.5, forall_protected_leaves: [],
        trotter_steps: 10, layers: [{}, {}],
        total_free_energy: 1.2,
        mean_abs_stress_energy: 0.01,
      }, 1);
      if (out !== null) {
        expect(out.citation, `${key} output should carry a citation`).toBeTruthy();
      }
    }
  });
});
