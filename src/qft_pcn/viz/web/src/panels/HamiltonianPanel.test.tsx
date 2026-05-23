import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { HamiltonianPanel } from './HamiltonianPanel';
import { hamiltonianFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<HamiltonianPanel frame={hamiltonianFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<HamiltonianPanel frame={emptyFrame} />);
});

describe('HamiltonianPanel readouts', () => {
  it('renders N, d_local, and species count readout cells', () => {
    render(<HamiltonianPanel frame={hamiltonianFrame} />);
    expect(screen.getByText('N')).toBeInTheDocument();
    expect(screen.getByText('d_local')).toBeInTheDocument();
    expect(screen.getByText('species')).toBeInTheDocument();

    // species count = 2 (scalar, gauge) from the fixture.
    const speciesCell = screen.getByText('species').parentElement;
    expect(speciesCell).not.toBeNull();
    expect(speciesCell!.textContent).toContain('2');
  });

  it('does not crash when a baseline frame with same-shape curvature is supplied (mini-map renders the diff)', () => {
    // Same shape, halved values -> diff matrix is well-defined.
    const baseline = {
      step: 0,
      layer_states: {
        hamiltonian: {
          n_sites: 4,
          d_local: 2,
          species_dims: [2, 2],
          species: ['scalar', 'gauge'],
          curvature: [
            [0.0, 0.1, -0.05, 0.025],
            [0.1, 0.0, 0.15, -0.1],
            [-0.05, 0.15, 0.0, 0.05],
            [0.025, -0.1, 0.05, 0.0],
          ],
        },
      },
    };
    const { container } = render(
      <HamiltonianPanel
        frame={hamiltonianFrame}
        baselineFrame={baseline}
      />,
    );
    // Mini-map canvas still present.
    expect(
      container.querySelector('canvas[aria-label="curvature mini-map"]'),
    ).toBeTruthy();
  });

  it('falls back to the current curvature when baseline shape differs', () => {
    const baseline = {
      step: 0,
      layer_states: {
        hamiltonian: {
          curvature: [
            [0.0, 0.2],
            [0.2, 0.0],
          ],
        },
      },
    };
    const { container } = render(
      <HamiltonianPanel
        frame={hamiltonianFrame}
        baselineFrame={baseline}
      />,
    );
    // Still renders the mini-map (silent fallback).
    expect(
      container.querySelector('canvas[aria-label="curvature mini-map"]'),
    ).toBeTruthy();
  });

  it('renders a KaTeX-rendered H = ... block', () => {
    const { container } = render(<HamiltonianPanel frame={hamiltonianFrame} />);
    const katex = container.querySelector('[data-testid="hamiltonian-katex"]');
    expect(katex).not.toBeNull();
    // KaTeX injects a `.katex` span when rendering succeeds.
    expect(katex!.querySelector('.katex')).not.toBeNull();
  });
});
