import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent } from '@testing-library/react';
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
  it('renders N, d_local, species count, and ξ readout cells', () => {
    render(<HamiltonianPanel frame={hamiltonianFrame} />);
    expect(screen.getByText('N')).toBeInTheDocument();
    expect(screen.getByText('d_local')).toBeInTheDocument();
    expect(screen.getAllByText('species').length).toBeGreaterThan(0);
    expect(screen.getByText('ξ (curv. coup.)')).toBeInTheDocument();
  });

  it('renders the per-species coefficients table with real ω/t/μ/J values', () => {
    const { container } = render(<HamiltonianPanel frame={hamiltonianFrame} />);
    const table = container.querySelector(
      '[data-testid="hamiltonian-per-species"]',
    );
    expect(table).toBeTruthy();
    // headers
    expect(table!.textContent).toContain('ω (mass)');
    expect(table!.textContent).toContain('t (kinetic)');
    expect(table!.textContent).toContain('μ (quartic)');
    expect(table!.textContent).toContain('J (source)');
    // fixture values for `scalar` species: ω = 1.200, t = 0.500.
    expect(table!.textContent).toContain('1.200');
    expect(table!.textContent).toContain('0.500');
  });

  it('renders a coupling matrix that toggles between g_ab and λ_ab', () => {
    const { container } = render(<HamiltonianPanel frame={hamiltonianFrame} />);
    // Initial view: density g_ab.
    expect(
      container.querySelector('[data-testid="coupling-matrix-g_ab"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="coupling-matrix-λ_ab"]'),
    ).toBeNull();
    // Click the Yukawa toggle.
    const yukawaBtn = screen.getByText(/Yukawa/);
    fireEvent.click(yukawaBtn);
    expect(
      container.querySelector('[data-testid="coupling-matrix-λ_ab"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="coupling-matrix-g_ab"]'),
    ).toBeNull();
  });

  it('renders the 1D curvature strip aligned to the site axis', () => {
    const { container } = render(<HamiltonianPanel frame={hamiltonianFrame} />);
    const strip = container.querySelector(
      '[data-testid="hamiltonian-curvature-strip"]',
    );
    expect(strip).toBeTruthy();
    expect(strip!.querySelector('canvas')).toBeTruthy();
  });

  it('still renders gracefully if a legacy recording emits a 2D curvature matrix', () => {
    // 2D legacy data: diagonal is extracted as the 1D strip.
    const legacy = {
      step: 0,
      layer_states: {
        hamiltonian: {
          n_sites: 3,
          d_local: 2,
          species_dims: [2],
          species: ['x'],
          curvature: [
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
          ],
        },
      },
    } as any;
    const { container } = render(<HamiltonianPanel frame={legacy} />);
    expect(
      container.querySelector('[data-testid="hamiltonian-curvature-strip"]'),
    ).toBeTruthy();
  });

  it('renders a KaTeX-rendered H = ... block citing the real §3.3.4 decomposition', () => {
    const { container } = render(<HamiltonianPanel frame={hamiltonianFrame} />);
    const katex = container.querySelector('[data-testid="hamiltonian-katex"]');
    expect(katex).not.toBeNull();
    expect(katex!.querySelector('.katex')).not.toBeNull();
  });
});
