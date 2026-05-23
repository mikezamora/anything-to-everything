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

  it('renders a KaTeX-rendered H = ... block', () => {
    const { container } = render(<HamiltonianPanel frame={hamiltonianFrame} />);
    const katex = container.querySelector('[data-testid="hamiltonian-katex"]');
    expect(katex).not.toBeNull();
    // KaTeX injects a `.katex` span when rendering succeeds.
    expect(katex!.querySelector('.katex')).not.toBeNull();
  });
});
