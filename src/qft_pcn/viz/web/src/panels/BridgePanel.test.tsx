import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { BridgePanel } from './BridgePanel';
import { bridgeFrame, emptyFrame } from './__fixtures__/frames';

describe('BridgePanel', () => {
  it('mounts with a frame', () => {
    render(<BridgePanel frame={bridgeFrame} />);
  });

  it('mounts with an empty layer state', () => {
    render(<BridgePanel frame={emptyFrame} />);
  });

  it('renders the core readouts', () => {
    render(<BridgePanel frame={bridgeFrame} />);
    expect(screen.getByText('trotter steps')).toBeInTheDocument();
    expect(screen.getByText('⟨H⟩')).toBeInTheDocument();
    expect(screen.getByText('converged')).toBeInTheDocument();
  });

  it('renders the solved_ast block (placeholder text when None)', () => {
    const { container } = render(<BridgePanel frame={bridgeFrame} />);
    const block = container.querySelector('[data-testid="bridge-solved-ast"]');
    expect(block).toBeTruthy();
    expect(block!.textContent).toContain('not populated');
  });

  it('renders embedded MPS miniature when present', () => {
    const { container } = render(<BridgePanel frame={bridgeFrame} />);
    const mps = container.querySelector(
      '[data-testid="bridge-mps-miniature"]',
    );
    expect(mps).toBeTruthy();
    expect(mps!.textContent).toContain('χ:');
  });

  it('renders embedded Hamiltonian miniature when present', () => {
    const { container } = render(<BridgePanel frame={bridgeFrame} />);
    const ham = container.querySelector(
      '[data-testid="bridge-hamiltonian-miniature"]',
    );
    expect(ham).toBeTruthy();
    expect(ham!.textContent).toContain('species:');
  });
});
