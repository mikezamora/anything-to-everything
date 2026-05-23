import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MeraRelaxPanel } from './MeraRelaxPanel';
import { meraRelaxFrame, emptyFrame } from './__fixtures__/frames';

describe('MeraRelaxPanel', () => {
  it('mounts with a frame', () => {
    render(<MeraRelaxPanel frame={meraRelaxFrame} />);
  });

  it('mounts with an empty layer state', () => {
    render(<MeraRelaxPanel frame={emptyFrame} />);
  });

  it('accepts an optional baselineFrame prop (no-op)', () => {
    render(
      <MeraRelaxPanel frame={meraRelaxFrame} baselineFrame={meraRelaxFrame} />,
    );
    expect(screen.getByText('n_leaves')).toBeInTheDocument();
  });

  it('renders the core readouts', () => {
    render(<MeraRelaxPanel frame={meraRelaxFrame} />);
    expect(screen.getByText('total energy')).toBeInTheDocument();
    expect(screen.getByText('n_leaves')).toBeInTheDocument();
    expect(screen.getByText('∀-protected')).toBeInTheDocument();
  });

  it('renders the AST text in a <pre> block', () => {
    const { container } = render(<MeraRelaxPanel frame={meraRelaxFrame} />);
    const pre = container.querySelector('[data-testid="mera-relax-ast-text"]');
    expect(pre).toBeTruthy();
    expect(pre!.tagName.toLowerCase()).toBe('pre');
    expect(pre!.textContent).toContain('forall');
  });

  it('renders the forall-protected leaf indices', () => {
    const { container } = render(<MeraRelaxPanel frame={meraRelaxFrame} />);
    const block = container.querySelector(
      '[data-testid="mera-relax-forall-leaves"]',
    );
    expect(block).toBeTruthy();
    expect(block!.textContent).toContain('0, 5, 17');
  });

  it('renders per-term residuals as a table', () => {
    const { container } = render(<MeraRelaxPanel frame={meraRelaxFrame} />);
    const table = container.querySelector(
      '[data-testid="mera-relax-residuals"]',
    );
    expect(table).toBeTruthy();
    expect(table!.textContent).toContain('R-AddZero');
    expect(table!.textContent).toContain('R-Eq-Refl');
  });
});
