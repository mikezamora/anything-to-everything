import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { MeraRelaxPanel } from './MeraRelaxPanel';
import { meraRelaxFrame, emptyFrame } from './__fixtures__/frames';
import { useVizStore } from '../store';
import type { Frame } from '../lib/types';

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

  describe('Recent firings (step-replay annotator wiring)', () => {
    beforeEach(() => {
      useVizStore.getState().resetAll();
    });

    function makeFrame(step: number, addZeroValue: number): Frame {
      return {
        step,
        layer_states: {
          mera_relax: {
            total_energy: 0.01,
            residuals: [
              { rule_id: 'R-AddZero', site: 4, value: addZeroValue },
            ],
            n_leaves: 80,
            layer_bond_dims: [16, 16],
            forall_protected_leaves: [0],
            ast_text: 'forall x:Nat. x',
            step,
          },
        },
      };
    }

    it('renders the firings list when frames push declining residuals', () => {
      const store = useVizStore.getState();
      store.openRun('run-A');
      store.setActiveRun('run-A');
      store.pushFrame('run-A', makeFrame(0, 0.4));
      store.pushFrame('run-A', makeFrame(1, 0.4));
      const last = makeFrame(2, 0.05);
      store.pushFrame('run-A', last);

      const { container } = render(<MeraRelaxPanel frame={last} />);
      const block = container.querySelector(
        '[data-testid="mera-relax-recent-firings"]',
      );
      expect(block).toBeTruthy();
      expect(block!.textContent).toContain('R-AddZero fired at site 4');
    });

    it('omits the firings block when there is no active run', () => {
      const { container } = render(<MeraRelaxPanel frame={meraRelaxFrame} />);
      expect(
        container.querySelector('[data-testid="mera-relax-recent-firings"]'),
      ).toBeNull();
    });
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
