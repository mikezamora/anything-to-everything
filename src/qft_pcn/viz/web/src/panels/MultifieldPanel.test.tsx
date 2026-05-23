import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { MultifieldPanel } from './MultifieldPanel';
import { multifieldFrame, emptyFrame } from './__fixtures__/frames';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

it('mounts with a frame', () => {
  render(<MultifieldPanel frame={multifieldFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<MultifieldPanel frame={emptyFrame} />);
});

describe('MultifieldPanel readouts', () => {
  it('renders mean |g| readout', () => {
    render(<MultifieldPanel frame={multifieldFrame} />);
    expect(screen.getByText('mean |g|')).toBeInTheDocument();
  });

  it('renders one per-field strength row per field name', () => {
    render(<MultifieldPanel frame={multifieldFrame} />);
    const names = Object.keys(
      (multifieldFrame.layer_states.multifield as { fields: object }).fields,
    );
    for (const name of names) {
      expect(screen.getByText(`‖${name}.Φ‖₂`)).toBeInTheDocument();
    }
  });

  it('renders a Δ next to mean |g| when a baseline frame is supplied', () => {
    // multifieldFrame.mean_abs_coupling = 0.35; baseline 0.30 -> +0.050.
    const baseline = {
      ...multifieldFrame,
      layer_states: {
        ...multifieldFrame.layer_states,
        multifield: {
          ...(multifieldFrame.layer_states.multifield as object),
          mean_abs_coupling: 0.3,
        },
      },
    };
    const { container } = render(
      <MultifieldPanel frame={multifieldFrame} baselineFrame={baseline} />,
    );
    const deltas = container.querySelectorAll('.panel-readout-delta');
    expect(deltas.length).toBeGreaterThan(0);
    expect(deltas[0].textContent).toContain('+0.050');
  });

  it('renders a MetricsStrip path once enough frames are pushed', () => {
    let s = useVizStore.getState();
    s.openRun('A');
    s = useVizStore.getState();
    s.setActiveRun('A');
    for (let i = 0; i < 5; i++) {
      s = useVizStore.getState();
      s.pushFrame('A', {
        step: i,
        layer_states: { multifield: { mean_abs_coupling: i * 0.1 } },
      });
    }
    const { container } = render(<MultifieldPanel frame={multifieldFrame} />);
    expect(container.querySelectorAll('path').length).toBeGreaterThanOrEqual(1);
  });
});
