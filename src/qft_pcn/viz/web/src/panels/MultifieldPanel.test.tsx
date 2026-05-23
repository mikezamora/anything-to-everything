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

  it('renders a per-pair signed coupling cell for each (a,b) in couplings (D-5)', () => {
    render(<MultifieldPanel frame={multifieldFrame} />);
    // multifieldFrame has couplings: alpha|beta=+0.6, beta|gamma=-0.3, alpha|gamma=+0.15
    expect(screen.getByText('g[alpha|beta]')).toBeInTheDocument();
    expect(screen.getByText('g[beta|gamma]')).toBeInTheDocument();
    expect(screen.getByText('g[alpha|gamma]')).toBeInTheDocument();
    // Signed prefix renders so positive/negative direction is visible.
    expect(screen.getByText('+0.600')).toBeInTheDocument();
    expect(screen.getByText('-0.300')).toBeInTheDocument();
  });

  it('renders per-pair MetricsStrip traces (one per (a,b) coupling key) once enough frames are pushed (D-5)', () => {
    let s = useVizStore.getState();
    s.openRun('B');
    s = useVizStore.getState();
    s.setActiveRun('B');
    for (let i = 0; i < 5; i++) {
      s = useVizStore.getState();
      s.pushFrame('B', {
        step: i,
        layer_states: {
          multifield: {
            mean_abs_coupling: 0.1 + 0.02 * i,
            couplings: {
              'alpha|beta': 0.1 * i,
              'beta|gamma': -0.05 * i,
            },
          },
        },
      });
    }
    const { container } = render(<MultifieldPanel frame={multifieldFrame} />);
    // 1 mean|g| + N pair traces. multifieldFrame.couplings has 3 pairs ⇒ ≥4 paths.
    const paths = container.querySelectorAll('.metrics-strip path');
    expect(paths.length).toBeGreaterThanOrEqual(4);
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
