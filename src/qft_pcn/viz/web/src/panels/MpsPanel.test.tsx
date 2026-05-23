import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { MpsPanel } from './MpsPanel';
import { mpsFrame, emptyFrame } from './__fixtures__/frames';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

it('mounts with a frame', () => {
  render(<MpsPanel frame={mpsFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<MpsPanel frame={emptyFrame} />);
});

describe('MpsPanel readouts', () => {
  it('renders N, d_local, and total S readout cells', () => {
    render(<MpsPanel frame={mpsFrame} />);
    expect(screen.getByText('N')).toBeInTheDocument();
    expect(screen.getByText('d_local')).toBeInTheDocument();
    expect(screen.getByText('total S')).toBeInTheDocument();
  });

  it('renders a Δ next to total S when a baseline frame is supplied', () => {
    // mpsFrame.entropies sum ≈ 1.5; baseline frame with sum 1.0 -> +0.500.
    const baseline = {
      step: 0,
      layer_states: {
        mps: {
          bond_dims: [1, 2, 4, 4, 2, 1],
          entropies: [0.0, 0.2, 0.4, 0.3, 0.1],
          n_sites: 6,
          d_local: 2,
        },
      },
    };
    const { container } = render(
      <MpsPanel frame={mpsFrame} baselineFrame={baseline} />,
    );
    const deltas = container.querySelectorAll('.panel-readout-delta');
    expect(deltas.length).toBeGreaterThan(0);
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
        layer_states: {
          mps: {
            bond_dims: [1, 2, 2, 1],
            entropies: [0.0, i * 0.1, 0.05],
            n_sites: 4,
            d_local: 2,
          },
        },
      });
    }
    const { container } = render(<MpsPanel frame={mpsFrame} />);
    expect(container.querySelectorAll('path').length).toBeGreaterThanOrEqual(1);
  });
});
