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
