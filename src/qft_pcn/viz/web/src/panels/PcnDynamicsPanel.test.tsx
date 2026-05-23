import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { PcnDynamicsPanel } from './PcnDynamicsPanel';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

const dynFrame = {
  step: 4,
  layer_states: {
    'pcn-dynamics': {
      total_free_energy: 1.234,
      per_layer_free_energy: [0.5, 0.4, 0.334],
      per_layer_e_norm: [0.9, 0.6, 0.3],
      per_layer_pi_mean: [1.0, 1.1, 1.2],
      n_layers: 3,
      step: 4,
    },
  },
} as any;

const emptyDynFrame = { step: 0, layer_states: {} } as any;

describe('PcnDynamicsPanel', () => {
  it('mounts with a frame', () => {
    render(<PcnDynamicsPanel frame={dynFrame} />);
  });

  it('mounts with an empty layer state', () => {
    render(<PcnDynamicsPanel frame={emptyDynFrame} />);
  });

  it('renders the total F readout', () => {
    render(<PcnDynamicsPanel frame={dynFrame} />);
    expect(screen.getByText('total F')).toBeInTheDocument();
  });

  it('renders one row per PCN layer', () => {
    const { container } = render(<PcnDynamicsPanel frame={dynFrame} />);
    const bodyRows = container.querySelectorAll('.pcn-dynamics-table tbody tr');
    expect(bodyRows.length).toBe(3);
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
          'pcn-dynamics': {
            total_free_energy: 1.0 - i * 0.1,
            per_layer_free_energy: [0.5, 0.5 - i * 0.05],
            per_layer_e_norm: [0.5, 0.4],
            per_layer_pi_mean: [1.0, 1.1],
            n_layers: 2,
          },
        },
      });
    }
    const { container } = render(<PcnDynamicsPanel frame={dynFrame} />);
    expect(container.querySelectorAll('path').length).toBeGreaterThanOrEqual(1);
  });
});
