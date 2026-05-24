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
      per_layer_kl: [0.41, 0.33, 0.27],
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

  it('renders the KL column when per_layer_kl is present', () => {
    const { container } = render(<PcnDynamicsPanel frame={dynFrame} />);
    const headers = Array.from(
      container.querySelectorAll('.pcn-dynamics-table thead th'),
    ).map((th) => th.textContent ?? '');
    expect(headers).toContain('KL');
    // The first row's KL cell renders the fixture's first KL entry.
    const firstRowCells = container.querySelectorAll(
      '.pcn-dynamics-table tbody tr:first-child td',
    );
    // Columns: layer | F | KL | ‖E‖₂ | mean Π → KL is the 3rd cell.
    expect(firstRowCells[2].textContent).toBe('0.410');
  });

  it('omits the KL column when per_layer_kl is absent', () => {
    const noKlFrame = {
      step: 1,
      layer_states: {
        'pcn-dynamics': {
          total_free_energy: 0.5,
          per_layer_free_energy: [0.5],
          per_layer_e_norm: [0.2],
          per_layer_pi_mean: [1.0],
          n_layers: 1,
        },
      },
    } as any;
    const { container } = render(<PcnDynamicsPanel frame={noKlFrame} />);
    const headers = Array.from(
      container.querySelectorAll('.pcn-dynamics-table thead th'),
    ).map((th) => th.textContent ?? '');
    expect(headers).not.toContain('KL');
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
