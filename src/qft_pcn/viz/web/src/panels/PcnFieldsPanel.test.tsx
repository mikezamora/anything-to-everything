import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PcnFieldsPanel } from './PcnFieldsPanel';

// `snapshot_pcn_fields` emits phi/E/Pi as 3D `(channels, Nx, Ny)`. The fixture
// mirrors the live wire shape so the panel render path is exercised the way
// the production snapshot pipeline feeds it.
const frame = {
  step: 3,
  layer_states: {
    'pcn-fields': {
      layers: [
        { phi: [[[0, 0], [0, 0]]], E: [[[0, 0], [0, 0]]],
          Pi: [[[1, 1], [1, 1]]], channels: 1 },
        { phi: [[[0.1, 0.1], [0.1, 0.1]]], E: [[[0, 0], [0, 0]]],
          Pi: [[[1, 1], [1, 1]]], channels: 1 },
      ],
      step: 3,
    },
  },
} as any;

describe('PcnFieldsPanel', () => {
  it('renders one card per PCN layer', () => {
    render(<PcnFieldsPanel frame={frame} />);
    expect(screen.getAllByText(/layer/i).length).toBeGreaterThanOrEqual(2);
  });
  it('renders the layer-depth readout', () => {
    render(<PcnFieldsPanel frame={frame} />);
    expect(screen.getByText(/depth/i)).toBeInTheDocument();
  });
  it('mounts with empty layer state without crashing', () => {
    render(<PcnFieldsPanel frame={{ step: 0, layer_states: {} } as any} />);
    expect(screen.getByText(/PCN Fields/)).toBeInTheDocument();
  });
  // Regression: with the live 3D `(C, Nx, Ny)` wire shape, `<Heatmap>` must
  // still receive a 2D grid (via as2DGrid) — otherwise rows/cols collapse and
  // the panel falls back to the "—" placeholder for every layer.
  it('renders an <svg> heatmap per layer when phi is 3D (channels, Nx, Ny)', () => {
    const { container } = render(<PcnFieldsPanel frame={frame} />);
    const svgs = container.querySelectorAll('svg');
    expect(svgs.length).toBeGreaterThanOrEqual(2);
    // Each SVG should contain heatmap cells.
    expect(svgs[0].querySelectorAll('rect').length).toBeGreaterThan(0);
  });
});
