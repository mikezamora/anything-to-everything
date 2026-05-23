import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PcnFieldsPanel } from './PcnFieldsPanel';

const frame = {
  step: 3,
  layer_states: {
    'pcn-fields': {
      layers: [
        { phi: [[0, 0], [0, 0]], E: [[0, 0], [0, 0]],
          Pi: [[1, 1], [1, 1]], channels: 1 },
        { phi: [[0.1, 0.1], [0.1, 0.1]], E: [[0, 0], [0, 0]],
          Pi: [[1, 1], [1, 1]], channels: 1 },
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
});
