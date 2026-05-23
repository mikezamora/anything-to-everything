import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { QpcnPanel } from './QpcnPanel';
import { qpcnFrame, emptyFrame } from './__fixtures__/frames';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

it('mounts with a frame', () => {
  render(<QpcnPanel frame={qpcnFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<QpcnPanel frame={emptyFrame} />);
});

describe('QpcnPanel readouts', () => {
  it('renders the energy readout', () => {
    render(<QpcnPanel frame={qpcnFrame} />);
    expect(screen.getByText('energy')).toBeInTheDocument();
    // qpcnFrame energy = -1.732
    expect(screen.getByText('-1.7320')).toBeInTheDocument();
  });

  it('renders a pred-error table with obs + error columns', () => {
    render(<QpcnPanel frame={qpcnFrame} />);
    expect(screen.getByText('obs')).toBeInTheDocument();
    expect(screen.getByText('error')).toBeInTheDocument();
    // Each fixture pred_errors key should appear as a row.
    for (const k of Object.keys(qpcnFrame.layer_states.qpcn!.pred_errors as Record<string, number>)) {
      expect(screen.getByText(k)).toBeInTheDocument();
    }
  });

  it('renders an energy Δ and a Δ column in the pred-error table when a baseline frame is supplied', () => {
    // qpcnFrame.energy = -1.732; baseline -1.5 -> -0.232 (improvement, green).
    const baseline = {
      step: 0,
      layer_states: {
        qpcn: {
          energy: -1.5,
          pred_errors: { phi: 0.3, E: 0.05 },
          params: {},
        },
      },
    };
    const { container } = render(
      <QpcnPanel frame={qpcnFrame} baselineFrame={baseline} />,
    );
    const eDelta = container.querySelector('.qpcn-energy-delta') as HTMLElement;
    expect(eDelta).toBeTruthy();
    expect(eDelta.textContent).toContain('-0.2320');
    // pred-error Δ column present.
    const errDeltas = container.querySelectorAll('.qpcn-error-delta');
    expect(errDeltas.length).toBeGreaterThan(0);
  });

  it('hides live-param sliders by default (not paused)', () => {
    const s = useVizStore.getState();
    s.openRun('R1');
    useVizStore.getState().setActiveRun('R1');
    const { queryByTestId } = render(<QpcnPanel frame={qpcnFrame} />);
    expect(queryByTestId('qpcn-param-sliders')).toBeNull();
  });

  it('renders sliders for writable QPCN params when paused on an active run', () => {
    const s = useVizStore.getState();
    s.openRun('R1');
    useVizStore.getState().setActiveRun('R1');
    useVizStore.getState().setPaused(true);
    const { getByTestId } = render(<QpcnPanel frame={qpcnFrame} />);
    expect(getByTestId('qpcn-param-sliders')).toBeInTheDocument();
    // fixture has `mass` as a writable suffix; `coupling`/`hopping` are not.
    expect(getByTestId('qpcn-slider-mass')).toBeInTheDocument();
  });

  it('renders MetricsStrip paths for energy + each learnable param once enough frames are pushed', () => {
    let s = useVizStore.getState();
    s.openRun('A');
    s = useVizStore.getState();
    s.setActiveRun('A');
    for (let i = 0; i < 5; i++) {
      s = useVizStore.getState();
      s.pushFrame('A', {
        step: i,
        layer_states: {
          qpcn: {
            energy: -1.0 - 0.1 * i,
            pred_errors: { phi: 0.2 - 0.01 * i, E: 0.05 },
            params: { mass: 1.0 + 0.01 * i, coupling: 0.3 - 0.005 * i, hopping: -0.5 },
            bond_dims: [1, 2, 2, 1],
            entropies: [0.0, 0.1, 0.05],
            occupations: [1.0, 1.0, 1.0],
            step: i,
          },
        },
      });
    }
    const { container } = render(<QpcnPanel frame={qpcnFrame} />);
    // One path per series: energy + 3 params = 4.
    const paths = container.querySelectorAll('.metrics-strip path');
    expect(paths.length).toBeGreaterThanOrEqual(4);
  });
});
