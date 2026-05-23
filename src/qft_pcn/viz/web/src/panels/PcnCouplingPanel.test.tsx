import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { PcnCouplingPanel } from './PcnCouplingPanel';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

const couplingFrame = {
  step: 7,
  layer_states: {
    'pcn-coupling': {
      kappa_R: 0.125,
      mean_abs_stress_energy: 0.5,
      mean_abs_ricci: 0.321,
      qpcn_observable_energy: 0.75,
      step: 7,
    },
  },
} as any;

const smallCouplingFrame = {
  step: 1,
  layer_states: {
    'pcn-coupling': {
      kappa_R: 0.125,
      mean_abs_stress_energy: 0.0,
      mean_abs_ricci: 0.0,
      qpcn_observable_energy: 0.0,
      step: 1,
    },
  },
} as any;

const emptyCouplingFrame = { step: 0, layer_states: {} } as any;

describe('PcnCouplingPanel', () => {
  it('mounts with a frame', () => {
    render(<PcnCouplingPanel frame={couplingFrame} />);
  });

  it('mounts with an empty layer state', () => {
    render(<PcnCouplingPanel frame={emptyCouplingFrame} />);
  });

  it('renders the PCN->QFT and QFT->PCN arrows', () => {
    render(<PcnCouplingPanel frame={couplingFrame} />);
    expect(screen.getByTestId('arrow-pcn-to-qft')).toBeInTheDocument();
    expect(screen.getByTestId('arrow-qft-to-pcn')).toBeInTheDocument();
  });

  it('arrow widths respond to fixture magnitudes', () => {
    const { rerender } = render(<PcnCouplingPanel frame={smallCouplingFrame} />);
    const upSmall = Number(
      screen.getByTestId('arrow-pcn-to-qft').getAttribute('stroke-width'),
    );
    const downSmall = Number(
      screen.getByTestId('arrow-qft-to-pcn').getAttribute('stroke-width'),
    );

    rerender(<PcnCouplingPanel frame={couplingFrame} />);
    const upLarge = Number(
      screen.getByTestId('arrow-pcn-to-qft').getAttribute('stroke-width'),
    );
    const downLarge = Number(
      screen.getByTestId('arrow-qft-to-pcn').getAttribute('stroke-width'),
    );

    expect(upLarge).toBeGreaterThan(upSmall);
    expect(downLarge).toBeGreaterThan(downSmall);
  });

  it('renders readouts (κ_R, mean |T|, mean |R|, ⟨H⟩)', () => {
    render(<PcnCouplingPanel frame={couplingFrame} />);
    expect(screen.getByText('κ_R')).toBeInTheDocument();
    expect(screen.getByText('mean |T|')).toBeInTheDocument();
    expect(screen.getByText('mean |R|')).toBeInTheDocument();
    // ⟨H⟩ appears twice: as a readout cell label AND as the down-arrow
    // SVG label (D-11 fix renamed the arrow from ⟨O⟩ to ⟨H⟩ to match the
    // actual driver = q._last_energy).
    expect(screen.getAllByText('⟨H⟩').length).toBeGreaterThanOrEqual(1);
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
          'pcn-coupling': {
            kappa_R: 0.1 + i * 0.01,
            mean_abs_stress_energy: 0.1 + i * 0.05,
            mean_abs_ricci: 0.2 + i * 0.03,
            qpcn_observable_energy: 0.5 - i * 0.05,
          },
        },
      });
    }
    const { container } = render(<PcnCouplingPanel frame={couplingFrame} />);
    expect(container.querySelectorAll('path').length).toBeGreaterThanOrEqual(1);
  });
});
