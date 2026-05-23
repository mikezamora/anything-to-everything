import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { it, expect } from 'vitest';
import { LogicPanel } from './LogicPanel';
import { logicFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<LogicPanel frame={logicFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<LogicPanel frame={emptyFrame} />);
});

it('renders an empty-state message when no logic layer is in the frame', () => {
  render(<LogicPanel frame={{ step: 0, layer_states: {} } as any} />);
  expect(
    screen.getByText(/no logic substrate active/i),
  ).toBeInTheDocument();
});

it('renders terms and total_energy when fed a live frame', () => {
  const frame = {
    step: 1,
    layer_states: {
      logic: {
        n_sites: 6,
        term_count: 25,
        terms: [
          { rule_id: 'R-Beta', site: 0, arity: 2 },
          { rule_id: 'R-Arith-Pre', site: 1, arity: 2 },
        ],
        lambda_beta: 1.0,
        lambda_arith: 0.5,
        lambda_if: 0.75,
        residuals: [0.1, 0.05],
        total_energy: 0.15,
      },
    },
  } as any;
  render(<LogicPanel frame={frame} />);
  // header should show the ⟨H⟩ readout when total_energy is present.
  expect(screen.getByText(/⟨H⟩/)).toBeInTheDocument();
});
