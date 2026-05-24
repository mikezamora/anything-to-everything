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

it('renders the real per-bond entropy chart when bond_entropies is supplied (D-4)', () => {
  // bond_entropies is the load-bearing §8.1 readout: binder = entanglement.
  // The chart MUST be rendered when this field is present so a viewer can
  // verify the soul invariant rather than the previous decorative arcs.
  const frame = {
    step: 1,
    layer_states: {
      logic: {
        n_sites: 4,
        term_count: 6,
        terms: [{ rule_id: 'R-Beta', site: 0, arity: 2 }],
        bond_entropies: [0.1, 0.4, 0.2],
        total_energy: 0.01,
      },
    },
  } as any;
  const { container } = render(<LogicPanel frame={frame} />);
  expect(
    container.querySelector('[data-testid="logic-bond-entropy"]'),
  ).toBeTruthy();
});

it('renders the fully-relaxed badge when all residuals are ~0 (Phase-1 H_eval eigenstate)', () => {
  // Default-preset behaviour: logic.beta-reduce starts in an H_eval
  // eigenstate so residuals are emitted as all-zero. Without the badge,
  // every term renders at min colour intensity and the panel reads as
  // frozen even though the substrate is correct.
  const frame = {
    step: 3,
    layer_states: {
      logic: {
        n_sites: 4,
        term_count: 2,
        terms: [
          { rule_id: 'R-Beta', site: 0, arity: 2 },
          { rule_id: 'R-Arith-Pre', site: 1, arity: 2 },
        ],
        residuals: [0, 0],
        total_energy: 0.0,
      },
    },
  } as any;
  render(<LogicPanel frame={frame} />);
  const badge = screen.getByTestId('logic-relaxed-badge');
  expect(badge).toBeInTheDocument();
  expect(badge).toHaveTextContent(/fully relaxed/i);
  expect(badge).toHaveTextContent(/H_eval is diagonal/);
});

it('does NOT render the fully-relaxed badge when at least one residual is driving', () => {
  // If any residual is meaningfully > 1e-6, the existing colour-by-
  // residual encoding is the honest signal; no badge should appear.
  const frame = {
    step: 3,
    layer_states: {
      logic: {
        n_sites: 4,
        term_count: 2,
        terms: [
          { rule_id: 'R-Beta', site: 0, arity: 2 },
          { rule_id: 'R-Arith-Pre', site: 1, arity: 2 },
        ],
        residuals: [0, 0.05],
        total_energy: 0.05,
      },
    },
  } as any;
  render(<LogicPanel frame={frame} />);
  expect(screen.queryByTestId('logic-relaxed-badge')).toBeNull();
});

it('does NOT render the bond-entropy chart when bond_entropies is absent (no state attached)', () => {
  // The chart only makes sense when a real MPS state is available; in
  // recordings without a state, the snapshot emits bond_entropies = null
  // and the chart MUST be omitted (no faux data).
  const frame = {
    step: 1,
    layer_states: {
      logic: {
        n_sites: 4,
        term_count: 6,
        terms: [{ rule_id: 'R-Beta', site: 0, arity: 2 }],
      },
    },
  } as any;
  const { container } = render(<LogicPanel frame={frame} />);
  expect(
    container.querySelector('[data-testid="logic-bond-entropy"]'),
  ).toBeNull();
});
