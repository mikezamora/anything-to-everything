import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { it, expect } from 'vitest';
import { ManifoldPanel } from './ManifoldPanel';
import { manifoldFrame, emptyFrame } from './__fixtures__/frames';

it('renders the mean |R| readout', () => {
  render(<ManifoldPanel frame={manifoldFrame} />);
  expect(screen.getByText('mean |R|')).toBeInTheDocument();
});

it('exposes a channel selector combobox', () => {
  render(<ManifoldPanel frame={manifoldFrame} />);
  expect(screen.getByRole('combobox', { name: /channel/i })).toBeInTheDocument();
});

it('exposes height-channel toggle buttons for h_xx, h_xy, h_yy, tr(h) per §3.2', () => {
  render(<ManifoldPanel frame={manifoldFrame} />);
  // The metric perturbation is a rank-2 tensor; all three components must
  // be reachable from the toolbar so the user can audit shear (h_xy) and
  // the orthogonal h_yy, not just h_xx.
  expect(screen.getByRole('button', { name: 'h_xx' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'h_xy' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'h_yy' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'tr(h)' })).toBeInTheDocument();
  // h_xx is the default-active channel.
  expect(screen.getByRole('button', { name: 'h_xx' })).toHaveClass('active');
  // Clicking another channel makes it active.
  fireEvent.click(screen.getByRole('button', { name: 'h_xy' }));
  expect(screen.getByRole('button', { name: 'h_xy' })).toHaveClass('active');
  expect(screen.getByRole('button', { name: 'h_xx' })).not.toHaveClass('active');
});

it('mounts with an empty layer state', () => {
  render(<ManifoldPanel frame={emptyFrame} />);
});

it('renders a Δ next to mean |R| when a baseline frame is supplied', () => {
  // manifoldFrame.mean_abs_ricci = 0.42; baseline with 0.40 -> +0.020.
  const baseline = {
    ...manifoldFrame,
    layer_states: {
      ...manifoldFrame.layer_states,
      manifold: {
        ...(manifoldFrame.layer_states.manifold as object),
        mean_abs_ricci: 0.4,
      },
    },
  };
  const { container } = render(
    <ManifoldPanel frame={manifoldFrame} baselineFrame={baseline} />,
  );
  const delta = container.querySelector('.panel-readout-delta') as HTMLElement;
  expect(delta).toBeTruthy();
  expect(delta.textContent).toContain('+0.020');
});
