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

it('renders an extension-pending badge when the layer state is empty', () => {
  render(<LogicPanel frame={{ step: 0, layer_states: {} } as any} />);
  expect(screen.getByText(/extension pending/i)).toBeInTheDocument();
});
