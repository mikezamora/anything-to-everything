import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { it, expect } from 'vitest';
import { VqcPanel } from './VqcPanel';
import { vqcFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<VqcPanel frame={vqcFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<VqcPanel frame={emptyFrame} />);
});

it('renders an extension-pending badge when the layer state is empty', () => {
  render(<VqcPanel frame={{ step: 0, layer_states: {} } as any} />);
  expect(screen.getByText(/extension pending/i)).toBeInTheDocument();
});
