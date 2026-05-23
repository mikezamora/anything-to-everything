import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
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

it('mounts with an empty layer state', () => {
  render(<ManifoldPanel frame={emptyFrame} />);
});
