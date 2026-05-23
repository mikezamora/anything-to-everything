import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MeraPanel } from './MeraPanel';
import { meraFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<MeraPanel frame={meraFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<MeraPanel frame={emptyFrame} />);
});

describe('MeraPanel readouts', () => {
  it('renders leaves, layers, and max χ readout cells', () => {
    render(<MeraPanel frame={meraFrame} />);
    expect(screen.getByText('leaves')).toBeInTheDocument();
    expect(screen.getByText('layers')).toBeInTheDocument();
    expect(screen.getByText('max χ')).toBeInTheDocument();
  });

  it('accepts an optional baselineFrame prop (no-op)', () => {
    render(<MeraPanel frame={meraFrame} baselineFrame={meraFrame} />);
    expect(screen.getByText('leaves')).toBeInTheDocument();
  });
});
