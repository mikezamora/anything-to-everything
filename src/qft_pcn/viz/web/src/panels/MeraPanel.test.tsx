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

  it('renders an iso err (max) readout and a per-layer sparkline', () => {
    const { container } = render(<MeraPanel frame={meraFrame} />);
    expect(screen.getByText('iso err (max)')).toBeInTheDocument();
    expect(
      container.querySelector('[data-testid="mera-iso-sparkline"]'),
    ).toBeTruthy();
  });

  it('renders a side-by-side two-disk layout when baselineFrame has data', () => {
    const { container } = render(
      <MeraPanel frame={meraFrame} baselineFrame={meraFrame} />,
    );
    const layout = container.querySelector('[data-testid="mera-layout"]');
    expect(layout).toBeTruthy();
    expect(layout!.getAttribute('data-compare')).toBe('side-by-side');
    expect(
      container.querySelector('[data-testid="mera-disk-active"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="mera-disk-baseline"]'),
    ).toBeTruthy();
  });

  it('renders a single-disk layout when no baseline is supplied', () => {
    const { container } = render(<MeraPanel frame={meraFrame} />);
    const layout = container.querySelector('[data-testid="mera-layout"]');
    expect(layout!.getAttribute('data-compare')).toBe('single');
    expect(
      container.querySelector('[data-testid="mera-disk-baseline"]'),
    ).toBeNull();
  });
});
