import { describe, it, expect } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { render, screen, act } from '@testing-library/react';
import { PanelReadouts } from './PanelReadouts';

describe('PanelReadouts', () => {
  it('renders one cell per entry', () => {
    render(<PanelReadouts cells={[
      { label: 'energy', value: 1.23 },
      { label: 'χ', value: 8 },
    ]} />);
    expect(screen.getByText('energy')).toBeInTheDocument();
    expect(screen.getByText('1.23')).toBeInTheDocument();
  });

  it('renders a green Δ with sign for negative deltas (improvement)', () => {
    const { container } = render(<PanelReadouts cells={[
      { label: 'energy', value: -1.5, baselineValue: -1.0 },
    ]} />);
    const delta = container.querySelector('.panel-readout-delta') as HTMLElement;
    expect(delta).toBeTruthy();
    expect(delta.textContent).toContain('-0.500');
    expect(delta.style.color.replace(/\s/g, '')).toBe('rgb(154,237,193)');
  });

  it('renders a red Δ with + sign for positive deltas (regression)', () => {
    const { container } = render(<PanelReadouts cells={[
      { label: 'err', value: 0.5, baselineValue: 0.2 },
    ]} />);
    const delta = container.querySelector('.panel-readout-delta') as HTMLElement;
    expect(delta).toBeTruthy();
    expect(delta.textContent).toContain('+0.300');
    expect(delta.style.color.replace(/\s/g, '')).toBe('rgb(239,144,144)');
  });

  it('falls back to current-only when baseline is absent', () => {
    const { container } = render(<PanelReadouts cells={[
      { label: 'energy', value: 1.0 },
    ]} />);
    expect(container.querySelector('.panel-readout-delta')).toBeNull();
  });

  it('falls back to current-only when baseline is non-numeric (null)', () => {
    const { container } = render(<PanelReadouts cells={[
      { label: 'energy', value: 1.0, baselineValue: null },
    ]} />);
    expect(container.querySelector('.panel-readout-delta')).toBeNull();
  });

  it('flashes the cell whose highlightId matches the event', () => {
    const { container } = render(<PanelReadouts cells={[
      { label: 'energy', value: 1.0, highlightId: 'energy' },
    ]} />);
    act(() => {
      window.dispatchEvent(new CustomEvent('viz:highlight-readout',
        { detail: { id: 'energy' } }));
    });
    expect(container.querySelector('.panel-readout.highlight')).toBeTruthy();
  });
});
