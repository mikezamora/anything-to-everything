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
