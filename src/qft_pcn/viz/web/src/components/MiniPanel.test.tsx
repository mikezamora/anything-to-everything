import '@testing-library/jest-dom/vitest';
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MiniPanel } from './MiniPanel';

describe('MiniPanel', () => {
  it('renders the live pcn-dynamics panel at scale(0.5)', () => {
    const { container, queryByTestId } = render(<MiniPanel layer="pcn-dynamics" />);
    expect(queryByTestId('mini-panel-live')).toBeTruthy();
    expect(queryByTestId('mini-panel-placeholder')).toBeNull();
    const scaled = container.querySelector('.mini-panel-scale') as HTMLElement;
    expect(scaled).toBeTruthy();
    expect(scaled.style.transform).toBe('scale(0.5)');
    // Live panel content (per-layer table headers) actually mounted.
    expect(container.textContent).toContain('layer');
    expect(container.textContent).toContain('total F');
  });

  it('renders the live pcn-coupling panel at scale(0.5)', () => {
    const { container, queryByTestId } = render(<MiniPanel layer="pcn-coupling" />);
    expect(queryByTestId('mini-panel-live')).toBeTruthy();
    const scaled = container.querySelector('.mini-panel-scale') as HTMLElement;
    expect(scaled.style.transform).toBe('scale(0.5)');
  });

  it('renders the static SVG placeholder for heavy panels (manifold)', () => {
    const { queryByTestId, container } = render(<MiniPanel layer="manifold" />);
    expect(queryByTestId('mini-panel-placeholder')).toBeTruthy();
    expect(queryByTestId('mini-panel-live')).toBeNull();
    expect(container.querySelector('svg')).toBeTruthy();
    expect(container.textContent).toContain('manifold');
    expect(container.textContent).toContain('Open the Viz route for live rendering');
  });

  it('renders placeholders for every other heavy layer', () => {
    for (const layer of [
      'multifield',
      'mps',
      'hamiltonian',
      'qpcn',
      'mera',
      'vqc',
      'logic',
      'mera_relax',
      'bridge',
      'pcn-fields',
    ]) {
      const { queryByTestId, unmount } = render(<MiniPanel layer={layer} />);
      expect(queryByTestId('mini-panel-placeholder'), `placeholder for ${layer}`).toBeTruthy();
      unmount();
    }
  });
});
