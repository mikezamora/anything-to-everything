// src/qft_pcn/viz/web/src/components/SectionedLayerSelector.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SectionedLayerSelector } from './SectionedLayerSelector';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

describe('SectionedLayerSelector', () => {
  it('renders the three section headers', () => {
    render(<SectionedLayerSelector />);
    expect(screen.getByText('QFT Substrate')).toBeInTheDocument();
    expect(screen.getByText('PCN Substrate')).toBeInTheDocument();
    expect(screen.getByText('QPCN Fusion')).toBeInTheDocument();
  });

  it('selects a layer when its button is clicked', () => {
    render(<SectionedLayerSelector />);
    fireEvent.click(screen.getByRole('button', { name: 'mps' }));
    expect(useVizStore.getState().selectedLayer).toBe('mps');
  });

  it('clicking a section header selects its intro pseudo-layer', () => {
    render(<SectionedLayerSelector />);
    fireEvent.click(screen.getByRole('button', { name: /qft substrate/i }));
    expect(useVizStore.getState().selectedLayer).toBe('intro-qft');
  });

  it('section is collapsible', () => {
    const { container } = render(<SectionedLayerSelector />);
    const qftHeader = screen.getByRole('button', { name: /qft substrate/i });
    // Layers start visible — clicking should collapse, click again expands.
    fireEvent.click(qftHeader.parentElement!.querySelector(
      '.section-collapse-toggle')!);
    expect(container.querySelector('.section-qft.collapsed')).toBeTruthy();
  });
});
