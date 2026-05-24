import '@testing-library/jest-dom/vitest';
import { render, fireEvent, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { CommandPalette } from './CommandPalette';
import { useVizStore } from '../store';

describe('CommandPalette', () => {
  beforeEach(() => {
    useVizStore.getState().resetAll();
  });

  it('is closed by default', () => {
    render(<CommandPalette />);
    expect(screen.queryByTestId('command-palette')).toBeNull();
  });

  it('opens on Cmd+K', () => {
    render(<CommandPalette />);
    fireEvent.keyDown(document, { key: 'k', metaKey: true });
    expect(screen.getByTestId('command-palette')).toBeInTheDocument();
  });

  it('opens on Ctrl+K too', () => {
    render(<CommandPalette />);
    fireEvent.keyDown(document, { key: 'k', ctrlKey: true });
    expect(screen.getByTestId('command-palette')).toBeInTheDocument();
  });

  it('closes on Escape', () => {
    render(<CommandPalette />);
    fireEvent.keyDown(document, { key: 'k', metaKey: true });
    expect(screen.getByTestId('command-palette')).toBeInTheDocument();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByTestId('command-palette')).toBeNull();
  });

  it('typing into the input shows results', () => {
    render(<CommandPalette />);
    fireEvent.keyDown(document, { key: 'k', metaKey: true });
    const input = screen.getByTestId('command-palette-input') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'Yukawa' } });
    const results = screen.getByTestId('command-palette-results');
    expect(results.textContent).toContain('multifield-yukawa');
  });

  it('clicking a layer result updates store.selectedLayer and closes the palette', () => {
    render(<CommandPalette />);
    fireEvent.keyDown(document, { key: 'k', metaKey: true });
    const input = screen.getByTestId('command-palette-input') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'manifold' } });

    // First layer result for "manifold" must be the manifold layer.
    const layerResult = screen.getByTestId('command-palette-result-layer-manifold');
    fireEvent.click(layerResult);

    expect(useVizStore.getState().selectedLayer).toBe('manifold');
    expect(screen.queryByTestId('command-palette')).toBeNull();
  });
});
