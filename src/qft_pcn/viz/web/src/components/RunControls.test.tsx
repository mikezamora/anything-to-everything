import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { RunControls } from './RunControls';
import { _resetPresetCache } from '../lib/presets';
import { useVizStore } from '../store';

const presetsFixture = [{
  id: 'manifold.flat', layer: 'manifold',
  label: 'Flat — zero curvature start',
  description: 'baseline',
  spec_overrides: { layers: ['manifold'], steps: 30, grid: 12,
                    seed: 0, params: { manifold: { source: 'flat' } } },
}];
const schemaFixture = {
  manifold: { type: 'object', properties: {
    source: { type: 'string', enum: ['flat', 'hot-spot'], default: 'flat' },
  } },
};

beforeEach(() => {
  _resetPresetCache();
  useVizStore.getState().resetAll();
  (globalThis as any).fetch = vi.fn(async (url: any) => {
    const u = String(url);
    if (u.endsWith('/presets'))
      return { ok: true, json: async () => presetsFixture } as any;
    if (u.endsWith('/params/schema'))
      return { ok: true, json: async () => schemaFixture } as any;
    if (u.endsWith('/run'))
      return { ok: true, json: async () => ({ run_id: 'R1' }) } as any;
    throw new Error('unexpected ' + u);
  }) as any;
});

describe('RunControls', () => {
  it('loads presets into the dropdown', async () => {
    render(<RunControls />);
    await waitFor(() =>
      expect(screen.getByText('Flat — zero curvature start'))
        .toBeInTheDocument());
  });

  it('applying a preset sets steps/grid/seed/params from spec_overrides',
     async () => {
    render(<RunControls />);
    await waitFor(() => screen.getByText('Flat — zero curvature start'));
    fireEvent.change(screen.getByRole('combobox', { name: /preset/i }),
      { target: { value: 'manifold.flat' } });
    const stepsInput = screen.getByLabelText(/steps/i) as HTMLInputElement;
    expect(stepsInput.value).toBe('30');
  });

  it('Run posts a RunSpec containing the applied params', async () => {
    render(<RunControls />);
    await waitFor(() => screen.getByText('Flat — zero curvature start'));
    fireEvent.change(screen.getByRole('combobox', { name: /preset/i }),
      { target: { value: 'manifold.flat' } });
    fireEvent.click(screen.getByText('Run'));
    await waitFor(() => {
      const last = ((globalThis as any).fetch).mock.calls
        .find((c: any[]) => String(c[0]).endsWith('/run'));
      expect(last).toBeTruthy();
      const body = JSON.parse(last[1].body);
      expect(body.params.manifold.source).toBe('flat');
    });
  });
});
