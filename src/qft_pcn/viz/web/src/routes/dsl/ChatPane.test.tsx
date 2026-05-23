import '@testing-library/jest-dom/vitest';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChatPane } from './ChatPane';
import { useVizStore } from '../../store';
import { _resetLlmCache } from '../../lib/llm';

beforeEach(() => {
  _resetLlmCache();
  useVizStore.getState().resetAll();
  (globalThis as any).fetch = vi.fn(async (url: any) => {
    const u = String(url);
    if (u.endsWith('/dsl/models'))
      return { ok: true, json: async () =>
        [{ name: 'gemma3:4b' }] } as any;
    if (u.endsWith('/dsl/translate'))
      return { ok: true, json: async () =>
        ({ dsl: { fields: [{ name: 'A', cutoff: 2 }] } }) } as any;
    throw new Error('unexpected ' + u);
  }) as any;
});

describe('ChatPane', () => {
  it('loads models into the picker', async () => {
    render(<ChatPane />);
    await waitFor(() =>
      expect(screen.getByText('gemma3:4b')).toBeInTheDocument());
  });

  it('Send appends a user turn + an assistant turn with the dsl artifact',
     async () => {
    render(<ChatPane />);
    await waitFor(() => screen.getByText('gemma3:4b'));
    fireEvent.change(screen.getByPlaceholderText(/ask the QPCN/i),
                     { target: { value: 'simulate A' } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));
    await waitFor(() => {
      expect(useVizStore.getState().chat.length).toBeGreaterThanOrEqual(2);
      expect(useVizStore.getState().dslText).toContain('"name": "A"');
    });
  });
});
