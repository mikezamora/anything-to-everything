import { describe, it, expect, vi, beforeEach } from 'vitest';
import { connectRun } from './ws';
import { useVizStore } from '../store';

/** Minimal stand-in for the browser WebSocket, capturing handlers. */
class FakeWebSocket {
  static last: FakeWebSocket | null = null;
  url: string;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.last = this;
  }

  /** Simulate a server message arriving. */
  emit(data: string): void {
    this.onmessage?.({ data });
  }

  close(): void {}
}

describe('connectRun', () => {
  beforeEach(() => {
    useVizStore.getState().resetAll();
    FakeWebSocket.last = null;
    vi.stubGlobal('WebSocket', FakeWebSocket as unknown as typeof WebSocket);
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ run_id: 'run-123' }),
      })),
    );
  });

  it('opens the run in the store and routes frames under its id', async () => {
    const spec = { layers: ['manifold'], steps: 2, grid: 8 };
    const handle = await connectRun(spec);
    expect(handle.runId).toBe('run-123');

    // The /run request must be a JSON POST carrying the run spec.
    const fetchMock = fetch as unknown as ReturnType<typeof vi.fn>;
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/run');
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>)['Content-Type']).toBe(
      'application/json',
    );
    expect(JSON.parse(init.body as string)).toEqual(spec);

    // openRun + setActiveRun must have been called.
    const state = useVizStore.getState();
    expect(state.runs.has('run-123')).toBe(true);
    expect(state.activeRunId).toBe('run-123');

    const socket = FakeWebSocket.last!;
    expect(socket.url).toContain('/ws/run-123');

    socket.emit(JSON.stringify({ step: 0, layer_states: {} }));
    socket.emit(JSON.stringify({ step: 1, layer_states: {} }));

    const run = useVizStore.getState().runs.get('run-123')!;
    expect(run.frames.length).toBe(2);
    expect(useVizStore.getState().currentFrame()?.step).toBe(1);
  });

  it('clears live for the run on the done sentinel', async () => {
    await connectRun({ layers: ['manifold'], steps: 1, grid: 8 });
    FakeWebSocket.last!.emit(JSON.stringify({ done: true }));
    expect(useVizStore.getState().runs.get('run-123')!.live).toBe(false);
  });

  it('records an error message and clears live for the run', async () => {
    await connectRun({ layers: ['manifold'], steps: 1, grid: 8 });
    FakeWebSocket.last!.emit(JSON.stringify({ error: 'boom' }));
    expect(useVizStore.getState().runs.get('run-123')!.live).toBe(false);
    expect(useVizStore.getState().error).toBe('boom');
  });

  it('clears live when the socket closes without a sentinel', async () => {
    await connectRun({ layers: ['manifold'], steps: 1, grid: 8 });
    expect(useVizStore.getState().runs.get('run-123')!.live).toBe(true);
    FakeWebSocket.last!.onclose?.();
    expect(useVizStore.getState().runs.get('run-123')!.live).toBe(false);
  });

  it('records an error for a malformed message instead of throwing', async () => {
    await connectRun({ layers: ['manifold'], steps: 1, grid: 8 });
    expect(() => FakeWebSocket.last!.emit('not json{')).not.toThrow();
    expect(useVizStore.getState().error).toMatch(/Malformed WebSocket message/);
  });

  it('rejects when POST /run responds non-ok', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 500, json: async () => ({}) })),
    );
    await expect(
      connectRun({ layers: ['manifold'], steps: 1, grid: 8 }),
    ).rejects.toThrow(/\/run: 500/);
  });
});
