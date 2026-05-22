import { describe, it, expect, vi, beforeEach } from 'vitest';
import { connectRun } from './ws';
import { useVizStore } from '../store';

/** Minimal stand-in for the browser WebSocket, capturing handlers. */
class FakeWebSocket {
  static last: FakeWebSocket | null = null;
  url: string;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;

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
    useVizStore.getState().reset();
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

  it('routes a parsed frame into the store via pushFrame', async () => {
    const handle = await connectRun({ layers: ['manifold'], steps: 2, grid: 8 });
    expect(handle.runId).toBe('run-123');

    const socket = FakeWebSocket.last!;
    expect(socket.url).toContain('/ws/run-123');

    socket.emit(JSON.stringify({ step: 0, layer_states: {} }));
    socket.emit(JSON.stringify({ step: 1, layer_states: {} }));

    expect(useVizStore.getState().frames.length).toBe(2);
    expect(useVizStore.getState().currentFrame()?.step).toBe(1);
  });

  it('clears live on the done sentinel', async () => {
    await connectRun({ layers: ['manifold'], steps: 1, grid: 8 });
    FakeWebSocket.last!.emit(JSON.stringify({ done: true }));
    expect(useVizStore.getState().live).toBe(false);
  });

  it('records an error message', async () => {
    await connectRun({ layers: ['manifold'], steps: 1, grid: 8 });
    FakeWebSocket.last!.emit(JSON.stringify({ error: 'boom' }));
    expect(useVizStore.getState().live).toBe(false);
    expect(useVizStore.getState().error).toBe('boom');
  });
});
