/**
 * WebSocket client for the QFT-PCN viz server.
 *
 * `connectRun` POSTs a `RunSpec` to `/run`, opens the run's WebSocket, and
 * routes each streamed message into the Zustand store: normal `Frame`s are
 * pushed, the `{done: true}` sentinel ends the live stream, and an `{error}`
 * message records the failure. The returned handle lets the caller close the
 * socket early.
 */

import { useVizStore } from '../store';
import type { Frame, RunSpec } from './types';

export interface RunHandle {
  /** The opaque run id returned by `POST /run`. */
  runId: string;
  /** Close the WebSocket if it is still open. */
  close: () => void;
}

/** Build the `ws(s)://` URL for a run id, honoring the current page origin. */
function wsUrl(runId: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws/${runId}`;
}

/** Route one parsed WebSocket message into the store. */
function handleMessage(raw: string): void {
  const store = useVizStore.getState();
  try {
    const msg = JSON.parse(raw) as
      | Frame
      | { done: true }
      | { error: string };

    if ('error' in msg) {
      store.setLive(false);
      store.setError(msg.error);
    } else if ('done' in msg) {
      store.setLive(false);
    } else {
      store.pushFrame(msg);
    }
  } catch (err) {
    store.setError(`Malformed WebSocket message: ${String(err)}`);
  }
}

/**
 * Start a run and stream its frames into the store.
 *
 * Resets the store, registers the run via `POST /run`, then opens the
 * WebSocket. Resolves once the socket has been created.
 */
export async function connectRun(spec: RunSpec): Promise<RunHandle> {
  useVizStore.getState().reset();

  const res = await fetch('/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(spec),
  });
  if (!res.ok) {
    throw new Error(`POST /run failed: ${res.status}`);
  }
  const { run_id: runId } = (await res.json()) as { run_id: string };

  const socket = new WebSocket(wsUrl(runId));
  socket.onmessage = (ev: MessageEvent) => handleMessage(String(ev.data));
  socket.onerror = () => {
    useVizStore.getState().setLive(false);
    useVizStore.getState().setError('WebSocket error');
  };
  socket.onclose = () => {
    // If the backend drops the connection without the {done:true}
    // sentinel, ensure the store does not stay live forever.
    useVizStore.getState().setLive(false);
  };

  return {
    runId,
    close: () => socket.close(),
  };
}
