/**
 * Thin client for /runs/{id}/{pause,resume,step}. Each helper POSTs and
 * returns the server's JSON response. Errors are surfaced to the caller
 * (the UI sets store.error).
 */

async function post(path: string): Promise<unknown> {
  const res = await fetch(path, { method: 'POST' });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

export const transport = {
  pause: (runId: string) => post(`/runs/${runId}/pause`),
  resume: (runId: string) => post(`/runs/${runId}/resume`),
  step: (runId: string) => post(`/runs/${runId}/step`),
};
