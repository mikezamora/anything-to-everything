/**
 * Right pane of the DSL route. Shows the last observed frame for the
 * active run, plus a button to export the current run's RunSpec back to a
 * DSL JSON download.
 */

import { exportRunAsDsl } from '../../lib/dsl';
import { useVizStore } from '../../store';

export function RunOutputPane() {
  const activeRunId = useVizStore((s) => s.activeRunId);
  const run = useVizStore((s) =>
    activeRunId ? s.runs.get(activeRunId) : undefined);
  const setError = useVizStore((s) => s.setError);

  const exportDsl = async () => {
    if (!activeRunId) return;
    try {
      const dsl = await exportRunAsDsl(activeRunId);
      const blob = new Blob([JSON.stringify(dsl, null, 2)],
                             { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `run-${activeRunId}.dsl.json`;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    } catch (e) { setError(String(e)); }
  };

  return (
    <div className="run-output-pane">
      <div className="run-output-header">
        <span>run: <code>{activeRunId ?? '—'}</code></span>
        <button type="button" disabled={!activeRunId} onClick={exportDsl}>
          Export DSL
        </button>
      </div>
      {run ? (
        <pre className="run-output-frame">
          {JSON.stringify(run.frames[run.frames.length - 1] ?? {}, null, 2)}
        </pre>
      ) : (
        <p className="empty">No active run.</p>
      )}
    </div>
  );
}
