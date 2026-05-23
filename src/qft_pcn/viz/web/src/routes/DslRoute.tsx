/**
 * Three-region DSL route: chat pane | Monaco editor | run output.
 * SteppedFlowBar overlays when steppedMode is on.
 */

import { ChatPane } from './dsl/ChatPane';
import { DslEditor } from './dsl/DslEditor';
import { RunOutputPane } from './dsl/RunOutputPane';
import { SteppedFlowBar } from './dsl/SteppedFlowBar';
import { useVizStore } from '../store';

export function DslRoute() {
  const steppedMode = useVizStore((s) => s.steppedMode);
  const setSteppedMode = useVizStore((s) => s.setSteppedMode);
  return (
    <div className="dsl-route">
      <header className="dsl-route-header">
        <label>
          <input type="checkbox" checked={steppedMode}
                 onChange={(e) => setSteppedMode(e.target.checked)} />
          stepped-flow mode
        </label>
      </header>
      {steppedMode && <SteppedFlowBar />}
      <div className="dsl-route-body">
        <ChatPane />
        <DslEditor />
        <RunOutputPane />
      </div>
    </div>
  );
}
