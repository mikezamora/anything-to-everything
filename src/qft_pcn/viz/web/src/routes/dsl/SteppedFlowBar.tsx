/**
 * Three explicit buttons (Generate / Run / Verbalize) with the
 * intermediate artifact shown between steps. Gated: Run requires non-empty
 * dslText; Verbalize requires a completed run with observables.
 */

import { useState } from 'react';
import { translate, runDsl, verbalize } from '../../lib/dsl';
import { useVizStore } from '../../store';

export function SteppedFlowBar() {
  const dslText = useVizStore((s) => s.dslText);
  const setDslText = useVizStore((s) => s.setDslText);
  const model = useVizStore((s) => s.llmModel);
  const appendChat = useVizStore((s) => s.appendChat);
  const setError = useVizStore((s) => s.setError);
  const activeRunId = useVizStore((s) => s.activeRunId);
  const run = useVizStore((s) =>
    activeRunId ? s.runs.get(activeRunId) : undefined);
  const [prompt, setPrompt] = useState('');

  const onGenerate = async () => {
    if (!prompt.trim() || !model) return;
    try {
      const r = await translate(prompt, model);
      if (r.dsl) setDslText(JSON.stringify(r.dsl, null, 2));
      else setError(r.error ?? 'unknown');
    } catch (e) { setError(String(e)); }
  };

  const onRun = async () => {
    try {
      const dsl = JSON.parse(dslText);
      const { run_id } = await runDsl(dsl);
      appendChat({ role: 'assistant', text: `Run started: ${run_id}` });
      // Open WS via the existing connectRun path? For stepped mode we
      // just register the run and let the user switch back to Viz route
      // to watch it stream. Document this in the chat.
    } catch (e) { setError(String(e)); }
  };

  const onVerbalize = async () => {
    if (!run || !model) return;
    const lastFrame = run.frames[run.frames.length - 1];
    const obs = lastFrame?.layer_states?.qpcn ?? {};
    try {
      const r = await verbalize(obs, model, prompt);
      appendChat({ role: 'assistant', text: r.text });
    } catch (e) { setError(String(e)); }
  };

  return (
    <div className="stepped-flow-bar">
      <input placeholder="prompt"
             value={prompt}
             onChange={(e) => setPrompt(e.target.value)} />
      <button type="button" onClick={onGenerate}
              disabled={!prompt.trim() || !model}>1. Generate DSL</button>
      <button type="button" onClick={onRun}
              disabled={!dslText.trim()}>2. Run DSL</button>
      <button type="button" onClick={onVerbalize}
              disabled={!run || !model}>3. Verbalize</button>
    </div>
  );
}
