// src/qft_pcn/viz/web/src/routes/dsl/ChatPane.tsx
/**
 * Chat-mode driver for the DSL route. Loads Ollama models into a picker,
 * sends prompts to /dsl/translate, populates the DSL editor with the
 * returned DSL, accumulates chat turns.
 */

import { useEffect, useState } from 'react';
import { translate } from '../../lib/dsl';
import { defaultModel, loadModels } from '../../lib/llm';
import { useVizStore } from '../../store';
import type { LlmModel } from '../../lib/types';

export function ChatPane() {
  const [models, setModels] = useState<LlmModel[]>([]);
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);

  const chat = useVizStore((s) => s.chat);
  const append = useVizStore((s) => s.appendChat);
  const setDslText = useVizStore((s) => s.setDslText);
  const model = useVizStore((s) => s.llmModel);
  const setModel = useVizStore((s) => s.setLlmModel);
  const setError = useVizStore((s) => s.setError);

  useEffect(() => {
    loadModels()
      .then((ms) => {
        setModels(ms);
        if (!model) setModel(defaultModel(ms));
      })
      .catch((e) => setError(String(e)));
  }, []);

  const send = async () => {
    if (!prompt.trim() || !model) return;
    setBusy(true);
    append({ role: 'user', text: prompt });
    try {
      const result = await translate(prompt, model);
      if (result.dsl) {
        setDslText(JSON.stringify(result.dsl, null, 2));
        append({ role: 'assistant',
                 text: `Emitted DSL into the editor.`,
                 artifact: { kind: 'dsl', payload: result.dsl } });
      } else {
        append({ role: 'assistant',
                 text: `DSL validation failed: ${result.error}\n` +
                       `${(result.validation_errors ?? []).join('\n')}` });
      }
      setPrompt('');
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="chat-pane">
      <div className="chat-header">
        <label>model
          <select value={model ?? ''}
                  onChange={(e) => setModel(e.target.value || null)}>
            <option value="">(none)</option>
            {models.map((m) => (
              <option key={m.name} value={m.name}>{m.name}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="chat-turns">
        {chat.map((t, i) => (
          <div key={i} className={`chat-turn chat-turn-${t.role}`}>
            <strong>{t.role}</strong>
            <pre>{t.text}</pre>
          </div>
        ))}
      </div>
      <div className="chat-input">
        <textarea placeholder="Ask the QPCN..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)} />
        <button type="button" onClick={send} disabled={busy || !model}>
          {busy ? 'sending…' : 'Send'}
        </button>
      </div>
    </div>
  );
}
