/**
 * Monaco-backed JSON editor bound to store.dslText.
 *
 * Server is authoritative for schema validation (POST /dsl/run reports
 * full validation errors); the inline status line just checks that the
 * editor content parses as JSON so the user can spot syntax errors
 * before submitting.
 */

import Editor from '@monaco-editor/react';
import { useVizStore } from '../../store';

export function DslEditor() {
  const dslText = useVizStore((s) => s.dslText);
  const setDslText = useVizStore((s) => s.setDslText);

  let parseStatus = 'empty';
  if (dslText.trim()) {
    try { JSON.parse(dslText); parseStatus = 'JSON OK'; }
    catch (e) { parseStatus = `JSON error: ${(e as Error).message}`; }
  }

  return (
    <div className="dsl-editor">
      <Editor
        height="100%"
        defaultLanguage="json"
        value={dslText}
        onChange={(v) => setDslText(v ?? '')}
        options={{ minimap: { enabled: false },
                   fontSize: 12, scrollBeyondLastLine: false }}
      />
      <div className="dsl-editor-status">{parseStatus}</div>
    </div>
  );
}
