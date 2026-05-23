// src/qft_pcn/viz/web/src/panels/IntroPcnPanel.tsx
import { PanelShell } from './PanelShell';
import { SECTIONS } from '../lib/sections';
import type { Frame } from '../lib/types';

export function IntroPcnPanel({ frame, baselineFrame: _baselineFrame }: {
  frame: Frame; baselineFrame?: Frame;
}) {
  const sec = SECTIONS.find((s) => s.key === 'pcn')!;
  return (
    <PanelShell title={sec.title} step={frame.step} hasData
                emptyMessage="">
      <div className="intro-panel">
        <p className="intro-tagline">{sec.tagline}</p>
        {sec.story.map((p, i) => <p key={i}>{p}</p>)}
        <h4>Layers in this section</h4>
        <ul>
          {sec.layerSummaries.map((ls) => (
            <li key={ls.layer}>
              <strong>{ls.layer}</strong> — {ls.oneLine}
            </li>
          ))}
        </ul>
        {sec.references && (
          <>
            <h4>References</h4>
            <ul>
              {sec.references.map((r, i) =>
                <li key={i}><a href={r.href}>{r.label}</a></li>)}
            </ul>
          </>
        )}
      </div>
    </PanelShell>
  );
}
