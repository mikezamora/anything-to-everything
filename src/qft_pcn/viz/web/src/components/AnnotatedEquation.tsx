// src/qft_pcn/viz/web/src/components/AnnotatedEquation.tsx
/**
 * Render an equation registered in lib/equations.ts:
 *   - KaTeX equation
 *   - Per-symbol glossary line with role-coloured swatches
 *   - Natural-language gloss
 *   - Source citation footer
 */

import katex from 'katex';
import { EQUATIONS, ROLE_COLOR, ROLE_LABEL } from '../lib/equations';

interface Props {
  id: string;
}

export function AnnotatedEquation({ id }: Props) {
  const eq = EQUATIONS[id];
  if (!eq) {
    return (
      <div className="annotated-eq annotated-eq-missing">
        missing equation: <code>{id}</code>
      </div>
    );
  }
  const katexHtml = katex.renderToString(eq.tex, {
    throwOnError: false,
    displayMode: true,
  });
  return (
    <div className="annotated-eq">
      <div
        className="annotated-eq-tex"
        dangerouslySetInnerHTML={{ __html: katexHtml }}
      />
      <ul className="annotated-eq-symbols">
        {eq.symbolGlosses.map((sg, i) => {
          const symHtml = katex.renderToString(sg.symbol, {
            throwOnError: false,
            displayMode: false,
          });
          return (
            <li key={i} className="annotated-eq-symbol-row">
              <span
                className="annotated-eq-swatch"
                style={{ background: ROLE_COLOR[sg.role] }}
                title={ROLE_LABEL[sg.role]}
              />
              <span
                className="annotated-eq-symbol-name"
                dangerouslySetInnerHTML={{ __html: symHtml }}
              />
              <span className="annotated-eq-symbol-gloss">
                — {sg.gloss}{' '}
                <small>({ROLE_LABEL[sg.role]})</small>
              </span>
            </li>
          );
        })}
      </ul>
      <p className="annotated-eq-gloss">{eq.gloss}</p>
      <div className="annotated-eq-cite">{eq.sourceCitation}</div>
    </div>
  );
}
