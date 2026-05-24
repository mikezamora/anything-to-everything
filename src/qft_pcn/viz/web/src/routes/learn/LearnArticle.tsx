// src/qft_pcn/viz/web/src/routes/learn/LearnArticle.tsx
/**
 * Render an ArticleSpec section list. Dispatch per kind:
 *   prose, equation, workedExample, trainingDynamics, miniViz, callout.
 *
 * Heavy panels (R3F / Three.js) are NOT embedded — miniViz sections for
 * those layers are deferred per EXTENSIONS.md (`learn-heavy-miniviz`).
 * Cheap-panel miniViz sections render a static SVG placeholder for v1.
 */

import { ARTICLES } from './articles';
import type { ArticleSpec, ArticleSection } from '../../lib/article-types';
import { AnnotatedEquation } from '../../components/AnnotatedEquation';

interface Props {
  articleId?: string;
  article?: ArticleSpec;
}

export function LearnArticle({ articleId, article: directArticle }: Props) {
  const article: ArticleSpec | undefined =
    directArticle ?? ARTICLES.find((a) => a.id === articleId);
  if (!article) {
    return (
      <article className="learn-article learn-article-missing">
        <p>No article registered for id "{articleId}".</p>
      </article>
    );
  }

  return (
    <article className="learn-article">
      <header>
        <h1>{article.title}</h1>
        <p className="learn-article-path">{article.sectionPath.join(' › ')}</p>
        {article.prerequisites && article.prerequisites.length > 0 && (
          <p className="learn-article-prereq">
            Prerequisites: {article.prerequisites.join(', ')}
          </p>
        )}
      </header>
      {article.sections.map((sec, i) => renderSection(sec, i))}
      {article.citations.length > 0 && (
        <footer className="learn-article-cites">
          <h4>Citations</h4>
          <ul>
            {article.citations.map((c, i) => (
              <li key={i}><a href={c.href}>{c.label}</a></li>
            ))}
          </ul>
        </footer>
      )}
    </article>
  );
}

function renderSection(sec: ArticleSection, key: number) {
  switch (sec.kind) {
    case 'prose':
      return (
        <section key={key} className="learn-section-prose">
          {sec.body.map((p, i) => <p key={i}>{p}</p>)}
        </section>
      );
    case 'equation':
      return (
        <section key={key} className="learn-section-equation">
          <AnnotatedEquation id={sec.equationId} />
          {sec.caption && <p className="learn-eq-caption">{sec.caption}</p>}
        </section>
      );
    case 'workedExample': {
      const ex = sec.example;
      return (
        <section key={key} className="learn-section-worked">
          <h3>{ex.title}</h3>
          <p><strong>Setup.</strong> {ex.setup}</p>
          <ol>
            {ex.steps.map((s, i) => (
              <li key={i}>
                {s.description}
                {s.equationId && <AnnotatedEquation id={s.equationId} />}
                <div className="learn-worked-result"><em>⇒ {s.result}</em></div>
              </li>
            ))}
          </ol>
          <p><strong>Takeaway.</strong> {ex.takeaway}</p>
        </section>
      );
    }
    case 'trainingDynamics':
      return (
        <section key={key} className="learn-section-dynamics">
          <h3>Training dynamics</h3>
          <AnnotatedEquation id={sec.updateRuleId} />
          <h4>Expect</h4>
          <ul>{sec.expect.map((b, i) => <li key={i}>{b}</li>)}</ul>
          <h4>If you see…</h4>
          <ul>
            {sec.pathologies.map((p, i) => (
              <li key={i}><strong>{p.signal}</strong> — {p.cause}</li>
            ))}
          </ul>
        </section>
      );
    case 'miniViz':
      return (
        <section key={key} className="learn-section-miniviz">
          <p className="learn-miniviz-placeholder">
            <em>Mini-viz for layer "{sec.layer}" using fixture "{sec.fixtureFrameId}" — see EXTENSIONS.md anchor #learn-heavy-miniviz for status.</em>
          </p>
        </section>
      );
    case 'callout':
      return (
        <aside
          key={key}
          className={`learn-section-callout learn-callout-${sec.severity}`}
        >
          {sec.body}
        </aside>
      );
  }
}
