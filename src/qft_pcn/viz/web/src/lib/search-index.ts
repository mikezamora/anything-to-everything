/**
 * Cross-panel concept search index.
 *
 * Resolves the `cross-panel-concept-search` deferred extension
 * (src/qft_pcn/viz/EXTENSIONS.md#cross-panel-concept-search).
 *
 * Builds an in-memory inverted index at module load over:
 *   - EXPLAINERS[layer].tabs (overview prose + element meanings + tab
 *     labels);
 *   - EQUATIONS[id] (gloss + per-symbol glosses);
 *   - ARTICLES[i] (title + every prose-section paragraph).
 *
 * Tokenisation: lowercased, split on /\W+/, drop tokens shorter than 3
 * characters. Query results are scored by matching-token count; ties
 * broken by source order. The snippet is the first matching paragraph
 * truncated to ~140 characters.
 */

import { EXPLAINERS } from './explainer';
import { EQUATIONS } from './equations';
import { ARTICLES } from '../routes/learn/articles';

export interface SearchResult {
  title: string;
  snippet: string;
  target: { kind: 'layer' | 'article' | 'equation'; id: string };
}

interface IndexedDoc {
  result: SearchResult;
  /** Paragraphs (already plain prose) so the snippet picker can return
   *  the first paragraph that hits the query. */
  paragraphs: string[];
  /** Lowercased token set for fast intersection. */
  tokens: Set<string>;
  /** Insertion order — used to break score ties deterministically. */
  order: number;
}

const TOKEN_RE = /\W+/g;

function tokenize(text: string): string[] {
  if (!text) return [];
  return text
    .toLowerCase()
    .split(TOKEN_RE)
    .filter((t) => t.length >= 3);
}

function tokensFromParagraphs(paragraphs: string[]): Set<string> {
  const out = new Set<string>();
  for (const p of paragraphs) for (const t of tokenize(p)) out.add(t);
  return out;
}

function truncate(text: string, max = 140): string {
  const trimmed = text.replace(/\s+/g, ' ').trim();
  if (trimmed.length <= max) return trimmed;
  return trimmed.slice(0, max - 1).trimEnd() + '…';
}

function firstMatchingParagraph(
  paragraphs: string[],
  queryTokens: string[],
): string {
  for (const p of paragraphs) {
    const lower = p.toLowerCase();
    if (queryTokens.some((q) => lower.includes(q))) return p;
  }
  return paragraphs[0] ?? '';
}

/** Module-load build of the doc list. */
const DOCS: IndexedDoc[] = (() => {
  const docs: IndexedDoc[] = [];
  let order = 0;

  // --- Layer explainers ---------------------------------------------------
  for (const [layer, spec] of Object.entries(EXPLAINERS)) {
    const paragraphs: string[] = [];
    paragraphs.push(spec.title);
    paragraphs.push(spec.oneLine);
    for (const w of spec.tabs.overview.what) paragraphs.push(w);
    for (const el of spec.tabs.overview.elements) {
      paragraphs.push(`${el.name}: ${el.meaning}`);
    }
    // Tab labels (constant across layers, but cheap and lets a query like
    // "watch" or "training dynamics" surface layers).
    paragraphs.push('Overview Math Worked Example Training Dynamics Watch');
    docs.push({
      result: {
        title: spec.title,
        snippet: truncate(spec.oneLine),
        target: { kind: 'layer', id: layer },
      },
      paragraphs,
      tokens: tokensFromParagraphs(paragraphs),
      order: order++,
    });
  }

  // --- Equations ----------------------------------------------------------
  for (const [id, eq] of Object.entries(EQUATIONS)) {
    const paragraphs: string[] = [];
    paragraphs.push(eq.gloss);
    for (const sg of eq.symbolGlosses) paragraphs.push(sg.gloss);
    docs.push({
      result: {
        title: id,
        snippet: truncate(eq.gloss),
        target: { kind: 'equation', id },
      },
      paragraphs,
      tokens: tokensFromParagraphs(paragraphs),
      order: order++,
    });
  }

  // --- Articles -----------------------------------------------------------
  for (const art of ARTICLES) {
    const paragraphs: string[] = [art.title];
    for (const sec of art.sections) {
      if (sec.kind === 'prose') {
        for (const b of sec.body) paragraphs.push(b);
      }
    }
    docs.push({
      result: {
        title: art.title,
        snippet: truncate(paragraphs[1] ?? art.title),
        target: { kind: 'article', id: art.id },
      },
      paragraphs,
      tokens: tokensFromParagraphs(paragraphs),
      order: order++,
    });
  }

  return docs;
})();

/**
 * Search the in-memory index. Scores by the number of query tokens that
 * appear in the doc's token set; ties resolved by source order (layers,
 * then equations, then articles, each in their registry order).
 */
export function search(query: string, limit = 20): SearchResult[] {
  const qTokens = tokenize(query);
  if (qTokens.length === 0) return [];
  const scored: { score: number; order: number; doc: IndexedDoc }[] = [];
  for (const doc of DOCS) {
    let score = 0;
    for (const t of qTokens) if (doc.tokens.has(t)) score++;
    if (score > 0) scored.push({ score, order: doc.order, doc });
  }
  scored.sort((a, b) => (b.score - a.score) || (a.order - b.order));
  return scored.slice(0, limit).map(({ doc }) => ({
    ...doc.result,
    snippet: truncate(firstMatchingParagraph(doc.paragraphs, qTokens)),
  }));
}
