/**
 * Cmd+K cross-panel concept search palette.
 *
 * Resolves the `cross-panel-concept-search` deferred extension
 * (src/qft_pcn/viz/EXTENSIONS.md#cross-panel-concept-search).
 *
 * Opens on Cmd+K / Ctrl+K, closes on Escape or click-outside. Clicking
 *   - a layer result calls `store.selectLayer(id)` and switches to
 *     the Viz route;
 *   - an article result calls `store.setRoute('learn')` and sets the
 *     URL fragment to the article id so the Learn route can scroll;
 *   - an equation result switches to the Learn route as a best-effort
 *     navigation (no per-equation panel registry exists yet).
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { search, type SearchResult } from '../lib/search-index';
import { useVizStore } from '../store';

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const selectLayer = useVizStore((s) => s.selectLayer);
  const setRoute = useVizStore((s) => s.setRoute);

  // Global Cmd+K / Ctrl+K toggle; Escape close.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen((v) => !v);
        return;
      }
      if (e.key === 'Escape') {
        setOpen(false);
      }
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    if (open && inputRef.current) inputRef.current.focus();
    if (!open) setQuery('');
  }, [open]);

  const results = useMemo<SearchResult[]>(
    () => (open && query.trim().length > 0 ? search(query, 20) : []),
    [open, query],
  );

  if (!open) return null;

  function handleSelect(r: SearchResult) {
    if (r.target.kind === 'layer') {
      selectLayer(r.target.id);
      setRoute('viz');
    } else if (r.target.kind === 'article') {
      setRoute('learn');
      if (typeof window !== 'undefined' && window.location) {
        try {
          window.location.hash = r.target.id;
        } catch {
          /* jsdom edge cases — ignore */
        }
      }
    } else {
      // equation: best-effort — surface the Learn route.
      setRoute('learn');
    }
    setOpen(false);
  }

  function onOverlayClick(e: React.MouseEvent<HTMLDivElement>) {
    if (e.target === overlayRef.current) setOpen(false);
  }

  return (
    <div
      ref={overlayRef}
      className="command-palette-overlay"
      data-testid="command-palette"
      onClick={onOverlayClick}
    >
      <div className="command-palette" role="dialog" aria-label="Concept search">
        <input
          ref={inputRef}
          className="command-palette-input"
          data-testid="command-palette-input"
          type="text"
          placeholder="Search panels, articles, equations…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <ul className="command-palette-results" data-testid="command-palette-results">
          {results.length === 0 && query.trim().length > 0 && (
            <li className="command-palette-empty">No matches.</li>
          )}
          {results.map((r, i) => (
            <li
              key={`${r.target.kind}:${r.target.id}:${i}`}
              className={`command-palette-result command-palette-result-${r.target.kind}`}
              data-testid={`command-palette-result-${r.target.kind}-${r.target.id}`}
              onClick={() => handleSelect(r)}
            >
              <div className="command-palette-result-title">
                <span className="command-palette-kind">{r.target.kind}</span>
                {' · '}
                {r.title}
              </div>
              <div className="command-palette-result-snippet">{r.snippet}</div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
