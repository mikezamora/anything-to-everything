// src/qft_pcn/viz/web/src/routes/learn/LearnContents.tsx
/**
 * Left contents tree for the Learn route. Groups registered articles
 * by their first sectionPath entry (the "chapter") so the tree mirrors
 * the textbook outline. Clicking an article entry sets the active id.
 */

import { ARTICLES } from './articles';

interface Props {
  activeId: string | null;
  onSelect: (id: string) => void;
}

export function LearnContents({ activeId, onSelect }: Props) {
  // Group by first sectionPath entry, preserving registration order
  // (which we author to match the textbook outline §0 → §5).
  const chapters: Record<string, typeof ARTICLES> = {};
  for (const art of ARTICLES) {
    const ch = art.sectionPath[0] ?? '(uncategorised)';
    (chapters[ch] ??= []).push(art);
  }

  return (
    <nav className="learn-contents">
      {Object.entries(chapters).map(([chapter, articles]) => (
        <div key={chapter} className="learn-contents-chapter">
          <h3>{chapter}</h3>
          <ul>
            {articles.map((art) => (
              <li key={art.id}>
                <button
                  type="button"
                  className={art.id === activeId ? 'active' : ''}
                  onClick={() => onSelect(art.id)}
                >
                  {art.sectionPath.slice(1).join(' · ') || art.title}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}
