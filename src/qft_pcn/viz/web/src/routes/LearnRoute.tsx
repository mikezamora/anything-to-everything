// src/qft_pcn/viz/web/src/routes/LearnRoute.tsx
/**
 * Two-region Learn route: contents tree (left) + article body (right).
 * Default active article: the Orientation chapter (§0).
 */

import { useState } from 'react';
import { LearnContents } from './learn/LearnContents';
import { LearnArticle } from './learn/LearnArticle';
import { ARTICLES } from './learn/articles';

export function LearnRoute() {
  const defaultId = ARTICLES[0]?.id ?? '';
  const [activeId, setActiveId] = useState(defaultId);
  return (
    <div className="learn-route">
      <LearnContents activeId={activeId} onSelect={setActiveId} />
      <main className="learn-route-body">
        <LearnArticle articleId={activeId} />
      </main>
    </div>
  );
}
