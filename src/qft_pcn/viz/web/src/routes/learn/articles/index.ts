// src/qft_pcn/viz/web/src/routes/learn/articles/index.ts
/**
 * Registry of all Learn-route articles. Order in this array determines
 * the contents-tree display order. Articles populated by T10 (Orientation
 * + Foundations) and T13-T15 (per-panel + DSL).
 */

import type { ArticleSpec } from '../../../lib/article-types';

// Articles register themselves via re-export. Each article file exports
// a const named `article` of type ArticleSpec.

// IMPORTANT: when adding a new article, import it here AND push it onto
// ARTICLES below. The order is the textbook outline.

export const ARTICLES: ArticleSpec[] = [];
