// src/qft_pcn/viz/web/src/routes/learn/articles/index.ts
/**
 * Registry of all Learn-route articles. Order in this array determines
 * the contents-tree display order. Articles populated by T10 (Orientation
 * + Foundations) and T13-T15 (per-panel + DSL).
 */

import type { ArticleSpec } from '../../../lib/article-types';
import { article as orientation } from './orientation';
import { article as foundationsVectorsTensors } from './foundations-vectors-tensors';
import { article as foundationsHilbertOperators } from './foundations-hilbert-operators';
import { article as foundationsVariationalFe } from './foundations-variational-fe';
import { article as foundationsRiemannian } from './foundations-riemannian';

// IMPORTANT: when adding a new article, import it here AND push it onto
// ARTICLES below. The order is the textbook outline.

export const ARTICLES: ArticleSpec[] = [
  orientation,
  foundationsVectorsTensors,
  foundationsHilbertOperators,
  foundationsVariationalFe,
  foundationsRiemannian,
];
