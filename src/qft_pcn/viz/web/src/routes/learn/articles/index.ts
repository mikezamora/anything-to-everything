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
import { article as qftMps } from './qft-mps';
import { article as qftMera } from './qft-mera';
import { article as qftHamiltonian } from './qft-hamiltonian';
import { article as qftVqc } from './qft-vqc';
import { article as pcnFields } from './pcn-fields';
import { article as pcnDynamics } from './pcn-dynamics';
import { article as pcnMultifield } from './pcn-multifield';
import { article as fusionManifold } from './fusion-manifold';
import { article as fusionPcnCoupling } from './fusion-pcn-coupling';
import { article as fusionQpcn } from './fusion-qpcn';
import { article as fusionLogic } from './fusion-logic';
import { article as fusionMeraRelax } from './fusion-mera-relax';
import { article as fusionBridge } from './fusion-bridge';
import { article as dslWalkthrough } from './dsl-walkthrough';

// IMPORTANT: when adding a new article, import it here AND push it onto
// ARTICLES below. The order is the textbook outline.

export const ARTICLES: ArticleSpec[] = [
  orientation,
  foundationsVectorsTensors,
  foundationsHilbertOperators,
  foundationsVariationalFe,
  foundationsRiemannian,
  qftMps,
  qftMera,
  qftHamiltonian,
  qftVqc,
  pcnFields,
  pcnDynamics,
  pcnMultifield,
  fusionManifold,
  fusionPcnCoupling,
  fusionQpcn,
  fusionLogic,
  fusionMeraRelax,
  fusionBridge,
  dslWalkthrough,
];
