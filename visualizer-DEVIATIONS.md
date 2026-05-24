# Visualizer Deviations from QFT_PCN_ARCHITECTURE.md

**Audit date**: 2026-05-23 (post D-12..D-16 resolution)
**Audit scope**: post Learn (19 articles) + Training route + tabbed `ExplainerPane`
(13 layers × 5 tabs) + `EQUATIONS` registry (12 entries) + `INTERPRETERS`
(13 functions) + per-panel content.

**Methodology**: Re-walked the same files audited in the previous round
(`src/qft_pcn/viz/web/src/lib/{equations,explainer,interpreters}.ts`,
`src/qft_pcn/viz/web/src/routes/learn/articles/*.tsx`, panel modules) and
re-verified each architecture-doc section citation against
`QFT_PCN_ARCHITECTURE.md`, including a grep for residual `§1.1` invocations
in the binding-as-entanglement sense.

---

## Status: clean

D-12..D-16 from the previous round have been resolved in commits
39c5f2f, f30507a, 9a34e9c, a850895, b5a41a6 respectively. The follow-on
re-audit additionally caught and fixed three residual binding-as-entanglement
`§1.1` citations in `fusion-mera-relax.tsx` and a stale `§1.1` reference in
`fusion-manifold.tsx` (rolled into this audit-refresh commit).

Remaining `§1.1` mentions in the viz tree (verified non-deviant):
- `orientation.tsx:25` — refers to the **Learn route's own** §1.1 (Foundations).
- `foundations-vectors-tensors.tsx:3` — header comment for the Learn route's
  own §1.1 (Foundations — Vectors & Tensors).
- `foundations-hilbert-operators.tsx:38` — refers to the Learn route's own
  §1.1 (tensor-network ansätze topic).

None of these point to the architecture document; they are intra-Learn
outline references and are correctly numbered against the Learn route's
own table of contents.

No new deviations of D-12..D-16 severity were found.
