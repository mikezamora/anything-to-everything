# Visualizer Enhancements

**Audit date**: 2026-05-23
**Audit scope**: post Learn (19 articles) + Training route + tabbed
`ExplainerPane` expansion. Refreshed; prior entries E-1 … E-30 remain accurate
where the underlying code did not change. New entries E-31 … E-37 cover the
Learn/Training/tabbed-explainer surface specifically.

---

## E-31 — Strengths the post-expansion viz now nails

- **Tabbed ExplainerPane is genuinely textbook-grade.** All 13 layers carry
  five tabs (Overview / Math / Worked Example / Training Dynamics / Watch);
  every worked example is hand-computable from the setup; pathology entries
  read like a debugging FAQ rather than boilerplate.
- **Worked-example arithmetic is honest.** §1.3 (F = 0.0134), §3.3
  (g_ab → 0.4976), §4.3 (⟨H⟩ = −0.197 after one Trotter step) all check out
  symbolically and numerically. This is what makes the Learn route trustable
  as a teaching artefact rather than a decorative gloss.
- **Mera-relax explainer carries the §10.10 invariant correctly.** The
  binder-witness `forall_protected_leaves` is named, sourced
  (`mera_encoder:forall_protected_leaves`), and tied to the failure mode that
  matters ("AST text loses 'forall'" → frozen_leaves not threaded). This is
  the model for how every invariant-bearing panel should explain itself.
- **Interpreter heuristics are tiered.** `manifold` thresholds at 0.4 /
  0.02 produce different sentences; `logic` thresholds at 0.05; `multifield`
  at 0.5 / 0.02. The thresholds are not arbitrary — they match the regimes
  the matching panels' tooltips talk about.
- **EQUATIONS role palette is internally consistent.** A reader who learns
  the six-colour key (input / param-learn / param-const / output / state /
  observable) on the first equation can decode the rest without re-reading.

---

## E-32 — Article cross-references that would deepen the textbook

- **foundations-variational-fe → fusion-qpcn.** §1.3 derives F for a Gaussian
  PCN site; §4.3 derives ⟨H⟩ for a single qubit. Articulating that ⟨H⟩
  *plays the role of* F on the quantum side (architecture §3.4 correspondence
  table) would let a reader carry the "one objective, many substrates"
  intuition from PCN → QPCN. Currently §1.3 only forward-refs §4.2.
- **qft-mps → fusion-logic.** The "binder = bond entanglement" claim in
  §4.4 is the load-bearing reason MPS is the right belief carrier for logic.
  §2.1 (MPS) currently treats bond dim as a generic entanglement budget;
  one paragraph saying "in §4.4 we will see that this same budget is what
  realises variable binding" would land the architecture's central
  programming–quantum correspondence earlier.
- **foundations-riemannian → fusion-manifold + fusion-pcn-coupling.** §1.4
  introduces R as "a coordinate-invariant heatmap"; §4.1/§4.2 use R as the
  geometric response to T_μν. A forward reference and a back-reference would
  let a reader scrub the texts without losing thread.
- **pcn-multifield → qft-hamiltonian.** The Yukawa coupling in §3.3 has the
  same algebraic shape as `lambda_ab phi_a phi_b` in the QFT Hamiltonian
  (architecture §3.3.4 H_1). The PCN article notes this once at the end;
  the QFT-side Hamiltonian article does not reciprocate, missing a chance
  to ground the "same gradient drives both" claim.

---

## E-33 — Equations that could enrich existing articles

- **`metric-source-rate` (new)**: `dh_μν/dt = κ_R T_μν[E] − γ h_μν` is the
  rate equation the §4.1 / §4.2 worked examples use, but it is not in
  `EQUATIONS`. Registering it would let `fusion-manifold` and
  `fusion-pcn-coupling` link to a typed equation card instead of inlining
  the formula in prose.
- **`pcn-flow` (new)**: `dPhi/dt = (J_g)^T (Pi E) + D Δ_g Phi + top_down`
  (architecture §3.1) is the master PCN flow rule. Currently
  `pcn-dynamics.tsx` describes it in prose but cites only
  `free-energy-functional`. A dedicated equation card with role-coloured
  Φ / E / Π / D would tie the PCN article to the explainer palette.
- **`stress-energy` (new)**: `T_μν[E] = ∂_μ E ∂_ν E − ½ g_μν |∂E|²`
  (architecture §3.2) is invoked by `fusion-pcn-coupling.tsx` step 1 of
  the worked example. Registering it would let the §1.4 (Riemannian)
  article forward-reference the actual source term rather than describing
  it abstractly.
- **`mps-canonical-form` (new)**: an entry showing the mixed canonical
  decomposition `|ψ⟩ = (left-iso) S (right-iso)` would underpin three
  separate places that currently talk about canonical form informally
  (`qft-mps`, `qft-mera`, `fusion-bridge`).

---

## E-34 — Interpreter heuristics that could sharpen

- **`mps` interpreter**: currently reports χ_max + total entropy. Adding a
  "saturated bond count" (number of bonds at χ_max) would surface the
  pathology the explainer already documents ("Every bond pinned at χ_max
  with a heavy tail").
- **`hamiltonian` interpreter**: ignores step. With access to a per-step
  `frame.layer_state['top_term']`, it could call out which Hamiltonian
  term carries the most ⟨H⟩ this frame — the missing "where is the
  energy?" diagnostic.
- **`pcn-dynamics` interpreter**: currently states only `total F`. The
  explainer's per-layer F decomposition is the most informative diagnostic
  (which layer is the bottleneck) and is shipped in the snapshot
  (`per_layer_free_energy[l]`); the interpreter could surface the argmax-l.
- **`vqc` interpreter**: static (n_qubits × n_layers + boilerplate). With
  `theta_norm` or `theta_var` from the snapshot it could note when
  training has visibly stalled (theta stationary).
- **`bridge` interpreter**: reports only `trotter_steps`. Reporting
  converged flag + final energy from `RunResult` would mirror the
  `qpcn` interpreter and finally make the bridge panel diagnostic without
  cross-referencing.

---

## E-35 — Citation hygiene that would prevent regression

The deviations file lists three specific miscitations (D-13, D-14, D-16) and
one systemic one (D-15). A lightweight enhancement to prevent regression:

- Introduce an `archAnchor` field in `ArticleSpec.citations` whose values
  are validated at build time against a generated index of architecture-doc
  section anchors. The current `label` strings are free-form, so drift
  goes undetected until a human audit catches it.
- Same idea for the explainer's `references` array.

---

## E-36 — Learn-route UX polish

- **Contents tree**: currently linear. A two-pane "outline | reading list"
  split (textbook order on the left, suggested per-panel sequence on the
  right) would let users approach the Learn route either way without
  re-reading the orientation prose.
- **Prerequisite arrows**: `qft-mps` declares prereqs
  `[foundations-hilbert-operators, foundations-vectors-tensors]`, but few
  others do. Filling these in (and rendering them as breadcrumb chips at
  the top of each article) would make the textbook ordering self-documenting.
- **Training narrative**: the Training route walks the per-frame loop
  step by step. One observation: it is currently a single long page; a
  per-step expand/collapse with the matching snapshot field highlighted
  in the panel ribbon below would make it usable as a live debugger
  rather than only as a teaching artefact.
- **Equation hover gloss**: clicking an equation card opens the symbol
  glosses; hovering currently does nothing. A hover preview (the gloss
  string + the role-colour swatch) would shorten the loop for readers
  cross-referencing between articles.

---

## E-37 — Things the architecture itself could clarify (out of viz scope but worth flagging)

- The architecture's coupling-descent rule at §4.5 reads
  `dg_ij/dt = -∂F/∂g_ij = -⟨Phi_i Phi_j⟩_M / Vol(M)`, which (with a
  +g·Phi_i Phi_j convention in F) gives g shrinking on correlated fields
  — opposite to the prose claim "Correlated fields grow their coupling."
  The §3.3 article correctly resolves this by noting the sign convention
  in `multifield.py` puts g into F with a minus sign, but the architecture
  doc itself does not. A single-sentence fix at architecture §4.5 would
  remove the apparent contradiction.
