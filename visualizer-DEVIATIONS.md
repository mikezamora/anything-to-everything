# Visualizer Deviations from QFT_PCN_ARCHITECTURE.md

**Audit date**: 2026-05-23
**Audit scope**: post Learn (19 articles) + Training route + tabbed `ExplainerPane`
(13 layers × 5 tabs) + `EQUATIONS` registry (12 entries) + `INTERPRETERS`
(13 functions) + per-panel content.

**Methodology**: Read `QFT_PCN_ARCHITECTURE.md` for ground truth, then walked
`src/qft_pcn/viz/web/src/lib/{equations,explainer,interpreters}.ts`,
`src/qft_pcn/viz/web/src/routes/learn/articles/*.tsx`, and the panel modules.
Cross-checked architecture section numbers (§), hand-computed every worked-example
arithmetic chain, and verified each `Role` annotation on `EQUATIONS` symbols
against the architecture's own state/parameter/observable taxonomy.

Cap per protocol: top 5 deviations of comparable severity. They are listed in
descending order of severity and are queued for the follow-up fix loop.

---

## D-12 — qft-mera worked example: χ=4 parameter count is silently wrong

**File**: `src/qft_pcn/viz/web/src/routes/learn/articles/qft-mera.tsx:103`

The §2.2 MERA article computes a χ=2 parameter count correctly (16+24+2 = 42 real
DOF), then claims an analogous χ=4 count:

> "At χ = 4 the same count balloons to roughly 4² + 2 · (2 · 4 · 4 - 4²)
> + (2 · 4 - 2) = 16 + 32 + 6 = 54"

This treats the disentangler at χ=4 as if it still acted on a 4-dimensional space.
A disentangler at χ=χ' acts on χ'² × χ'² space, i.e. a 16×16 unitary at χ=4,
so its real DOF is dim U(16) = 256, not 16. Similarly an isometry from two
χ=4 sites to one χ=4 site is a 4×16 partial isometry living on the complex
Stiefel V_4(C^16), with real dimension 2·4·16 − 4² = 112, not 16. The "balloons
to 54" sentence understates the true count by an order of magnitude and
contradicts the article's own "doubling χ multiplies per-tensor parameter count
by 8-to-16" line in the same chapter. Architecture §2.3 (MPS bond cost is
O(χ²), generalises to χ³–χ⁹ for MERA contractions) implies the correct scaling
that the article itself elsewhere acknowledges.

**Fix**: Recompute at χ=4 honestly, or drop the worked numeric comparison and
present the scaling law (`disentangler ~ χ⁴, isometry ~ χ³`) verbally.

---

## D-13 — fusion-qpcn citation: "QFT_PCN_ARCHITECTURE.md §5 (QPCN)" is a wrong section

**File**: `src/qft_pcn/viz/web/src/routes/learn/articles/fusion-qpcn.tsx:132`

The article's `citations` array points to architecture §5 for the QPCN. But
`QFT_PCN_ARCHITECTURE.md §5` is "Tests and Verification" — it has no QPCN
material. The QPCN is described in architecture §2.3 (overview) and §4.7.5
(implementation in `qft/qpcn.py`).

Because the citations block uses an explicit `href` to the architecture file,
this is unambiguous miscitation, not an intra-viz outline reference.

**Fix**: Replace with `§2.3 + §4.7.5 (QPCN — quantum predictive coder)`.

---

## D-14 — pcn-multifield citation: "§3.3 (Multi-field generalisation)" is a wrong section

**File**: `src/qft_pcn/viz/web/src/routes/learn/articles/pcn-multifield.tsx:125`

The article cites `QFT_PCN_ARCHITECTURE.md §3.3 (Multi-field generalisation)`,
but architecture §3.3 is "Quantum field theory primitives" (Fock space, MPS,
Hamiltonian, TEBD, energy expectation). Multi-field lives in architecture §2.2
(overview) and §4.5 (`MultiFieldNetwork`). The §4.5 citation in the same array
is correct; the §3.3 one is not.

**Fix**: Replace `§3.3` with `§2.2 (Multi-field layer)`.

---

## D-15 — `§1.1` invoked as the variable-binding-as-entanglement principle, but architecture §1.1 is "Target domains"

**Files**:
- `src/qft_pcn/viz/web/src/lib/explainer.ts:213` (mera_relax)
- `src/qft_pcn/viz/web/src/lib/explainer.ts:308, 318` (logic)
- `src/qft_pcn/viz/web/src/panels/LogicPanel.tsx:10, 48, 248`
- `src/qft_pcn/viz/web/src/panels/LogicPanel.test.tsx:47`
- `src/qft_pcn/viz/web/src/panels/MeraRelaxPanel.tsx:11`
- `src/qft_pcn/viz/web/src/routes/learn/articles/fusion-logic.tsx:19`

Multiple load-bearing claims in panels and the §4.4 Logic article cite
"§1.1 invariant — variable binding equals bond entanglement, never classical
lookup." But `QFT_PCN_ARCHITECTURE.md §1.1` is `Target domains`. The
variable-binding-as-entanglement principle lives in architecture **§8.1** ("The
deep correspondence: variable binding *is* entanglement"). The Learn route's
own internal §1.1 is `Foundations — Vectors & Tensors` (per
`foundations-vectors-tensors.tsx:12`), so the citation also fails to land
inside the viz's own numbering.

The cited "§1.1 architecture-soul note" appears to be inherited from a
historical user-memory convention; nothing in the current `QFT_PCN_ARCHITECTURE.md`
or in the Learn outline assigns this directive that section number. Because
the principle is genuinely load-bearing for the Logic + MeraRelax panels, the
miscitation matters: a reader following the link arrives at the wrong section.

**Fix**: Either (a) globally rename all "§1.1" invocations of binding=entanglement
to "§8.1" (the architecture-doc anchor), or (b) introduce an explicit Learn
article with id `architecture-soul` at §1.0 and update citations to land there.

---

## D-16 — fusion-pcn-coupling: paragraph invokes "§1.1 of the architecture: belief and evidence are not different kinds of thing" — no such claim in §1.1

**File**: `src/qft_pcn/viz/web/src/routes/learn/articles/fusion-pcn-coupling.tsx:61`

The article says:

> "The architectural reason for this insistence is in §1.1 of the architecture:
> *belief* and *evidence* are not different kinds of thing in the QPCN — they
> are two views of the same underlying entanglement structure."

Architecture §1.1 is `Target domains`. It contains no statement about belief
vs. evidence; the closest claim in the architecture about the loop's
symmetry is in §2.1 / §3.4 (the bidirectional coupling table) and §3.1
(F as the single objective both sides minimise). Attributing a load-bearing
architectural claim to a section that does not contain it weakens the
article's reliability for a reader who actually opens the architecture doc.

**Fix**: Drop the spurious "§1.1 of the architecture" qualifier or replace
with `§3.4 (PCN-QFT correspondence table) + §3.1 (single F objective)`.

---

## Audit-clean items (verified, no deviation)

- The §1.3 free-energy worked example (`foundations-variational-fe.tsx`)
  arithmetic checks out end-to-end (F_site = 0.36 − 0.3466 ≈ 0.0134,
  dF/dPhi = 1.2, dF/dPi = −0.07).
- The §4.3 single-qubit imag-time worked example (`fusion-qpcn.tsx`)
  arithmetic checks out (cosh 0.1 ≈ 1.005, normalised state (0.995, −0.099),
  ⟨H⟩ = −0.197 after one step).
- The §3.3 cross-field worked example (`pcn-multifield.tsx`) arithmetic
  checks out (delta_a = −0.15, delta_b = 0.4, g_{ab} → 0.4976).
- The §4.1 metric-source worked example (`fusion-pcn-coupling.tsx`)
  steady-state derivation h_xx* = κ_R · A² / (2 D) matches the rate-equation
  set-up.
- Every `Role` annotation in `EQUATIONS` (input / param-learn / param-const
  / output / state / observable) is consistent with the architecture's
  taxonomy: e.g. `Phi`/`E`/`g`/`|psi>`/`rho_A` as `state`,
  `h_mu_nu`/`g_ij`/`theta`/`A^{s_k}` as `param-learn`, `eta_mu_nu`/`eta` as
  `param-const`, `<O>` as `observable`.
- All 13 `INTERPRETERS` reads are field-name-correct against the matching
  `snapshot_*` payloads documented in the explainer `code:` fields; sentences
  are accurate proxies for the underlying scalar (e.g. `total_energy`
  threshold-coded for "nearly normal form" matches the H_eval relaxation).
- The §10.10 MeraRelax explainer correctly identifies `forall_protected_leaves`
  as the load-bearing invariant under `mera_trotter_step(...,
  frozen_leaves=...)`, with no overclaim relative to architecture §10.10 /
  §10.11.

The remaining articles (orientation, foundations-vectors-tensors,
foundations-hilbert-operators, qft-hamiltonian, qft-vqc, pcn-fields,
pcn-dynamics, fusion-manifold, fusion-mera-relax, dsl-walkthrough) were spot-
checked for arch alignment and surfaced no deviations of the same severity as
D-12..D-16.
