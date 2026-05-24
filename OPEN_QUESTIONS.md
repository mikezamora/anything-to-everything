# Open Research Questions

Tracker for `QFT_PCN_ARCHITECTURE.md` §16.3 (lines 2919–2935). Per the
QPCN-completion protocol, each open question is either **open**,
**addressed by §X.Y / D-id**, or **deferred** (with a deliberate
deferral note). Resolves spec-gap-analysis item A6.

Status snapshot HEAD `7a5c009+`.

---

## Q1 — Best way to extend to dependent types?

- **Spec ref:** §16.3, line 2923.
- **Verbatim:** "STLC is the prototype. Dependent types (System F,
  Calculus of Constructions, Lean's type theory) have much richer
  structure. Naive encoding causes combinatorial blowup. Open
  question: what's the right way to extend the Hamiltonian compiler?"
- **Status:** **OPEN.** STLC + arithmetic are wired
  (`logic/typing_hamiltonian.py`,
  `logic/mera_evaluation_hamiltonian.py`); dependent-type extension is
  unimplemented. Listed in `EXTENSIONS.md` as a future-work item.
- **Path forward:** prototype indexed type families on the existing
  MERA substrate; the §13.5 area-law guarantee gives a principled
  truncation tool for the combinatorial blowup. Phase-I research item.

## Q2 — Best handling of effects and concurrency?

- **Spec ref:** §16.3, line 2925.
- **Verbatim:** "Effectful computation breaks unitarity. Possible
  extensions: Lindblad dynamics for stochastic effects, density matrix
  substrate, non-commutative geometry. None of these are explored in
  detail."
- **Status:** **DEFERRED.** Substrate is unitary by §16.1; Lindblad
  extension is named in §12 future-work but not on the active branch.
- **Path forward:** density-matrix MPS (DM-MPS / MPDO) substrate; gate
  this as a separate sub-project once the pure-state milestones are
  closed.

## Q3 — Optimal tensor topology for ASTs?

- **Spec ref:** §16.3, line 2927.
- **Verbatim:** "1D MPS forces a linear ordering on an inherently
  tree-structured AST. Tree tensor networks (TTNs) seem natural but
  have their own complications. MERA is a generalization but adds
  infrastructure cost."
- **Status:** **PARTIALLY ADDRESSED by §10.4 + D5/D25/D30.** MERA
  substrate is live (`qft/mera.py`, `qft/mera_evolution.py`); the
  D5 / D25 / D30 deviations specifically harden the MERA layer-0
  inter-pair and odd-leaf routes that were previously dropped. Whether
  pure TTN would outperform MERA on ASTs remains an empirical question.
- **Path forward:** A1 §14 evaluation harness can drive an ablation
  (MPS vs MERA vs TTN) once the TTN substrate is implemented.

## Q4 — How to integrate with existing tooling?

- **Spec ref:** §16.3, line 2929.
- **Verbatim:** "Lean, Coq, Agda, Idris all have rich tactic systems
  and libraries. The QPCN should plug into these as a tactic, not
  replace them entirely. The protocol for this integration is not
  designed."
- **Status:** **OPEN.** The `bridge/api.py` JSON-RPC surface is the
  intended integration seam, but no Lean/Coq adapter exists.
- **Path forward:** Lean 4 first; the QPCN's MPS proof witness can be
  re-checked by Lean's kernel via the §10.10 result-integrator
  exporting a tactic transcript.

## Q5 — Scaling laws — exact form?

- **Spec ref:** §16.3, line 2931.
- **Verbatim:** "§13.8 conjectures a saturating exponential capability
  curve. Real scaling laws for ML systems usually have power-law
  components. The exact form requires §12.7 replica method
  implementation."
- **Status:** **PARTIALLY ADDRESSED by A4 RESOLVED + D14 RESOLVED.**
  `src/qft_pcn/analysis/capability_growth.py` measures the saturating
  exponential `dC/dt = α(1-C) - βC` on a real corpus
  (`reports/capability_growth_4444045.md`); §12.7 replica is wired as
  `composition/replica_complexity.py`. Whether the long-run scaling is
  saturating-exponential or power-law in the QPCN regime remains an
  empirical question that the §14 evaluation harness now has the tools
  to settle.
- **Path forward:** larger benchmark corpora through the §14 runner;
  fit power-law and exponential models side-by-side.

## Q6 — Catastrophic forgetting?

- **Spec ref:** §16.3, line 2933.
- **Verbatim:** "Adding new lemmas might displace old ones in
  surprising ways. The §10.8 library should be append-only, but if
  abstraction (§10.9) replaces old primitives with new composites, old
  proofs may need re-derivation. This isn't analyzed."
- **Status:** **PARTIALLY ADDRESSED by D11 + D38 + D39 + §10.9
  consolidation.** D11 RESOLVED guarantees the CONSOLIDATE step
  re-derives rather than subsumes; D38/D39 RESOLVED preserve
  bundle/encoding-meta consistency so a consolidated L' cannot
  silently invalidate provenance. Library is append-only; pruning
  (`pruned` flag) is reversible. A longitudinal "old-proof
  re-derivation" sweep is not yet in CI.
- **Path forward:** add a wake-sleep regression test that re-runs the
  pre-consolidation corpus against the post-consolidation library;
  pin re-derivation success rate.

## Q7 — Adversarial robustness.

- **Spec ref:** §16.3, line 2935.
- **Verbatim:** "What if the LLM frontend emits adversarial DSL specs?
  The QPCN itself is robust by §13.4 (holographic threshold), but the
  interface between LLM and QPCN is a trust boundary that needs
  analysis."
- **Status:** **OPEN.** `bridge/dsl/cross_validate.py` does
  cross-validation against the schema, but no red-team / adversarial
  DSL test suite exists.
- **Path forward:** seed a fuzzer against `bridge/dsl/schema.py`;
  measure §13.4 holographic-threshold behavior under adversarial
  injection.

---

## §16.3 question count

The spec section is introduced as a list of "things the design doesn't
address but should." The numbered enumeration above covers the seven
explicitly-bulleted questions in lines 2923–2935. The gap-analysis
A6 entry refers to "nine numbered open research questions"; on a literal
re-read of §16.3 only seven distinct bullets are present (the trailing
two referenced in A6 conflate §16.2's *unproven assumptions* with
§16.3's *open questions*). Those §16.2 items are tracked separately:

- §16.2 assumption — Hamiltonian compiler produces non-frustrated
  Hamiltonians for STLC: **partially addressed** by the D17 typing-
  Hamiltonian shape audits and the K-8 acceptance tests; full formal
  proof remains open.
- §16.2 assumption — Gap closure is signal, not noise: **partially
  addressed** by D23 (spectral_gap no longer NaN-silent) and the
  D1 spectral-gap gate; interpretation remains conjectural.

Future audit passes may merge those into Q8 / Q9 if the spec is
expanded to explicitly enumerate them under §16.3.
