# Audit Pass 1 — §12 physics extensions — HEAD 2b6fd56

Read-only audit per `memory/qpcn-completion-protocol.md`. All 18 §12.x
implementations exist. Knowns in EXTENSIONS.md (A.1–A.5, the §12.6
Hessian, §12.13 meet-in-the-middle, §12.17 multi-cycle GNVW, §12.11
encoder corpus, §12.16 partial, §12.9 complex-block, §12.2 List-substrate
xfail) are NOT re-catalogued.

## ENHANCEMENTS (productive divergence)

- §12.1 anomaly — the spec's ABJ trace `Tr(T^a {T^b, T^c})` collapses,
  on diagonal leaf-local projector triples, to the operator-algebraic
  type-mismatch projector `<P_kind · (I - P_type) · P_value>`. The
  module makes this collapse explicit in its docstring and adds the
  `type_leaf` index field to `SymmetryGenerator` to guard against
  encoding-basis collisions (e.g. `KIND_ZERO == TYPE_NAT == 8`). The
  collision guard is genuine substrate hardening beyond the spec text.
- §12.8 dynamical PT — adds the `"relative"` non-analyticity trigger
  alongside the spec's `"absolute"` zero-crossing check, and surfaces
  both in `DPTEvent.trigger`. Faithful to Heyl 2018's
  "non-analyticity in the rate" reading of finite-N DPTs.
- §12.10 holographic compilation — wires the §12.2 Wilson-signature
  oracle (`verify_layer_equivalence`) and the §12.8 DPT detector
  (`detect_compilation_convergence`) into a single compilation-pipeline
  API. The cross-extension stack is productive even though no
  optimization pass ships (see DEVIATIONS).
- §12.15 Witten index — derives the Z₂ grading from
  `term_affected_leaves` (substrate causal-cone footprint parity).
  This is operator-algebraic (§1.1), not an AST-symbol parity.
- §12.18 Noether discovery — library covariance `C = Σ |fp⟩⟨fp|` +
  degenerate-eigenspace SO(r) generator construction is a clean
  operator-algebraic realization of "structural symmetry of the
  library." Honest scope: it discovers symmetries of lemma
  fingerprints, not of dynamical Hamiltonians.

## DEVIATIONS (must fix)

- §12.3 topological_degeneracy — `topological_degeneracy.py:233-273`
  (`count_proof_strategies`) — the module claims to count "essentially
  different proof strategies" via `K^g = 2^{b1*g}` of the *binding
  diagram's* first Betti number, but `b1` of the AST binding graph
  (tree edges ∪ use→binder edges) is generally **0 or 1** for closed
  programs and **bears no relation to the spec's acceptance targets**
  (`a+b=b+a` ⇒ 2 proofs, `(a+b)+c=a+(b+c)` ⇒ 1 proof, pumping-lemma
  variants). The implementation does not deliver the spec's claim:
  the binding-diagram Betti number is a topological invariant of the
  variable-binding graph, not the proof-homotopy count. Fix scope:
  either (a) re-anchor the count to the constraint-Hamiltonian's
  actual ground-state degeneracy (Wilson-loop algebra on **H**, not on
  the AST graph), or (b) honestly rename the function /
  WilsonLoopAlgebra to "binding-cycle Z₂ algebra dimension" and
  scope-limit the docstring to that observable.

- §12.4 bootstrap — `bootstrap.py:355-426`
  (`verify_typing_via_bootstrap`) — the spec capability is "derive
  performance bounds, side-effect classes, complexity bounds from the
  type signature alone" (acceptance: `f : List Int → Int` ⇒
  termination ≥ Ω(1), depth ≥ log(length); polymorphic `List a → List a`
  ⇒ parametricity). The implementation does NONE of this: it builds an
  SDP whose diagonal `X[i,i] == o_i` is fixed by the §12.1 anomaly
  trace and whose feasibility is exactly equivalent to "all anomaly
  diagonals fit under the truncation gap." This is a re-skinning of
  §12.1's well-typed/ill-typed split as an SDP. `dimension_bound` is
  `max(o_i)`, not an OPE-dimension bound on any physical observable.
  No bound on termination, depth, complexity, or side-effects is
  produced. Fix scope: either implement a real type-derived bootstrap
  SDP whose extremal value bounds a meaningful complexity observable
  (depth, length, OPE gap on the type's category), or honestly retitle
  the module as "SDP-form anomaly check" and document that the spec's
  capability acceptance is deferred.

- §12.7 replica_complexity — `replica_complexity.py:204-301`
  (`_default_hamiltonian` + `instance_partition_function`) — the
  spec's partition function is "count of valid proofs weighted by
  their complexity" (proof-space mass). The implementation computes
  `Z = geometric_mean_k tr(ρ_k · expm(-β · h_k))` where `h_k` is the
  `local_op` of a **generic FieldSpecies bosonic number operator**
  (bare_mass=1.0, kinetic=0.5) — a Hamiltonian unrelated to the proof
  search, the constraint algebra, or the theorem class. The resulting
  Z measures `<exp(-β · number_operator)>` against the encoded state's
  leaf marginal — not a proof-space partition function. The replica
  continuation then operates on a quantity disconnected from the
  spec's `<log Z>` semantics. Fix scope: replace `_default_hamiltonian`
  with the real proof-class constraint Hamiltonian (or
  `MeraEvalHamiltonian` / `MeraTypingHamiltonian` of the instance) so
  Z reflects proof-space mass; or scope-limit the docstring to admit
  that the current Z is a leaf-marginal observable proxy, not the
  spec's complexity-weighted proof-count.

- §12.10 holographic_compilation — `holographic_compilation.py:102-187`
  (`compile_to_mera_layers` + `verify_layer_equivalence`) — the spec
  capability is "compiler optimizations provably semantics-preserving
  by construction" with concrete operations (lowering, optimization,
  inlining, constant folding) realized as MERA-layer passes that
  discover optimization opportunities through RG flow. The
  implementation ships ONLY a layer-iteration record (each
  `CompilationLayer` carries handles to the same encoded state and
  meta — line 141 `state=state` is shared across all layers in a
  compilation) and a tautological `verify_layer_equivalence` (any two
  layers from one compilation share the same `state`, so their
  Wilson-signatures are bitwise equal — the test cannot fail). No
  optimization pass (lowering, inlining, constant folding) is
  implemented; no semantics-preserving rewrite is exercised; no
  optimization opportunity is discovered. Fix scope: either implement
  at least one concrete MERA-layer optimization pass (e.g. a
  disentangler simplification that demonstrably preserves the Wilson
  signature on a non-trivial pair of pre/post states), or honestly
  rename the module to "MERA-layer iteration record" and document
  that the §12.10 optimization capability is deferred.

- §12.12 quantum_extremal_surface — `quantum_extremal_surface.py:342-411`
  (`find_minimum_complexity_proof`) — the spec capability is "*a
  priori* prediction of proof complexity from theorem geometry alone,
  before doing any proof search" (a true lower bound that lets the
  architecture reject geometrically-impossible theorems without
  search). The implementation requires the caller to supply already-
  encoded **candidate proof MERAs** and only ranks them by their own
  midline entanglement entropy — i.e. it is a post-hoc ranking of
  proofs that have already been found, not an a-priori bound from the
  theorem statement. The capability ("get a lower bound *before*
  searching") is not delivered. `compute_qes_complexity` on the
  theorem state alone is the closest surface, but `find_minimum_*`
  requires the proofs in hand. Fix scope: either ship a routine that
  computes the QES lower bound from the theorem state alone (no
  candidates needed) and surfaces it as a search-budget gate, or
  rename `find_minimum_complexity_proof` to
  `rank_candidate_proofs_by_qes` and document that the §12.12
  a-priori-prediction capability is deferred.

