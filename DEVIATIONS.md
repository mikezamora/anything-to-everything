# QPCN DEVIATIONS (audit pass 1, HEAD 2b6fd56)

> **D1, D2, D7 RESOLVED** — see entries below for SHA + commit reference.

Anti-patterns / placeholders / broken logic / spec-vs-impl gaps that MUST
be fixed before QPCN can be claimed complete. Sourced from terminal audit
pass 1 (four parallel agents over §1-§5, §6-§9, §10, §12). Known items
already catalogued in `EXTENSIONS.md` are not re-listed here.

## CRITICAL (production-impacting)

### D1 — §6.3 spectral_gap gate refuses every production child — **RESOLVED**
- Location: `src/qft_pcn/composition/result_integrator.py` (reads
  `child_result.run_diagnostic.get("spectral_gap", 0.0)`); producer
  `src/qft_pcn/bridge/runtime/result.py:49-65` (`RunResult.to_dict`)
- Spec: §6.3, §9.3
- Issue: `RunResult.to_dict()` emits `energy`, `energy_per_term`,
  `final_bond_dimensions`, `converged`, `trotter_steps` — no
  `spectral_gap` field. The integrator default 0.0 makes the gate refuse
  every real child; every test stubs `run_diagnostic={"spectral_gap":
  1.0, ...}` to pass. In production via `solve_goal_graph`, every child
  hits `gap < GROUND_STATE_GAP` and is refused as "near-degenerate".
- Fix scope: medium — surface a real spectral-gap diagnostic from the
  bridge runtime (probe first excited state after relaxation, or carry
  the MERA-evolver gap through `RunResult.to_dict()`).
- Audit source: §6-§9

### D2 — §8 cache-hit by `goal_id` never fires — **RESOLVED**
- Location: `src/qft_pcn/composition/dispatcher.py::dispatch_siblings`,
  `src/qft_pcn/composition/orchestrator.py::solve_goal_graph`,
  `src/qft_pcn/composition/lemma_library.py`
- Spec: §8
- Issue: Spec mandates pre-dispatch query of `lemma_library` keyed by
  child `goal_id`; if cached, resolve to lemma with no child run. No
  such lookup exists; `LemmaLibrary` has no `goal_id -> Lemma` reverse
  index. `_content_id` mixes `source_run_id` into the hash but provides
  no inverse map. Siblings re-prove identical sub-goals every §10.11
  hierarchical run.
- Fix scope: medium — add `find_by_goal_id` index and pre-dispatch
  lookup that synthesizes a SOLVED `ChildResult`.
- Audit source: §6-§9

### D3 — §8 lazy re-evaluation of provisional / boundary-changed lemmas absent — **RESOLVED**
- Location: `src/qft_pcn/composition/orchestrator.py::solve_goal_graph`
  (pre-solve hook); `src/qft_pcn/composition/lemma_library.py::
  LemmaLibrary.re_evaluate_provisional`
- Spec: §8, §6.3
- Resolution: `LemmaLibrary.re_evaluate_provisional(energy_fn, *,
  residual_gate, ceiling)` walks every lemma with
  `derivation.conditional=True`, calls the caller-supplied `energy_fn`
  to recompute the residual, and promotes (`conditional=False` rewrite)
  / drops (manifest+npz eviction) / refreshes accordingly. The
  orchestrator invokes the hook before the solve loop when called with
  `provisional_energy_fn=`. Tests cover the drop / promote / mid-band /
  non-provisional-untouched / resolver-returns-None branches.

### D4 — Orchestrator does not plumb LLM reviser; principled path unreachable — **RESOLVED**
- Location: `src/qft_pcn/composition/orchestrator.py::solve_goal_graph`
  (signature now carries `llm_reviser=None`; forwarded at the
  `revise(...)` call site inside the revision loop),
  `src/qft_pcn/composition/revision.py` (LLM branch unchanged)
- Spec: §10, §6.5
- Resolution: `solve_goal_graph` now accepts an optional
  `llm_reviser` kwarg conforming to the `revision.LLMReviser` Protocol
  and passes it through as `revise(node, llm=llm_reviser, ...)`. When
  `None` (default) the deterministic `HeuristicReviser` catalogue
  drives revision exactly as before; when supplied, the LLM oracle is
  consulted first per spec §10. Unit test in
  `src/qft_pcn/composition/tests/test_revision.py`
  (`test_llm_reviser_branch_fires_when_supplied_via_orchestrator`)
  pins the orchestrator signature and the LLM-branch wiring.
- Audit source: §6-§9

### D5 — §6.3 `_ascend_one_layer` silently drops non-identity inter-pair disentanglers
- Location: `src/qft_pcn/qft/mera.py:883-918`,
  interaction with `apply_two_site_gate` at `mera.py:1329-1357`,
  `_materialize` guard at `mera.py:1265, 1272`
- Spec: §5.4 (Vidal 2008 §III.5 full ascending superoperator)
- Issue: Docstring asserts simplification holds because inter-pair
  disentanglers are identity — true on product/vacuum but
  `apply_two_site_gate` writes SVD-truncated, possibly non-unitary
  `recon4` into `inter_disentanglers[0][j]`. After any imag-time Trotter
  step on an odd bond, `local_expectation` / `_ascend_one_layer` drops
  the inter-pair disentangler from the causal cone, biasing the result.
  Cross-check via `_materialize` is unreachable
  (`NotImplementedError`), so the bias is unobserved.
- Fix scope: medium — (a) fold relevant adjacent
  `inter_disentanglers[ell][j_inter]` into layer-0 ascent (principled),
  or (b) hard-guard `local_expectation` / `two_site_expectation` to
  refuse when any `inter_disentanglers[0]` slot is non-identity
  (honest-fail).
- Audit source: §1-§5

## OPERATIONAL

### D6 — §9.5 monotonicity observed but not enforced — **RESOLVED**
- Location: `src/qft_pcn/composition/orchestrator.py::solve_goal_graph`
  + helper `src/qft_pcn/composition/goal_graph.py::make_monotonicity_tracker`
- Spec: §9.5, §13.5
- Issue: Spec: "K asserts this monotone decrease... a non-monotone step
  is a bug in the integrator." Orchestrator emits `F_hierarchy` via
  optional `on_step` callback but has no in-line assertion or refusal
  when a step increases F. Premature-clamp inflations pass silently.
- Resolution: `make_monotonicity_tracker(strict=True)` raises
  `MonotonicityViolation` (a `GoalGraphError` subclass) when
  `F_new > F_prev + 1e-6`. `solve_goal_graph` wraps the caller's
  `on_step` in this tracker by default; the new `enforce_monotonicity`
  parameter (default `True`) lets production callers downgrade to
  record-only for diagnostic replay while preserving the user callback.
  Tests `test_free_energy_assertion_fires_on_violation` and
  `test_free_energy_assertion_disabled_on_strict_false` pin both modes.
- Audit source: §6-§9

### D7 — `_frontier_priority` is structural fan-out proxy, not §5.3 precision-weighted schedule — **RESOLVED**
- Location: `src/qft_pcn/composition/orchestrator.py` (docstring
  acknowledges; flagged as follow-on per §5.3, lines 91-101)
- Spec: §5.3
- Issue: Spec permits a fallback only when precision estimates are
  unavailable, and "the code must say so explicitly". A leaf's
  precision / hole-density estimate is computable from the DSL spec at
  no significant cost. Not in EXTENSIONS.md.
- Fix scope: small — either record an EXTENSIONS entry deferring with
  a named dependency or compute a precision proxy from `node.goal.dsl_spec`
  and use it in `sorted(..., key=...)`.
- Audit source: §6-§9, §10 (dual-flagged)

### D8 — `revision.HeuristicReviser` emits content-identical sub-goals modulo metadata — **RESOLVED**
- Location: `src/qft_pcn/composition/revision.py` (`HeuristicReviser._build`)
- Spec: §6.5
- Resolution: Each catalogue entry now produces a SUBSTRATE-distinct
  decomposition rather than a metadata tag swap:
  * `swap_induction_variable` — single child, same footprint, with an
    explicit `induction_axis=flipped` flag mirrored into the
    `boundary` so the downstream compiler reads a different problem.
  * `split_conjunction_other_way` — TWO children covering disjoint
    halves of `node.goal.parent_leaves` (sub-goal count is a
    structural, compiler-visible difference). Skipped when the
    parent footprint has fewer than 2 leaves.
  * `strengthen_induction_hypothesis` — single child whose
    `parent_leaves` is widened by one fresh leaf (footprint extension
    is substrate-level). Skipped on empty footprints.
  When none of the remaining catalogue entries are substrate-feasible
  for the failing node, `HeuristicReviser.decompose` returns `[]` and
  `revise(...)` lifts that empty result so the orchestrator surfaces
  `RevisionExhausted` — no looping on identical work. Unit tests
  (`test_heuristic_reviser_returns_substrate_different_decompositions`,
  `test_heuristic_reviser_signals_exhausted_when_no_variation_possible`)
  pin the new behaviour.
- Audit source: §6-§9

### D9 — Quarantine flag respected at `ready` filter but not `all_solved` check — **RESOLVED**
- Location: `src/qft_pcn/composition/orchestrator.py` (sets
  `c.quarantined = True`) + helper
  `src/qft_pcn/composition/goal_graph.py::all_solved`
- Spec: §6.6
- Issue: Quarantined siblings were filtered from `ready` but counted in
  the inline `all_solved` check, so parent immediately entered
  `PENDING_REVISION` and revised away from partial progress.
- Resolution: New `goal_graph.all_solved(node)` helper applies the same
  `not c.quarantined` predicate used by `ready`. The orchestrator now
  routes through this helper, and the `_JointResult` joint sums only
  live (non-quarantined) children. An all-quarantined parent returns
  `False` (no covering proof) so PENDING_REVISION still fires when no
  live alternative exists. Tests `test_all_solved_skips_quarantined`,
  `test_all_solved_false_when_live_child_unsolved`, and
  `test_all_solved_false_when_all_children_quarantined` pin the
  three cases.
- Audit source: §6-§9

### D10 — `_bond_entanglement_of` silently degrades on any substrate exception
- Location: `src/qft_pcn/composition/goal_graph.py:286-292`
- Spec: §1.1 anti-shortcut, §12.16 path-fitness signal
- Issue: `try: return float(fn(cut)) except Exception: return 0.0`. A
  substrate fault (broken `entanglement_entropy` contract) becomes an
  invisible zero in the §12.16 signal. §1.1 anti-shortcut rule
  explicitly forbids "silent skip / graceful degrade" on substrate
  errors.
- Fix scope: small — narrow `except` to expected `(IndexError,
  ValueError)` shape and re-raise everything else.
- Audit source: §6-§9

### D11 — §10.9 CONSOLIDATE step is incomplete: subsumes cached lemma instead of re-deriving — RESOLVED
- Location: `src/qft_pcn/composition/wake_sleep.py::_consolidate`
- Spec: §10.9 (architecture lines 877-882)
- Issue: Spec pseudocode: "FOR each |Ψ_i⟩ in library: IF
  can_be_expressed_using_new_primitives: replace with shorter
  solution." Prior code added the entire cached lemma `sid` to
  `stale_dynamic` and pruned via `library.prune` whenever one cluster
  member matched a newly-promoted primitive's canonical density. No
  `library.replace(...)` call; no replacement synthesized. Silently
  destroyed lemmas whose larger structure had not been replaced.
- Resolution: `_consolidate` now synthesises a replacement
  :class:`CanonicalPrimitive` ``L'`` from the matched sub-piece's RDM
  (via `compute_canonical_form` on a single-member cluster), tags its
  provenance with `derived_from:{sid}` and `uses_primitive:{prim_source}`
  markers, and calls `library.replace(sid, L')` — which atomically
  persists ``L'`` BEFORE pruning ``sid``. If `library.replace` raises,
  ``sid`` is left in place (no orphan empty-library transition). New
  test `test_consolidate_re_derives_before_pruning` asserts every
  pruned sid has a recorded replacement carrying both provenance
  markers.
- Audit source: §10

### D12 — §12.3 topological_degeneracy: Betti number of binding graph is not the proof-homotopy count — RESOLVED
- Location: `src/qft_pcn/composition/topological_degeneracy.py`
  (`count_binding_graph_strategies`, renamed)
- Spec: §12.3
- Issue: Module claimed `K^g = 2^{b1*g}` of the binding diagram counts
  "essentially different proof strategies", but `b1` of the AST
  binding graph is a structural invariant of the encoded program, not
  the ground-subspace degeneracy of the constraint Hamiltonian.
- Resolution: honest-rename approach (option b). `count_proof_strategies`
  → `count_binding_graph_strategies`. Module + function docstrings now
  state explicitly that this is a STRUCTURAL invariant of the binding
  diagram (necessary-but-not-sufficient for the spec count), NOT the
  ground-subspace degeneracy. EXTENSIONS.md entry tracks the genuine
  proof-strategy enumeration via `eigvalsh` ground-subspace count as
  the deferred spec capability. Tests updated to use the new name and
  pin the structural invariant.
- Audit source: §12

### D13 — §12.4 bootstrap delivers no type-derived complexity bound — RESOLVED
- Location: `src/qft_pcn/composition/bootstrap.py`
  (`verify_typing_via_anomaly_sdp`)
- Spec: §12.4
- Issue: Spec capability: derive performance bounds, side-effect
  classes, complexity bounds from type signature alone (e.g.
  `f : List Int -> Int` ⇒ termination ≥ Ω(1), depth ≥ log(length)).
  Implementation builds an SDP whose diagonal is fixed by the §12.1
  anomaly trace; feasibility is exactly equivalent to "all anomaly
  diagonals fit under the truncation gap" — a re-skin of §12.1's
  well-typed/ill-typed split. `dimension_bound = max(o_i)` is not an
  OPE-dimension bound. No bound on termination, depth, complexity, or
  side-effects is produced.
- Resolution: option (b) — honest rename
  `verify_typing_via_bootstrap` → `verify_typing_via_anomaly_sdp`;
  module docstring scope-limited to "typing-feasibility SDP wrapper
  around §12.1 anomaly diagonals; sound necessary-but-not-sufficient
  bound on §12.4 acceptance." Full crossing-equation OPE-truncated
  bootstrap tracked in EXTENSIONS.md ("§12.4 conformal bootstrap full
  bound capabilities"). Tests updated to the renamed symbol.
- Audit source: §12

### D14 — §12.7 replica_complexity Z is a leaf-marginal observable, not a proof-space partition function — RESOLVED 2026-05-23
- Location:
  `src/qft_pcn/composition/replica_complexity.py:204-301`
  (`_default_hamiltonian`, `instance_partition_function`)
- Spec: §12.7
- Issue: Spec Z is "count of valid proofs weighted by their
  complexity." Implementation computes
  `Z = geometric_mean_k tr(rho_k @ expm(-beta * h_k))` where `h_k` is
  a generic bosonic number operator (bare_mass=1.0, kinetic=0.5)
  unrelated to proof search, constraint algebra, or theorem class.
  Replica continuation operates on a quantity disconnected from
  `<log Z>` semantics.
- Fix scope: large — (a) replace `_default_hamiltonian` with the real
  proof-class constraint Hamiltonian (or `MeraEvalHamiltonian` /
  `MeraTypingHamiltonian` of the instance), or (b) scope-limit the
  docstring and rename.
- Audit source: §12
- Resolution: honest rename path (b). `compute_typical_complexity` ->
  `compute_typical_field_marginal_complexity`; module + function
  docstrings now state plainly that Z is the leaf-marginal field
  free energy, NOT the §12.7 proof-space partition function. Legacy
  name retained as `DeprecationWarning`-emitting alias for existing
  call sites. Genuine proof-space Z
  (`sum_{proofs} exp(-beta * worldline_pi_action)`) tracked in
  `EXTENSIONS.md` "§12.7 proof-space partition function".
- Follow-up (audit pass 2 §12 — U1+U2, 2026-05-23):
  - U1: `predict_proof_difficulty` docstring (lines 567/587) still
    cited the deprecated alias `compute_typical_complexity`; body
    already calls the honest name. Docstring fixed.
  - U2: `tests/test_replica_complexity.py` pinned 8 call sites to
    the deprecated alias, emitting `DeprecationWarning` on every
    pytest run and leaving the honest name untested at the
    composition layer. All 8 call sites + the import now use
    `compute_typical_field_marginal_complexity`. Legacy alias kept
    in place for downstream backward compat.

### D15 — §12.10 holographic_compilation ships no optimization pass; verify is tautological — **RESOLVED**
- Location:
  `src/qft_pcn/composition/holographic_compilation.py`
- Spec: §12.10
- Issue: Spec capability: "compiler optimizations provably semantics-
  preserving by construction" with concrete passes (lowering,
  optimization, inlining, constant folding). Implementation: each
  `CompilationLayer` carries handles to the same encoded state, so
  the verify routine was bitwise tautological (same Wilson signature
  across all layers). No optimization pass exists; no semantics-
  preserving rewrite is exercised.
- Resolution: honest rename path (b). `compile_to_mera_layers` →
  `record_mera_layer_sequence`; `verify_layer_equivalence` →
  `verify_layer_state_identical`. Module docstring honestly scoped:
  record-only, shared-state identity, Wilson-signature delegation.
  Full §12.10 RG-flow optimization passes (compress, fuse, eliminate)
  tracked in `EXTENSIONS.md` ("§12.10 holographic compilation
  optimization passes"). Test file and ENHANCEMENTS E18 updated to the
  new names. The cross-extension API wiring (§12.2 + §12.8) remains
  sound; only the optimization-pass library is the deferred
  dependency.
- Audit source: §12

### D16 — §12.12 quantum_extremal_surface is post-hoc ranking, not a-priori prediction — **RESOLVED**
- Location:
  `src/qft_pcn/composition/quantum_extremal_surface.py`
  (`find_minimum_complexity_proof`)
- Spec: §12.12
- Resolution: Honest rename + a-priori bound shipped.
  `find_minimum_complexity_proof` renamed to
  `rank_completed_proofs_by_qes_area` (honest: post-hoc QES ranking of
  already-encoded candidates). New `compute_qes_lower_bound(theorem)`
  takes ONLY the theorem state and returns the QES on its midpoint
  encoding region — a valid a-priori lower bound on proof complexity,
  usable as a search-budget gate before any proof exists. Module
  docstring updated honestly: post-hoc ranking is fully operational;
  the a-priori bound is valid but loose (the tight minimum over
  proof-MERA topologies is tracked in EXTENSIONS.md "§12.12 full
  a-priori complexity bound"). Test
  `test_qes_lower_bound_under_optimal_proof_complexity` enforces that
  every constructed proof's QES area is >= the a-priori bound.
- Audit source: §12

## DOC GAPS

### D17 — `mera_typing_hamiltonian._stub` is dead code — **RESOLVED**
- Location: `src/qft_pcn/logic/mera_typing_hamiltonian.py:212-213`
- Spec: §5 typing rules; `memory/no-placeholders.md`
- Resolution: Dead `_stub` function deleted. Verified unused via
  `grep -rn "_stub" src/qft_pcn/` (only hit was the definition itself).
  Module parses clean post-removal.
- Audit source: §1-§5

### D18 — `lemma_library._validate_decoded` is a tautology with no EXTENSIONS entry — **RESOLVED**
- Location: `src/qft_pcn/composition/lemma_library.py:649-671`
- Spec: §4.5 step-2 validation pass
- Resolution: EXTENSIONS.md entry "Tensor-network typechecker for
  lemma decode validation" tracks the deferred operator-algebraic
  typechecker that would replace the current tautological
  `_validate_decoded`. The tautology stays in place (the anti-shortcut
  forbids the §1.6-violating inline AST typecheck); the residual gate
  carries the validation load until the deferred work lands.

### D19 — `use_log` dual-sense field; adapter docstring describes dead heuristic — RESOLVED
- Location:
  `src/qft_pcn/composition/lemma_library_adapter.py:89-99` (docstring);
  `src/qft_pcn/composition/wake_sleep.py:150-156` (TODO marker — now
  tracked in EXTENSIONS.md)
- Spec: §3.3 core-immunity
- Resolution: docstring rewritten to match actual logic — the explicit
  `_tiers` map (or `tier_of_callable` override) is authoritative;
  `use_log` is explicitly called out as NOT a tier signal because it
  carries `"replace:{old_id}"` markers that would misclassify subsumed
  primitives as "core". The deferred proper `tier` field on `Lemma` is
  now recorded in `EXTENSIONS.md`.
- Audit source: §10

### D25 — §5.5 `two_site_expectation` odd-leaf branch drops layer-0 intra disentanglers — RESOLVED
- Location: `src/qft_pcn/qft/mera.py` `two_site_expectation` odd-leaf
  branch (post-fix lines around 1088-1124); guard helper renamed to
  `_layer0_any_nontrivial` (was `_layer0_inter_is_nontrivial`)
- Spec: §5.5 (Vidal 2008 §III.5 four-site causal cone)
- Issue: Sister bug to D5. The D5 fix only guarded INTER-pair
  non-identity at layer 0; the odd-leaf branch of
  `two_site_expectation` ALSO silently dropped
  `disentanglers[0][j_inter]` and `disentanglers[0][j_inter+1]` (the
  INTRA-pair disentanglers wrapping the 4-site fold). Reachable after
  any even-leaf `apply_two_site_gate` (e.g. every `trotter_step` does
  this), and the bias was unobserved because cross-check via
  `_materialize` was unreachable.
- Resolution: extended the D5 guard to a uniform
  `_layer0_any_nontrivial` check (intra OR inter). When ANY layer-0
  disentangler is non-identity, both `local_expectation` and
  `two_site_expectation` route to the materialize fold, which honors
  the full layer-0 causal cone and `NotImplementedError`-fails loudly
  on layer-≥1 non-identity (preserving the layer-0-only modification
  invariant). New test
  `test_two_site_expectation_handles_non_identity_intra` constructs the
  intra-only regime (CNOT on (0, 1) leaves inter identity), asserts the
  guard fires, and checks the post-fold expectation matches the true
  Bell-marginal value (0.5) against the pre-fix silent-drop result (0).
- Audit source: §1-§5 (D25 / ND1)

### D29 — §10.11 spec mandates `composition/demo_hierarchical_proof.py` — RESOLVED
- Location: `src/qft_pcn/composition/demo_hierarchical_proof.py` (now
  present); load-bearing acceptance remains
  `src/qft_pcn/composition/tests/test_hierarchical_proof_demo.py`
- Spec: §10.11 hierarchical proof composition demo (runnable module)
- Resolution: a thin runnable wrapper imports the SAME substrate
  plumbing (`_ast_theorem`, `_HierarchicalDecomposer`, `_runner`) from
  the acceptance test and exposes a `run_demo()` entry point plus a
  `__main__` block that prints a concise narrative report. No
  duplication of the real-substrate logic — the wrapper re-uses the
  test module's symbols verbatim. Runnable via
  `python -m src.qft_pcn.composition.demo_hierarchical_proof`.
- Audit source: §10.11 deviation sweep
