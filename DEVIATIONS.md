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

### D4 — Orchestrator does not plumb LLM reviser; principled path unreachable
- Location: `src/qft_pcn/composition/orchestrator.py:325` (calls
  `revise(node, reviser=reviser, cache=cache)`),
  `src/qft_pcn/composition/revision.py:88-100` (LLM branch)
- Spec: §10, §6.5
- Issue: `solve_goal_graph` never forwards an `llm=` keyword to
  `revise`; the LLM branch in `revision.revise` is dead code from the
  orchestrator's vantage. No `llm:` parameter exists on
  `solve_goal_graph`.
- Fix scope: small — add `llm: LLMReviser | None = None` parameter to
  `solve_goal_graph` and forward to `revise`.
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

### D6 — §9.5 monotonicity observed but not enforced
- Location: `src/qft_pcn/composition/orchestrator.py::solve_goal_graph`
- Spec: §9.5, §13.5
- Issue: Spec: "K asserts this monotone decrease... a non-monotone step
  is a bug in the integrator." Orchestrator emits `F_hierarchy` via
  optional `on_step` callback but has no in-line assertion or refusal
  when a step increases F. Premature-clamp inflations pass silently.
- Fix scope: small — track previous F and raise (or refuse the
  integration) when new value strictly exceeds it beyond tolerance.
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

### D8 — `revision.HeuristicReviser` emits content-identical sub-goals modulo metadata
- Location: `src/qft_pcn/composition/revision.py` (catalogue entries
  `swap_induction_variable`, `split_conjunction_other_way`,
  `strengthen_induction_hypothesis`)
- Spec: §6.5
- Issue: Three catalogue entries produce decompositions differing only
  by a `spec["revision_strategy"]` string. Downstream compiler is not
  obliged to read that field. `FailedDecompositionCache.is_failed`
  returns False because goal_id changed, but the child run will fail
  identically. The cache cannot guard.
- Fix scope: medium — catalogue must produce decompositions whose
  `dsl_spec` content differs in compilable structure (sub-goal count,
  boundary).
- Audit source: §6-§9

### D9 — Quarantine flag respected at `ready` filter but not `all_solved` check
- Location: `src/qft_pcn/composition/orchestrator.py` (sets
  `c.quarantined = True`; `all_solved` walks `node.children` at
  `orchestrator.py:270-272`)
- Spec: §6.6
- Issue: Quarantined siblings are filtered from `ready` but counted in
  `all_solved`, so parent immediately enters `PENDING_REVISION` and
  revises away from partial progress. The §6.6 "quarantine + continue
  exploring alternatives in parallel" behavior is not realized; no
  parallel-alternative path exists.
- Fix scope: medium — (a) treat quarantined siblings as out-of-band
  failures that do not block parent completion when live siblings
  cover the proof, or (b) document the simplification in EXTENSIONS.md.
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

### D11 — §10.9 CONSOLIDATE step is incomplete: subsumes cached lemma instead of re-deriving
- Location: `src/qft_pcn/composition/wake_sleep.py::_consolidate` (lines
  120-165)
- Spec: §10.9 (architecture lines 877-882)
- Issue: Spec pseudocode: "FOR each |Ψ_i⟩ in library: IF
  can_be_expressed_using_new_primitives: replace with shorter
  solution." Current code adds the entire cached lemma `sid` to
  `stale_dynamic` and prunes via `library.prune` whenever one cluster
  member matches a newly-promoted primitive's canonical density. No
  `library.replace(...)` call; no replacement synthesized. Silently
  destroys lemmas whose larger structure has not been replaced.
- Fix scope: medium — synthesize a replacement MPS that uses the new
  primitive, then `prune_redundant_lemmas`.
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

### D15 — §12.10 holographic_compilation ships no optimization pass; verify is tautological
- Location:
  `src/qft_pcn/composition/holographic_compilation.py:102-187`
  (`compile_to_mera_layers`, `verify_layer_equivalence`); shared state
  alias at line 141 `state=state`
- Spec: §12.10
- Issue: Spec capability: "compiler optimizations provably semantics-
  preserving by construction" with concrete passes (lowering,
  optimization, inlining, constant folding). Implementation: each
  `CompilationLayer` carries handles to the same encoded state, so
  `verify_layer_equivalence` is bitwise tautological (same Wilson
  signature across all layers). No optimization pass exists; no
  semantics-preserving rewrite is exercised.
- Fix scope: large — (a) implement at least one concrete MERA-layer
  optimization pass (e.g. disentangler simplification preserving the
  Wilson signature on a non-trivial pre/post pair), or (b) rename to
  "MERA-layer iteration record" and document deferral.
- Audit source: §12

### D16 — §12.12 quantum_extremal_surface is post-hoc ranking, not a-priori prediction
- Location:
  `src/qft_pcn/composition/quantum_extremal_surface.py:342-411`
  (`find_minimum_complexity_proof`)
- Spec: §12.12
- Issue: Spec capability: a-priori prediction of proof complexity from
  theorem geometry alone, before any search — a true lower bound that
  lets the architecture reject geometrically-impossible theorems
  without search. Implementation requires the caller to supply already-
  encoded candidate proof MERAs and only ranks them by midline
  entanglement entropy.
- Fix scope: large — (a) ship a routine that computes QES lower bound
  from theorem state alone (no candidates) and surfaces it as a
  search-budget gate, or (b) rename to
  `rank_candidate_proofs_by_qes` and document deferral.
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

### D19 — `use_log` dual-sense field; adapter docstring describes dead heuristic
- Location:
  `src/qft_pcn/composition/lemma_library_adapter.py:90-94` (docstring
  fallback); `src/qft_pcn/composition/wake_sleep.py:150-156` (TODO
  marker)
- Spec: §3.3 core-immunity
- Issue: `use_log` carries provenance AND subsumed-source-ids /
  `"replace:{old_id}"` markers. Adapter `_tiers` is populated
  explicitly (all "dynamic"), so the use_log-based "core" heuristic in
  the docstring is dead code. A future reader following the docstring
  would misclassify subsumed/replaced primitives as "core" — a §3.3
  violation in a single touch. TODO is in wake_sleep but not
  EXTENSIONS.md.
- Fix scope: small — split into two fields or delete the docstring
  fallback paragraph; add EXTENSIONS entry if deferring.
- Audit source: §10
