# Audit Pass 1 — §6-§9 — HEAD 2b6fd56

Audit scope is the composition-tier local design spec
`docs/superpowers/specs/2026-05-22-cross-level-passing-design.md` (§4-§13)
plus `QFT_PCN_ARCHITECTURE.md` §10.10 / §11.2. EXTENSIONS.md was
cross-checked first; known scope-limits (LLM-revision integration, non-
contiguous decomposer demonstrator, multi-candidate `n_top_k`, projector
mode Hamiltonian wiring, etc.) are deliberately **not** re-listed below.

## ENHANCEMENTS (productive divergence)

- **Sibling-consensus QEC gate (§12.5 / §10.10).**
  `dispatcher.dispatch_siblings` filters `report.flagged` through a
  cohort-consensus rule: a syndrome-flagged child is only treated as
  genuinely corrupt when (a) the caller supplied an explicit pre-
  corruption snapshot for it, or (b) at least one sibling matches the
  parent signature below `qec_threshold` AND `len(children_states) >= 2`.
  Without this gate, every imaginary-time-evolved child would trip the
  syndrome (legitimate evolution away from a fresh parent encoding
  reads as "corruption"). The spec language ("disagreement among
  siblings") is recovered as a real outlier test. Productive: keeps the
  K-8 single-child path passable while preserving the §12.5 contract
  whenever a real cohort exists.
  Files: `composition/dispatcher.py:236-265`.

- **Joint-residual internal-node accounting (`_JointResult`).**
  Internal SOLVED nodes are given a `_JointResult` whose
  `residual_energy` is the **sum** of children's residuals, plus the
  matching `solved_ast` tuple. `compute_free_energy` then walks only
  leaves to avoid double-counting (`goal_graph.py:217-224`). The
  combination preserves §9.5 monotonicity in the depth>1 case (which
  the original spec text does not explicitly address). Productive: a
  smaller architectural patch than rewriting the accuracy sum.
  Files: `composition/orchestrator.py:74-89,283-293`,
  `composition/goal_graph.py:204-230`.

- **`source_run_id` namespacing of content-addressed lemma IDs.**
  `lemma_library._content_id` mixes `derivation.source_run_id` into the
  hash so two sub-proofs of the same proposition under different
  `goal_id`s register as distinct lemma entries. This is required by
  §10.11 hierarchical decomposition (L1 and L2 each prove the same
  theorem under different sibling contexts). The fallback (no
  source_run_id) is pure content-addressing. Documented in code.
  Files: `composition/lemma_library.py:680-713`.

- **Precision-weighted clamp via leaf blend, not via projector weight.**
  `Promoter.apply_init_clamp` realizes the §6.2 precision-weighted
  message by linearly blending cached vs host leaves
  (`s*cached + (1-s)*host`, renormalized) rather than scaling a
  projector weight. The blend is operator-algebraic (`§1.5`) and
  composes with §5.2a's freeze set without touching the Hamiltonian.
  Files: `composition/promoter.py:117-178`.

- **`parent_leaves: tuple[int, ...]` instead of a single `parent_site`
  int.** Already in EXTENSIONS.md — listed here only as a reminder
  that the wire format change unblocks non-contiguous / species-
  permuted decomposers as a first-class shape.

## DEVIATIONS (must fix)

- **The integrator's strict spectral-gap gate is never cleared in
  production. (§6.3, §9.3)**
  `result_integrator._spectral_gap` reads
  `child_result.run_diagnostic.get("spectral_gap", 0.0)` and the
  default 0.0 forces the gate to refuse. The bridge's actual
  `RunResult.to_dict()` (`bridge/runtime/result.py:49-65`) does NOT
  emit a `spectral_gap` field — `energy`, `energy_per_term`,
  `final_bond_dimensions`, `converged`, `trotter_steps` only. Every
  test that exercises the integrator manufactures
  `run_diagnostic={"spectral_gap": 1.0, ...}` in a stub `ChildResult`
  (see `test_result_integrator.py:288`,
  `test_cross_level_acceptance.py:247`,
  `test_orchestrator.py:140`, `test_dispatcher.py:223`, etc.). In a
  real `solve_goal_graph` run with the production `run_child` →
  `run_problem`, every child will hit
  `gap < GROUND_STATE_GAP` and be refused as "near-degenerate".
  Resolution: surface a real spectral-gap diagnostic from the bridge
  runtime (probe the first excited state of the converged Hamiltonian
  after relaxation, or carry the gap computed by the MERA evolver
  through `RunResult.to_dict()`). Not in EXTENSIONS.md.

- **§8 cache-hit by `goal_id` never fires.** Spec §8: *"Before
  dispatching a node, the dispatcher queries the library by `goal_id`:
  if a lemma for that exact goal already exists ... the node resolves
  to the cached lemma with no child run."* No code path in
  `dispatcher.dispatch_siblings` or `orchestrator.solve_goal_graph`
  inspects `lemma_library` (or `LemmaLibraryAdapter`) keyed by the
  child's `goal_id` before submitting. `lemma_library.LemmaLibrary`
  does not even expose a `goal_id -> Lemma` index; `_content_id` mixes
  `source_run_id` (the goal_id) into the lemma hash but there is no
  reverse lookup. As a result, siblings re-prove an identical
  sub-goal from scratch every time the §10.11 decomposition repeats a
  proposition. Not in EXTENSIONS.md. Resolution: add a
  `find_by_goal_id` index path and a pre-dispatch lookup that
  short-circuits `run_child` into a synthetic SOLVED ChildResult.

- **§8 lazy re-evaluation of provisional / boundary-changed lemmas is
  absent.** Spec §8: *"if a node's boundary (assumptions) changes
  after a lemma was cached under the old assumptions, the cache
  entry is invalidated and the node re-dispatched. Provisional
  integrations (§6.3) are always re-evaluated when a sibling
  completes."* Nothing in `orchestrator._solve` re-evaluates a
  `provisional=True` SOLVED node once a sibling lands; the
  `IntegrationOutcome.provisional` bit is set but consulted nowhere
  downstream. Not in EXTENSIONS.md.

- **Orchestrator does not plumb the LLM reviser; the principled
  path is unreachable from `solve_goal_graph`.** Spec §10 / §6.5:
  *"Asks G's LLM frontend (`bridge/llm.py`) ... A deterministic
  HeuristicReviser fallback exists for offline / no-LLM runs."*
  `orchestrator.solve_goal_graph` calls
  `revise(node, reviser=reviser, cache=cache)` with no
  `llm=` keyword (`orchestrator.py:325`); the LLM branch in
  `revision.revise` (`revision.py:88-100`) is dead code from the
  orchestrator's point of view. There is no `llm:` parameter on
  `solve_goal_graph` either. Not flagged in EXTENSIONS.md as a
  scope-limit. Resolution: add an `llm: LLMReviser | None = None`
  parameter to `solve_goal_graph` and forward to `revise`.

- **§9.5 monotonicity not enforced; only observed.** Spec §9.5 /
  §13.5: *"K asserts this monotone decrease in the acceptance suite
  ... a non-monotone step is a bug in the integrator (a premature or
  wrong clamp)."* `orchestrator.solve_goal_graph` only emits the
  current `F_hierarchy` via the optional `on_step` callback; there
  is no in-line assertion or refusal when a step would increase
  `F_hierarchy`. A premature clamp that inflates joint residuals
  passes silently. Resolution: track the previous F and raise (or
  refuse the integration) when the new value strictly exceeds it
  beyond a tolerance.

- **`_frontier_priority` is a structural fan-out proxy, not the
  precision-weighted schedule §5.3 demands.** The orchestrator's
  docstring acknowledges this honestly. Spec §5.3 *does* permit a
  fallback "only when the precision estimates are unavailable, and
  the code must say so explicitly." A leaf's precision /
  hole-density estimate is computable from the DSL spec (count
  `?`-leaves, total weight in `boundary`) at no significant cost.
  Not in EXTENSIONS.md. Resolution: either record this as an
  EXTENSIONS entry (deferring the principled schedule with a named
  dependency) or compute a precision proxy from
  `node.goal.dsl_spec` and use it for `sorted(..., key=...)`.

- **`revision.HeuristicReviser` emits a single replacement sub-goal
  inheriting the parent's `parent_leaves` verbatim.** Three
  catalogue entries (`swap_induction_variable`,
  `split_conjunction_other_way`, `strengthen_induction_hypothesis`)
  produce decompositions that differ only by a string field
  (`spec["revision_strategy"]`) — the compiler downstream is not
  obliged to read that string, and nothing in this audit's scope
  does. The "alternative decomposition" is therefore content-
  identical to the failed one modulo a metadata tag, so
  `FailedDecompositionCache.is_failed` returns False (different
  `goal_id` because content-hash changes) but the child run will
  almost certainly fail the same way. The cache cannot guard against
  this because the goal_id changed. Resolution: the heuristic
  catalogue must produce decompositions whose `dsl_spec` content
  actually differs in compilable structure (e.g. different sub-goal
  count, different boundary). Not in EXTENSIONS.md.

- **Quarantine flag is set but read only at the immediate `ready`
  filter.** `orchestrator._solve` sets `c.quarantined = True` after
  a child's `_solve` returns False, then filters at the top of the
  loop via `[c for c in node.children if not c.quarantined]`. But
  the `all_solved` check (`orchestrator.py:270-272`) walks
  `node.children`, including quarantined ones, so the parent
  immediately enters `PENDING_REVISION` and revises away from the
  partial progress. The §6.6 "quarantine + continue exploring
  alternatives in parallel" behavior is not realized: there is no
  parallel-alternative path; quarantine just delays the inevitable
  revision-loop. Resolution: either (a) treat quarantined siblings
  as out-of-band failures that do not block parent completion when
  the live siblings collectively cover the proof, or (b) document
  the simplification in EXTENSIONS.md.

- **`_bond_entanglement_of` silently degrades on any substrate
  exception.** `goal_graph.py:286-292`: `try: return float(fn(cut))
  except Exception: return 0.0`. The docstring justifies the
  fallback but a substrate fault (e.g. broken `entanglement_entropy`
  contract on a new MERA variant) becomes an invisible zero in the
  §12.16 path-fitness signal — the §1.1 anti-shortcut rule explicitly
  forbids "silent skip / graceful degrade" on substrate errors.
  Resolution: narrow the `except` to the expected
  `(IndexError, ValueError)` shape and re-raise everything else.

Audit reviewed `dispatcher.py`, `result_integrator.py`, `promoter.py`,
`orchestrator.py`, `goal_graph.py`, `revision.py`, `wake_sleep.py`,
`lemma_library.py`, `lemma_library_adapter.py`,
`logic/mera_evolution_logic.py`. No DEVIATIONS were found in
`promoter.py`, `wake_sleep.py`, `lemma_library_adapter.py`, or
`logic/mera_evolution_logic.py` beyond what EXTENSIONS.md already
records.
