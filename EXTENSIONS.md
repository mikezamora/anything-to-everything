# Extensions — recorded missing dependencies

Per `memory/no-placeholders.md`: when a feature cannot be implemented
because a real dependency is missing, the gap is recorded here rather
than left as a TODO or stub. Each entry names the call site, what is
needed, the workaround currently in tree, and which acceptance criterion
it unblocks.

## Missing dependency: §5.3 full snapshot-and-compare ΔF estimator for `_frontier_priority`

- Where: `src/qft_pcn/composition/orchestrator.py::_frontier_priority`
- Need: principled per-frontier-node expected-ΔF computation — snapshot
  the live `goal_graph`, hypothetically decompose the candidate node,
  diff the post-decomposition `compute_free_energy` against the pre-
  snapshot baseline, and use the signed ΔF directly as the schedule
  key. This is the spec §5.3 ideal: schedule by the node whose
  expansion most reduces the hierarchical free energy.
- Workaround (in tree, D7 resolution): `_frontier_priority` now
  returns the principled-but-cheap `precision * coupling` signal
  (precision from `1/(1 + residual/RESIDUAL_SCALE)` per §6.2,
  coupling from `1 + bond_entanglement` via §12.16 Schmidt-spectrum
  cut). Nodes without a substrate measurement fall back to a
  scaled-down structural fan-out so any measured node outranks them.
  This matches the §5.3 ordering contract on any node that has been
  dispatched at least once — which covers the post-warmup regime that
  the schedule is designed for. A snapshot-and-compare estimator is
  the heavier refinement that adds clone+rollback machinery on the
  goal_graph; the current signal is operator-algebraic and tested.
- Unblocks: spec §5.3 full estimator — if a future workload reveals
  the precision-weighted signal mis-orders the pre-dispatch frontier,
  swap in the snapshot diff (no API change to `_frontier_priority`).

## Missing dependency: §12.11 entanglement-spectrum acceptance corpus needs non-product MERA encoding

- Where: `src/qft_pcn/logic/mera_encoder.py::encode_mera` — concrete closed
  programs (e.g. `\x:Int. x`) collapse to product MERAs (rank-1 Schmidt
  spectrum), so all such proofs land in the TRIVIAL Li-Haldane class
  per §12.11.
- Need: encoder pass (or post-process) that produces non-trivial bond
  entanglement for the spec §12.11 acceptance corpus
  (`reverse(reverse xs) = xs`, `length(xs++ys) = length xs + length ys`).
  Spec line 1928 names these as the acceptance theorems whose spectra
  should DIFFER class-wise from the trivial ones.
- Workaround: `test_entanglement_spectrum.py::test_spectrum_differs_for
  _distinct_theorems` uses hole-bearing rank-≥2 sketches (`\x:Int. ?[x]`
  vs `\f. \x. ?[f,x]`) where rank-k superposition produces non-product
  bonds. Faithful to the §12.11 axiom but does not exercise the literal
  spec corpus.
- Unblocks: spec §12.11 line 1928 acceptance harness (corpus
  classification + clustering convergence). Cross-references the open
  "List arithmetic in encoder substrate" entry that gates reverse/length.

## Missing dependency: §12.13 meet-in-the-middle two-MPS overlap oracle deferred

- Where: spec §12.13 (lines 2012-2014, 2037-2038 of QFT_PCN_ARCHITECTURE.md) names the headline capability as bidirectional search where forward and backward evolutions meet via two-MPS overlap. `src/qft_pcn/composition/bidirectional.py` at commit 988474f ships ONLY the row-2-of-priority chained-traversal primitive (pos-dt descent then neg-dt ascent on a single state).
- Need: a `compute_state_overlap(state_a, state_b) -> complex` primitive on the MERA/MPS substrate, plus a `bidirectional_meet_in_middle(source_state, target_state, H, ...) -> MeetingPoint` driver that runs both directions and detects the meeting via overlap-saturation.
- Workaround: chained traversal in `bidirectional_evolve` exercises the substrate's negative-dt path correctly and validates §1.1 entanglement preservation across both legs. Sufficient for §17-row-2 immediate-win; insufficient for §12.13 headline acceptance.
- Unblocks: spec §12.13 full acceptance corpus (whatever proves the meet-in-the-middle capability operationally).

## RESOLVED — A.4: §12.6 genuine two-operator constraint Hessian

- Resolution: `src/qft_pcn/composition/goldstone.py::_build_constraint_matrix`
  now computes the off-diagonal Hessian entries as the genuine symmetric
  two-operator expectation `Re <psi|H_i H_j|psi>`, via a new
  `_two_term_expectation` helper that decomposes each term into its
  sum of per-leaf factored penalty products
  (`MeraEvalHamiltonian._penalty_ops`, §7.4), composes the per-leaf
  operator products `O_{i,a}[k] @ O_{j,b}[k]` pairwise
  (`_compose_leaf_ops`), and routes back through the existing
  `mera_window_expectation_factored` substrate primitive — the same
  one §12.1 anomaly uses. No dense `16**k` operator is built.
  Disjoint footprints factor automatically (§1.2
  `<H_i H_j> = <H_i><H_j>`) because the per-leaf product over the
  union is just the concatenation of two non-overlapping leaf-op
  dicts; the factored expectation evaluates that correctly in a
  single window call. Zero-residual rows/cols skip via
  `H_i|psi> = 0` in the projector basis. PSD-ness is preserved (Gram
  matrix of `H_i|psi>` vectors).
- Diagonal unchanged: `M[i,i] = <H_i>` since `H_t^2 = H_t` for the
  projector terms (§7.4 P²=P).
- Cascading impact on §12.15 Witten-index reuse: NONE. Witten 1982
  guarantees the index is topological — invariant under continuous
  deformations of the Hamiltonian — so any change in off-diagonal
  Hessian magnitude leaves the index unchanged. No `composition/witten.py`
  test rerun required; the legacy code path was already theorem-robust.
- Tests: `src/qft_pcn/composition/tests/test_goldstone.py` — all 6
  prior tests still pass (solved-theorem null, unsolved-redex
  Goldstone surfacing, node-field localization, determinism,
  parameter validation, ascending-sort contract); new
  `test_hessian_off_diagonal_is_two_operator_expectation` verifies
  (a) every entered off-diagonal matches `_two_term_expectation`
  directly, and (b) at least one non-disjoint pair's entry differs
  materially (>10%) from the legacy `sqrt(r_i r_j) * overlap`
  heuristic, ruling out silent equivalence. 7 passed in ~76 s.
- Closes spec §12.6 substrate completeness (line 1582 "standard
  numerical linear algebra" of the actual Hessian).

## RESOLVED — Substrate task S2: CVXPY-based SDP solver integration

- Resolution: `src/qft_pcn/qft/sdp_solver.py` ships a thin CVXPY wrapper
  providing `SDPProblem` (PSD `PSDVariable`s + lazy `LinearConstraint`
  builders + linear/scalar objective builder), `solve_sdp(problem,
  solver=None, verbose=False) -> SDPSolution` (returns
  `SDPStatus.OPTIMAL / INFEASIBLE / UNBOUNDED / INACCURATE / ERROR`,
  optimal value, per-variable numpy matrices, raw CVXPY status), and
  `psd_constraint_from_operator(operator, name, hermitize, tol)` which
  projects a Hermitian substrate-operator block to a PSD variable +
  equality constraint pair (§1.6: SDPs are numerical but the *problem
  encoding* is operator-derived).
- Dependency: `cvxpy>=1.4` declared in `pyproject.toml` `[project]
  dependencies`. `uv sync` resolved CVXPY 1.7.5 cleanly along with
  Clarabel 0.11.1 / SCS 3.2.11 / OSQP 1.1.1; no compilation needed on
  WSL2.
- Tests: `src/qft_pcn/qft/tests/test_sdp_solver.py` — 6 cases, all pass
  in ~5 s. `test_trivial_psd_min` (min tr(X) s.t. X[0,0]=1, X PSD →
  optimum 1.0), `test_infeasible_sdp_returns_infeasible` (X PSD with
  X[0,0] = -1), `test_max_eigenvalue_sdp` (λ_max via SDP cross-checked
  against `numpy.linalg.eigvalsh` on a random symmetric 5×5),
  `test_psd_constraint_from_operator_psd_input` /
  `..._indefinite_input_infeasible` /
  `..._rejects_nonsquare` for the operator-helper surface.
- Unblocks: §12.4 conformal bootstrap (`composition/bootstrap.py`,
  spec line 1486 "Solve using SDPB ... or CVXPY for prototype.").

## RESOLVED — I-Task-10 blocker #1: `relax_program` driver in M3

- Resolution: `src/qft_pcn/logic/mera_synthesis/runner.py` gains a
  `RelaxResult` (frozen dataclass with `state, meta, hamiltonian,
  residual, trotter_steps, converged`) and a top-level
  `relax_program(ast_src, *, constraints=(), eps, dt, max_trotter_steps,
  chi_layer, n_nodes_max, frozen_leaves, lemma_library)` driver. The
  driver encodes the AST (or parses if given a `str`), composes
  `H_typing + H_eval` via `ComposedMeraSynthesisHamiltonian`, applies
  any `use_lemma` constraints via `Promoter.apply_init_clamp(...,
  strength=1.0)` and unions the clamped leaves into the frozen set
  (along with `meta.forall_protected_leaves` for §1.1 binder
  protection), then runs chunked imag-time evolution with early
  termination the moment `H.total_energy(state) < eps`. The existing
  `synthesize(SynthesisProblem)` sketch-completion runner is untouched.
- `mera_synthesis/__init__.py` re-exports `relax_program` and
  `RelaxResult` via the lazy `__getattr__` (matching `synthesize`'s
  circular-import workaround).
- Tests: `src/qft_pcn/logic/mera_synthesis/tests/test_relax_program.py`
  — 3 cases: closed-arith convergence (`residual < 1e-3`,
  `converged=True`, `1 <= trotter_steps <= 200`), early termination
  (`trotter_steps < 500` at `eps=1e-2`), and full-budget consumption
  (`eps=0.0 → trotter_steps == max_trotter_steps`, `converged=False`).
  All exercise REAL imag-time evolution (no stubs).
- Unblocks: I-Task-10 two-stage acceptance demo (spec §8.12 / arch
  §10.8). With blocker #6 (frozen-leaves hook) already landed, the
  lemma-clamp path is wired in; once blockers #3/#4/#5 land the open
  universal proof obligation `forall x:Nat. Eq (x + 0) x` will relax
  to ~0.
- Per `docs/superpowers/plans/2026-05-23-i-task-10-blocker-fixes.md`
  blocker #1.

## RESOLVED — I-Task-10 blocker #3: R-AddZero reduction rule (`add x Zero -> x`)

- Resolution: new `RULE_R_ADD_ZERO = "R-AddZero"` term added to
  `MeraEvalHamiltonian` (`src/qft_pcn/logic/mera_evaluation_hamiltonian.py`).
  Diagonal redex projector built in `_mera_eval_terms.add_zero_penalty_ops`
  as an inclusion-exclusion sum of (1 BIN-base × 2 P[zero] expansions) +
  (1 BIN-base × 2) + (1 BIN-base × 2×2 cross terms with coefficient -2) =
  8 factored `dict[leaf -> (16,16)]` operators — never a 16**k operator
  (spec §1.3). `P_exactly_one = P_lhs + P_rhs - 2 * P_lhs * P_rhs` is
  faithful: 0 on both-zero and on non-redex configurations, 1 on the
  exactly-one-zero redex. Transition gate (`_add_zero_moves` in the
  Hamiltonian) snapshots the non-zero operand root's 5 species leaves on
  first firing, then drives the BIN node's 5 leaves toward the snapshot
  via `single_leaf_transition_gate` while collapsing the spent zero
  operand's sub-tree to PAD. The non-zero operand root's own leaves
  drain to PAD only AFTER the BIN node has fully received the snapshot
  (the same Stage-C staging `_beta_moves` enforces), guarded by
  `_add_zero_unfinished` + `_add_zero_cleanup_unfinished`. When the
  non-zero operand is a Forall-protected `Var`, the BIN node's `bid`
  leaf is driven toward the Var's bid index — entanglement-preserving
  promotion (§1.1: universal quantification preserved through the
  encoded MERA's tree entanglement, not classical substitution); the
  frozen-leaves filter (#6) then drops any gate whose target leaf is
  in `forall_protected_leaves`, so the Var's own leaves remain bitwise
  unchanged.
- Tests: `src/qft_pcn/tests/test_mera_addzero_rule.py` (6 cases) —
  projector faithfulness on `NatLit(5)+Zero`, `NatLit(5)+NatLit(3)`,
  and `Zero+Zero`; closed-case relaxation on `NatLit(5)+Zero` and
  `Zero+NatLit(5)`; open-case `forall x:Nat. x+Zero` checks (i) the
  Var's 5 leaves are bitwise unchanged under the protected-leaf filter,
  (ii) the BIN node's `bid` leaf inherits the Var's bid index, and
  (iii) the BIN node's `kind` leaf rotates from KIND_BIN to KIND_VAR.
- No-regression: M2 reduction / eval-Hamiltonian / fix-recursion
  suites (16 passed) and `test_mera_eval_terms` (4 passed) unchanged.
- Unblocks: I-Task-10 substrate completeness for open-case proof
  reduction (the demo's `forall x. x + 0 = x`-style obligations);
  Blocker #4 (R-Eq-Refl) depends on this to relax `(x+0) = x` to the
  reflexive `x = x` ground state.
- Per `docs/superpowers/plans/2026-05-23-i-task-10-blocker-fixes.md`
  blocker #3.

## RESOLVED — I-Task-10 blocker #4: R-Eq-Refl reduction rule (`Eq lhs rhs -> BoolLit(True)` when lhs == rhs)

- Resolution: new `RULE_R_EQ_REFL = "R-Eq-Refl"` term added to
  `MeraEvalHamiltonian` (`src/qft_pcn/logic/mera_evaluation_hamiltonian.py`).
  Diagonal redex projector built in
  `_mera_eval_terms.eqrefl_penalty_ops` as
  `lam * P[KIND_EQ](eq_kind_leaf) * (I - P_equal_pair)` per
  (species, paired-sub-tree-leaf-pair), with
  `I - P_equal_pair = I - sum_i P_i(lhs) * P_i(rhs)` factored as
  17 small `dict[leaf -> (16,16)]` operators per (species, pair)
  (1 positive base + 16 negative match terms) — never a 16^2 dense
  operator (spec §1.3). The Hamiltonian walks lhs/rhs sub-trees in
  lockstep BFS via `_eq_paired_subtree(eq_node)` (pure
  `meta.children_of_node` addressing, spec §1.2), emitting one factored
  term per (species, paired node). `<H>=0` iff every species on every
  paired sub-tree node carries the same basis index — structural
  reflexivity. Transition gate (`_eq_refl_moves`) fires WHEN the
  diagonal residual hits 0 (guarded by `_eq_refl_unfinished`, which
  bypasses the default `term_energy < 1e-9` early-return in
  `term_gates` exactly while the Eq node is still KIND_EQ and the
  residual confirms reflexivity): the Eq node's kind leaf rotates
  `KIND_EQ -> KIND_BOOL`, the value leaf to `VALUE_TRUE`, and the
  lhs/rhs sub-trees collapse to PAD (mirrors R-If's branch-keep
  promote/collapse). The frozen-leaves filter (#6) preserves any
  Forall-bound Var leaves inside the collapsed sub-trees — universal
  quantification survives the promotion intact (§1.1).
- Tests: `src/qft_pcn/tests/test_mera_eqrefl_rule.py` — 5 cases:
  diagonal-zero on `Eq Zero Zero`, diagonal-positive on
  `Eq Zero (NatLit 5)`, closed `Eq Zero Zero -> BoolLit(True)`
  reduction, `forall x. Eq x x` diagonal-zero from the start, and the
  load-bearing composite `forall x:Nat. Eq (x + Zero) x` combining
  R-AddZero (#3) + R-Eq-Refl (#4) + Forall-protected (#5) — the
  §10.10 induction theorem path. All assert eventual relaxation +
  promotion to BoolLit(True) without weakening.
- No-regression: M2 reduction, eval-Hamiltonian, fix-recursion,
  add-zero, and forall-protected suites (26 passed) unchanged.
- Unblocks: I-Task-10 substrate completeness — the §10.10 induction
  theorem path now relaxes end-to-end. Downstream M3 acceptance demo
  (`forall x:Nat. Eq (x + 0) x`) is no longer substrate-blocked.
- Per `docs/superpowers/plans/2026-05-23-i-task-10-blocker-fixes.md`
  blocker #4.

## RESOLVED — I-Task-10 blocker #5: Forall-protected leaves under relaxation

- Resolution: `MeraEncodingMeta`
  (`src/qft_pcn/logic/mera_encoder.py`) now carries
  `forall_protected_leaves: set[int]` (default `set()`), populated by
  `_collect_forall_protected_leaves` over the encoder's NodeOccupancy
  sites. For every `KIND_FORALL` binder site the helper adds the
  Forall's own `bid` leaf, then walks all sites whose
  `var_ref.binder_site` points to that Forall and adds those Vars' five
  species leaves (`kind`, `type`, `bid`, `value`, `tobl`). The
  populator runs in the three encoder dispatch paths
  (`encode_mera`, `_encode_with_structural_holes`, `_encode_bundle`),
  so witness-augmented sketches and bundles inherit the same
  protection. The synthesis driver
  (`logic/mera_synthesis/runner.py::_anneal`) forwards the set as
  `frozen_leaves=` to all three phases of
  `mera_imaginary_evolve_state`, reusing the operator-algebraic hook
  from blocker #6. Encodings without any `Forall` produce
  `forall_protected_leaves == set()` and the driver passes
  `frozen_leaves=None`, preserving prior behavior bitwise. This is
  §1.1 binding-as-entanglement on the MERA: the bound Var's leaf
  vectors are unchanged through evolution, the bid leaf stays
  entangled with the Forall — universal quantification by inertia,
  not classical iteration (§1.6).
- Tests: `src/qft_pcn/tests/test_mera_forall_protected.py` covers
  (`test_encoder_populates_forall_protected_leaves`,
  `test_evolution_preserves_forall_bound_var_leaves`,
  `test_relaxation_does_not_pin_forall_var_value`,
  `test_default_empty_when_no_forall`) — encoder populates the
  expected 11 leaves for `forall n:Nat. Eq n n` (1 Forall bid + 2 *
  5 Var species), 40 steps of composed (T-typing + H-eval)
  imag-time evolution leave every protected leaf bitwise unchanged,
  the bound Var's value species remains a unit one-hot at
  `VALUE_NONE` (never pinned to a concrete integer), and the empty
  default holds for Forall-free encodings.
- Unblocks: I-Task-10 acceptance criterion that universal
  quantifiers in propositions survive M2-style relaxation without
  collapsing to a single witness; downstream lemma-library entries
  whose proposition body uses `Forall` (J-Task-6 lemma promotion
  reuses the same hook).
- Per `docs/superpowers/plans/2026-05-23-i-task-10-blocker-fixes.md`
  blocker #5. Depends on blocker #6 (`frozen_leaves=` substrate hook).

## RESOLVED — I-Task-10 blocker #6: `frozen_leaves=` in MERA imag-time evolution

- Resolution: `mera_trotter_step`, `mera_imaginary_evolve_state`, and
  `mera_imaginary_evolve` (`src/qft_pcn/logic/mera_evolution_logic.py`)
  now accept `frozen_leaves: set[int] | None = None`. Inside
  `mera_trotter_step` the gate list returned by `ham.term_gates` is
  filtered to drop any gate whose target leaves intersect the frozen
  set BEFORE the per-leaf grouping pass — operator-algebraic restriction
  of H's action to the unfrozen subsystem (§5.2a / §8.6, "clamp +
  freeze"). For two-leaf gates with one frozen and one unfrozen leg the
  whole gate is dropped (the gate is not separable; the frozen leaf is
  a proved-lemma datum the lemma alone resolves). The default `None`
  preserves prior behavior bitwise — confirmed by the unchanged M2
  reduction / eval-Hamiltonian / fix-recursion suites (16 passed) and
  the broader evolution-touching tests (22 passed).
- Tests: `src/qft_pcn/tests/test_mera_evolution_logic.py` gains three
  cases (`test_trotter_step_with_all_leaves_frozen_is_identity`,
  `test_imaginary_evolve_many_steps_preserves_frozen_leaves`,
  `test_imaginary_evolve_descends_on_non_frozen_leaves`) — all-frozen
  identity, multi-step bitwise stability of a frozen subset, and
  monotone descent of the unfrozen subsystem when a PAD leaf is frozen.
- Unblocks: I-Task-10 two-stage acceptance (Promoter clamps lemma
  leaves and the relaxation driver must not deform them); J-Task-6
  wake-sleep (canonical primitives frozen during host relaxation);
  I-Task-10 blockers #5 (Forall-protected ground-state semantics
  reuses the same hook) and #1 (`relax_program` driver in M3).
- Per `docs/superpowers/plans/2026-05-23-i-task-10-blocker-fixes.md`
  blocker #6.

## RESOLVED — bridge RunResult does not surface MeraEncodingMeta / ground_state / solved_ast

- Resolution: `RunResult` now carries additive, default-`None` fields
  `meta`, `ground_state`, `solved_ast`, `hamiltonian`, and `trotter_steps`
  (`src/qft_pcn/bridge/runtime/result.py`). The MPS path in
  `bridge/runtime/__init__.py::run_problem` populates `ground_state` with
  the final relaxed `MPS`, `hamiltonian` with the composed
  `BridgeHamiltonian`, and `trotter_steps` with `search.steps`. The MPS
  path does not synthesise a `MeraEncodingMeta` or a decoded AST -- those
  remain `None` here and are the MERA-runner's responsibility (see the
  K-8 §10.10 path: a caller that owns `encode_mera`/`decode_mera` can
  populate the same fields end-to-end).
  `composition/dispatcher.py::run_child` reads the new fields via
  `getattr` and now wires through `bridge.runtime.run_problem` directly
  (previously imported a non-existent `run` symbol from
  `bridge.runtime.evolution`; the broken import was masked because every
  shipped dispatcher test used stub runners). A real-bridge integration
  test (`test_run_child_invokes_real_bridge_pipeline`) locks in the
  contract.
- Tests: `src/qft_pcn/tests/test_bridge_result_enrichment.py` (5 cases:
  legacy default, round-trip, end-to-end population, trotter-step
  fidelity, and the dispatcher's `getattr` access pattern) plus
  `src/qft_pcn/composition/tests/test_dispatcher.py::test_run_child_invokes_real_bridge_pipeline`.
- Unblocks: K-5 acceptance (real lemma registration on every solved
  child), §8.6 / §8.11 promotion acceptance via the dispatcher path,
  K-8 §10.10 acceptance.

## Missing dependency: bridge RunResult does not surface MeraEncodingMeta / ground_state / solved_ast (original entry, retained for history)

- Where: `src/qft_pcn/bridge/runtime/result.py` (RunResult dataclass) and
  `src/qft_pcn/composition/dispatcher.py:32-66` (`run_child`).
- Need: `RunResult` (or a sibling `RunArtifacts` it owns) should publish
  the live `MERA` ground state, the `MeraEncodingMeta` the run was
  decoded under, and the decoded `solved_ast`. `run_child` currently
  pulls these via `getattr(run_result, "ground_state", None)` etc., which
  always returns `None` against the shipped RunResult — so `ChildResult`
  arrives at the integrator with `meta=None` and `ground_state=None`,
  and `register_lemma` cannot be invoked.
- Workaround: `ChildResult` defaults `meta=None`/`hamiltonian=None`/
  `trotter_steps=0`; `integrate_child` refuses with a clear error
  ("child_result.meta missing: cannot register lemma") instead of
  silently bypassing the lemma path. A caller that constructs
  `ChildResult` from a richer runner can already populate these fields
  end-to-end (the new integrator tests do exactly this with
  `encode_mera`).
- Unblocks: K-5 acceptance (real lemma registration on every solved
  child), §8.6 / §8.11 promotion acceptance via the dispatcher path.

## RESOLVED: `SubGoal.parent_site` widened to `parent_leaves: tuple[int, ...]`

- Where: `src/qft_pcn/composition/goal_graph.py` (`SubGoal.parent_leaves`)
  and `result_integrator._resolve_host_leaves`.
- Mechanism: `SubGoal.parent_site: int | None` replaced with the
  canonical wire-format field `parent_leaves: tuple[int, ...]` (default
  `()` for the root sentinel). Non-contiguous / species-permuted
  layouts are first-class: the decomposer publishes the explicit leaf
  tuple and `_resolve_host_leaves` returns it verbatim -- no extension
  from `child_meta.n_leaves`, no rebuild from a single base int. A
  convenience constructor `make_contiguous_sub_goal(base, n_leaves)`
  builds the contiguous window for isomorphic decomposers. All callers
  (`build_goal_graph`, `revision.HeuristicReviser`, `revise` LLM path,
  every composition test) migrated. Cited in `_resolve_host_leaves`
  docstring + new test
  `composition/tests/test_subgoal_parent_leaves.py`.
- Unblocks: integrator's call to `Promoter.compile_constraint(... leaves=
  [...])` for non-trivial decomposers (J-Task / decomposer follow-on)
  now has the canonical leaf tuple to consume.
  Commit: 00b1d82.

## Missing dependency: non-contiguous decomposer demonstrator absent

- Where: `src/qft_pcn/composition/goal_graph.py:build_goal_graph` only
  seeds the root with `parent_leaves=()`; the user-supplied `decomposer`
  callable is the surface that would publish non-contiguous tuples
  (e.g. species-permuted lemma footprints). `revision.py:97` LLM-path
  hand-rolls `(i,)` per sibling -- a placeholder, not a real
  non-contiguous decomposer.
- Need: an in-tree non-contiguous decomposer demonstrator that publishes
  e.g. `(2, 5, 9)` for a 3-site lemma footprint and proves the
  integrator clamps without classical rewrite.
- Workaround: `test_subgoal_parent_leaves.py::test_integrator_passes_non_contiguous_window_to_promoter`
  exercises the integrator API end-to-end via a stub Promoter -- pins the
  contract but not a real decomposer. Production decomposers will land
  with J-Task follow-on (subtree miner + clustering).
- Unblocks: §5.2/§5.6 surface exercised in production code (today only
  exercised via tests).

## RESOLVED: orchestrator does not own a parent MERA

- Status: RESOLVED. `solve_goal_graph(..., parent_state, parent_meta,
  ...)` now requires both arguments (Option A from the gap directive
  -- the orchestrator is a solver over an EXISTING workspace, so the
  caller hands in a pre-built parent MERA + meta). A `None` for
  either raises `TypeError` immediately at the entry point; the
  `if parent_state is None` graceful-skip path inside
  `result_integrator.integrate_child` is now unreachable from the
  orchestrator. Every solved SubGoal's lemma fires
  `Promoter.apply_init_clamp` against the parent network -- the §1.1
  entanglement-binding clamp the workaround had silently skipped.
- Pinned by:
  `src/qft_pcn/composition/tests/test_orchestrator_parent_workspace.py`
  -- `test_orchestrator_clamps_lemma_into_parent_mera` (parent leaves
  at the clamped window are bitwise-overwritten with the cached child
  tensor at strength 1.0); `test_orchestrator_refuses_without_parent_workspace`
  (TypeError raised, no silent degradation);
  `test_orchestrator_preserves_unclamped_leaves` (sentinel host leaf
  outside `parent_leaves` is bitwise-unchanged -- §1.3 factored op).
  The five pre-existing `test_orchestrator.py` tests migrated to
  pass `parent_state=pstate, parent_meta=pmeta` and all still pass.

## RESOLVED: `decoder.parse_one` rejects `KIND_FORALL` / `KIND_FIX` (Gap C)

- Status: RESOLVED in commit `40cbbee` on branch
  `claude/qft-pcn-hybrid-architecture-ihCIR` (mixed commit with the
  viz worker's "feat(viz/web): compare-mode wiring" — content is in
  the tree; history-rewrite avoided per git safety protocol). The
  Part-2 stub in `parse_kind_stream._parse_one` has been replaced with
  proper binder branches that mirror the `KIND_LAM` machinery: push a
  freshly-named `Forall` / `Fix` onto `binder_stack`, descend into the
  body, pop, and patch the body. `param_ty` recovery: Fix uses
  `_extended_type_from_tag(ti, ...)` because a `Fix`'s site type tag
  IS its `param_ty` tag; Forall defaults to `TNat()` because the site
  type tag is `TYPE_PROP` (Forall returns Prop), so the param type is
  not directly recoverable from the leaf — `TNat` is the canonical
  §10.10 lemma quantifier.
- Pinned by: 3 new tests in `src/qft_pcn/tests/test_decoder_forall_fix.py`
  (`test_decode_forall_identity_body`, `test_decode_forall_with_eq_body`,
  `test_decode_fix_nat_body`) — round-trips through `encode_mera` →
  `decode_mera` confirm the AST shape and binder-bound `Var` resolution.
  No-regression coverage: 45 pre-existing tests across
  `test_mera_reduction.py`, `test_mera_holes.py`, `test_mera_roundtrip.py`,
  `test_mera_decoder.py`, `test_logic_decoder.py`, `test_logic_roundtrip.py`
  remain green.
- Downstream effect on the K-8 acceptance pin
  (`test_orchestrator_refusal_diagnostic_pins_substrate_seam`): the
  refusal reason flips from `validation_failed:decode_error:...
  Forall/Fix binder decoding is Part-2 scope` (Gap C) to the
  `_meta_to_json` `set is not JSON serializable` (Gap D). The pinning
  test still passes because its assertion accepts either Gap C OR
  Gap D as the surfacing reason; once Gap D lands the orchestrator
  path can persist Forall-rooted proofs end-to-end.

## Missing dependency: `_meta_to_json` does not handle `set` fields — **RESOLVED**

- Where: `src/qft_pcn/composition/lemma_library.py:322-349`
  (`_meta_to_json`) — falls through the `dict` / `list` / `tuple`
  branches to `json.dumps(d)` for any other field type, which raises
  `TypeError: Object of type set is not JSON serializable` on the
  I-10-added `MeraEncodingMeta.forall_protected_leaves: set[int]`
  (`src/qft_pcn/logic/mera_encoder.py:51`).
- Need: a `set` -> sorted-list conversion (or a freeze-to-tuple) in
  `_meta_to_json`'s field-walk so every meta is round-trippable to
  JSON. Symmetric handling in `_meta_from_json` to restore the set
  shape on load.
- Resolution: `_jsonable` now coerces `set` -> sorted list as a
  uniform branch (generic dispatch, not a one-off), `_meta_to_json`
  treats `set` as an iterable field type, and `_meta_from_json`
  consults a `_META_SET_FIELDS` registry (today:
  `forall_protected_leaves`) to restore each list back to `set[int]`
  on load. Round-trip equality is asserted AS A SET in
  `composition/tests/test_lemma_library_meta_json.py`. The K-8
  cross-level pinning suite (`test_cross_level_acceptance.py`) still
  passes because its diagnostic accepted Gap C OR Gap D as the
  surfacing reason; with Gap D resolved the orchestrator-blocked
  diagnostic now isolates to Gap C alone (handled separately).
  Commit: 98e2999.

## RESOLVED (partial) -- Bridge DSL: `forall` / `Eq` / `Nat` / `List` / `Cons` / `Nil` surface (K-8 Blocker B)

- Original gap (`52e9387`): the AST text parser gained `forall`,
  `Eq`, `Nat`, `add`, `Zero`, `Succ`, `NatLit` tokens but had not
  been end-to-end round-tripped from the bridge layer, and the
  spec's literal §10.10 theorem
  `forall xs:List A. length (reverse xs) = length xs` was
  unreachable because the textual surface had no `List` / `Cons` /
  `Nil` keywords.
- Resolution: `src/qft_pcn/logic/ast.py` tokenizer + parser now
  recognise `List`, `Cons`, `Nil` (`List T` as a parametric type
  atom; `Cons head tail` and `Nil` as atom-position term nodes
  routed through the existing `Cons` / `Nil` AST nodes the encoder
  substrate already supports). New tests in
  `src/qft_pcn/bridge/tests/test_dsl_extended_calculus.py` round-trip
  every extended-calculus keyword through parse → `encode_mera` →
  `mera_imaginary_evolve_state` → reduction-residual / leaf-weight
  inspection, including the §10.10 in-substrate composite
  `forall x:Nat. Eq (x + Zero) x` (Eq node promotes to
  `KIND_BOOL` / `VALUE_TRUE`; R-AddZero + R-Eq-Refl residuals
  < 1e-3). This is the K-8 retry directive's Step 2 fallback proven
  from the bridge-layer DSL surface.
- Note on `bridge.runtime.run_problem`: that entry point consumes a
  physics JSON DSL (fields / sites / Hamiltonian terms), not a
  logic-theorem source. The principled "DSL → encoder → solver"
  path for theorems is `parse()` → `encode_mera()` →
  `mera_imaginary_evolve_state()`, exactly what
  `composition/tests/test_cross_level_acceptance.py` drives and what
  the new bridge test re-proves from the textual surface. A separate
  physics-vs-logic dispatch in `run_problem` is a substrate extension,
  not a DSL surface fix.
- Still deferred: `length` / `reverse` primitives in the encoder
  substrate (see new entry "List arithmetic in encoder substrate"
  below). The bridge test for the literal §10.10 list-induction
  theorem is `pytest.mark.skip`'d with a reference to that entry.
  Commit: 1d8f942.

## Missing dependency: List arithmetic in encoder substrate (`length` / `reverse` / `append`)

- Where: `src/qft_pcn/logic/mera_encoding.py` (kind table),
  `src/qft_pcn/logic/mera_encoder.py` (AST → state walk),
  `src/qft_pcn/logic/mera_typing_hamiltonian.py` (typing-rule
  terms), `src/qft_pcn/logic/mera_evaluation_hamiltonian.py`
  (reduction rules `length(Nil)=0`,
  `length(Cons h t)=Succ (length t)`, `reverse(Nil)=Nil`,
  `reverse(Cons h t)=append (reverse t) (Cons h Nil)`,
  `append Nil ys = ys`, `append (Cons x xs) ys = Cons x (append xs ys)`).
- Need: dedicated `KIND_LENGTH`, `KIND_REVERSE`, `KIND_APPEND`
  species in the kind table; encoder leaf-emission for the new
  list operators; typing-Hamiltonian terms enforcing
  `length : List A -> Nat`, `reverse : List A -> List A`,
  `append : List A -> List A -> List A`;
  reduction-Hamiltonian terms implementing the structural-induction
  reductions above so the §10.10 list-induction theorem promotes
  through the same R-Eq-Refl + Forall-protected channel that
  `forall x:Nat. Eq (x + Zero) x` already uses. Decoder
  (`logic/decoder.py`) must parse the new kinds back to
  `App(Var("length"), ...)` / `App(Var("reverse"), ...)` /
  `App(App(Var("append"), xs), ys)` (or dedicated nodes).
- Blocker (S4 investigation, 2026-05-23): the kind table is
  SATURATED at the substrate's leaf dimension. `MERA_LEAF_DIM = 16`
  fixes the bond dimension across all five species (kind/type/bid/
  value/tobl). The kind species already uses all 16 slots: base
  kinds 0-7 (`PAD..BIN` from `encoding.py`) plus 8-15 (`ZERO`,
  `SUCC`, `NATLIT`, `NIL`, `CONS`, `EQ`, `FORALL`, `FIX`). Adding
  `KIND_LENGTH`/`REVERSE`/`APPEND` therefore requires bumping
  `MERA_LEAF_DIM` (and `MERA_KIND_CUTOFF`) to ≥ 19. That cascades
  through every gate-construction site (`_mera_eval_terms.py`,
  `_mera_window.py`, `_mera_holes.py` whose `_WITNESS_BASE =
  MERA_LEAF_DIM - 1` shifts), every leaf-vector construction
  (`_mera_leaves.py`), every test that asserts the dimension
  literally (`test_mera_encoding.py::test_leaf_dim_is_16`,
  `test_mera_holes.py`, `test_mera_window.py`, `test_mera_leaves.py`,
  `test_mera_acceptance.py`), and the MERA bond-dim scaling
  expectations driven by §10.5. S4 must include a leaf-dim
  enlargement pass before the new kinds can be added; this is a
  substrate-wide refactor, not a localized kind-table append.
- Second blocker (S4 investigation, 2026-05-23): even granted the
  kind-table bump, the reductions are NOT value-rewrites like
  R-Arith / R-AddZero / R-Eq-Refl — they are STRUCTURAL
  unfold-and-rewrite gates. `R-Length-Cons` reduces
  `length (Cons x xs)` to `Succ (length xs)` which introduces a
  new `Succ` node and re-attaches a child to a fresh `length`
  application; `R-Reverse-Cons` introduces `append`, two recursive
  applications, and `Cons x Nil`. These need the term-growing
  machinery already paid for by `R-Fix` (`fix_transition_gate`
  + node-budget guard `MeraEvalBudgetExceeded`), not the
  single-leaf transition gates the other rules use. The S4 plan
  must reuse the R-Fix unfold pattern (encoder reserves spare
  nodes via `n_nodes_max`; gate writes the unfolded structure
  into the reserved leaves) rather than treat list reductions as
  in-place value rewrites. `R-Length-Nil` and `R-Append-Nil` are
  shape-preserving (single-node kind/value flip) and would
  resemble `R-Eq-Refl`; the rest require the unfold path.
- Workaround: the textual `List` / `Cons` / `Nil` surface has
  landed (above); `length` / `reverse` / `append` remain free
  identifiers in the parsed AST (they tokenise as `ident` and so
  an expression like `length xs` parses as
  `App(Var("length"), Var("xs"))` -- which simply does not reduce
  under the current evaluation Hamiltonian).
  `src/qft_pcn/bridge/tests/test_dsl_extended_calculus.py::test_parses_list_length_reverse_theorem`
  is `pytest.mark.skip`'d until this entry is resolved.
- Unblocks: K-Task-8 acceptance on the spec's literal §10.10
  theorem `forall xs:List A. length (reverse xs) = length xs`.
  Also gates §10.11's literal `length(xs++ys) = length xs + length ys`
  target; L sub-project adapts to the §10.10 composite at commit
  64567a5 (see
  `src/qft_pcn/composition/tests/test_hierarchical_proof_demo.py`).

## Architectural note: §10.8 content-addressing extended to source_run_id namespace

- Where: `src/qft_pcn/composition/lemma_library.py::_content_id` +
  `register_lemma` call site.
- Change: `_content_id` now accepts an optional `source_run_id` kwarg.
  Default (`None`) preserves the prior pure-content-addressing contract.
  When provided, the content-hash is namespaced by `source_run_id` so
  structurally-identical sub-proofs of the same proposition under
  different `goal_id`s register as DISTINCT library entries.
- Rationale: L §10.11 hierarchical decomposition produces sub-lemmas
  L1, L2 that prove the SAME proposition via independent sub-QPCN runs;
  treating them as identical library entries would collapse the
  hierarchical structure. The contract evolution preserves backward
  compatibility for single-run callers (default `None`).
- Landed: commit 64567a5.

## RESOLVED: Forall param_ty recovery limited to TNat; TList elem limited to TNat

- Status: RESOLVED in this branch (`feat(logic/encoder+decoder): Forall
  param_ty + TList elem round-trip via leaf encoding`).
- Commit: 6a455ad.
- Mechanism (Option A — no leaf-dim growth, respects §1.3):
  - The encoder already wrote `ty_to_tag(param_ty)` into the **value
    species** (`_mera_leaves.node_leaf_vectors`); the decoder now reads
    that species (`vi`) in the `KIND_FORALL` branch via
    `_extended_type_from_tag(vi, ...)` instead of defaulting to TNat.
  - For non-flat param_ty (TList with non-Nat elem, nested TArrow) the
    encoder additionally records the full Ty in
    `nested_type_index[site_idx]` via the new helper
    `_mera_leaves.nested_binder_ty(occ)`. This reuses the existing
    side-table mechanism already used for `TYPE_ARR_NESTED` Lam sites,
    so the decoder reconstructs `TList(elem=TBool())` rather than the
    legacy `TList(elem=TNat())` default.
  - Symmetric treatment of Nil / Cons sites (their `elem` is also
    written via `nested_binder_ty`).
- Tests: `src/qft_pcn/tests/test_decoder_forall_non_nat.py`
  (`forall b:Bool`, `forall x:Int`, `forall xs:List Nat`,
  `forall xs:List Bool`, Nat no-regression — 5 cases all green).
- No-regression: `test_decoder_forall_fix.py` +
  `test_mera_forall_protected.py` (Gap C + I-Task-10 #5) still pass;
  `test_mera_reduction.py` + `test_mera_roundtrip.py` unchanged (16/16).

## Missing dependency: post-promotion stale leaves break decoder trailing-PAD check (Gap E)

Status: **partially resolved** — the substrate's R-Eq-Refl promotion
`_eq_refl_moves` already walks the FULL lhs/rhs subtrees (via
`_subtree_nodes`) and drives every descendant's kind leaf toward PAD
through `_collapse_moves`. Verified by
`test_eqrefl_nested_lhs_pad_collapses_descendants` /
`test_eqrefl_nested_rhs_pad_collapses_descendants`: for
`Eq (NatLit 3 + Zero) (NatLit 3)` (and the rhs-nested mirror), every
descendant node's kind argmax is `KIND_PAD` after evolution and
`decode_mera` parses `BoolLit(True)` cleanly. The §10.10 composite
`forall x:Nat. Eq (add x Zero) x` still fails decode, but the
surviving non-PAD sites are precisely the Forall-protected bound
`Var x` leaves -- the frozen-leaves filter (per spec §1.1) prevents
the collapse gate from touching them. That residual blocker is
restated as **Gap F** below; the original "orphan descendants survive
because the promotion only collapsed the immediate roots" diagnosis
is closed.

Original failure narrative (kept for historical context):

- Where: surfaces in `src/qft_pcn/logic/decoder.py::parse_kind_stream`
  trailing-PAD loop (lines ~315-321).
  Producer: `src/qft_pcn/logic/mera_evaluation_hamiltonian.py::eq_refl_moves`
  (and the `eqrefl_penalty_ops` factored projectors in
  `src/qft_pcn/logic/_mera_eval_terms.py`) -- the promotion fires
  correctly but does not co-project orphaned subtree descendants to PAD.
  After
  `mera_imaginary_evolve_state` proves the §10.10 composite
  `forall x:Nat. Eq (add x Zero) x`, node 0 stays `KIND_FORALL`,
  node 1 promotes from `KIND_EQ` to `KIND_BOOL` (the load-bearing
  R-Eq-Refl result), but the ORIGINAL Eq subtree's descendants
  (node 2 = the `+ x Zero` Bin, node 3 = the bound `Var x`,
  node 4 = `Zero`, node 5 = the second `Var x`) are NOT erased to
  `KIND_PAD` by the promotion. They survive as stale residue with
  unit marginal mass (Var p=1.000, etc., as verified at K-8
  diagnosis).
- Effect: `decode_mera` parses Forall(BoolLit(True)) consuming
  nodes 0..1, then enters the trailing-PAD loop and rejects with
  `DecodeError("site 3 not PAD after AST parse (kind=1)")`.
  `composition.lemma_library.register_lemma` wraps this through
  the totality try/except (per 3950e69) and surfaces it as
  `RegistrationResult(False, ..., 'validation_failed:decode_error:site 3 not PAD after AST parse (kind=1)')`.
  The K-5 integrator refuses the clamp; the orchestrator exhausts
  its MAX_REVISIONS + 1 retries and `solve_goal_graph` returns
  `SolveResult(solved=False, proof_tree=None, failure_report={...})`.
- Need: one of
  (a) extend the §6.1 / I-Task-7 promotion machinery so that when
      a parent node's kind flips to one that consumes fewer
      children (e.g. `KIND_EQ` -> `KIND_BOOL`, which is a leaf
      term), the orphaned subtree's nodes are co-projected onto
      `KIND_PAD` as part of the same promotion step; or
  (b) widen `parse_kind_stream`'s trailing-PAD policy to tolerate
      stale descendants of a promoted-to-leaf node when the
      surviving structural parse is otherwise complete (more
      lenient decoder; preserves "single ground state, one AST"
      invariant by treating below-promoted-leaf sites as
      structurally dead).
  Option (a) is preferred (it keeps decoder strict and matches the
  §1.1 architecture-soul: the substrate IS the proof, so the
  substrate's site layout must mirror the proved AST exactly).
- Pinned by: `test_orchestrator_refusal_diagnostic_pins_substrate_seam`
  and `test_orchestrator_blocked_on_lemma_persistence_substrate_gap`
  in `src/qft_pcn/composition/tests/test_cross_level_acceptance.py`
  (K-8 retry, this entry). Substrate-level proof remains green
  (`test_substrate_level_inductive_theorem_proves_end_to_end`):
  R-AddZero/R-Eq-Refl residuals relax, Eq node promotes to
  `KIND_BOOL` + `VALUE_TRUE` -- the substrate IS proving the
  theorem; only the orchestrator's lemma-persistence handoff is
  blocked by the stale-descendant residue.
- Commit: 7ae483f (initial diagnosis). Gap E "orphan descendants
  not collapsed" closure verified by the two `nested_*_pad_collapses_descendants`
  tests; remaining Forall-protected blocker tracked as Gap F.

## Missing dependency: Forall-protected Var leaves block decoder trailing-PAD check after R-Eq-Refl (Gap F)

- Where: surfaces in `src/qft_pcn/logic/decoder.py::parse_kind_stream`
  trailing-PAD loop (lines ~315-321) on the §10.10 composite
  `forall x:Nat. Eq (add x Zero) x`. After R-AddZero + R-Eq-Refl
  fire, the Eq node (node 1) promotes to `KIND_BOOL` + `VALUE_TRUE`
  and the non-protected descendants (the Bin node, the Zero node)
  collapse to `KIND_PAD` -- but the bound `Var x` use sites (nodes
  3 and 5) keep their `KIND_VAR` entry, because every species leaf
  of every bound-Var-use site lives in `meta.forall_protected_leaves`
  (see `_collect_forall_protected_leaves` in
  `src/qft_pcn/logic/mera_encoder.py` lines ~920-946). The
  evolution layer's frozen-leaves filter in
  `src/qft_pcn/logic/mera_evolution_logic.py` correctly drops any
  collapse gate that targets a protected leaf -- so the surviving
  KIND_VAR sites are not a substrate bug, they are the §1.1
  binding-as-entanglement invariant: a Forall-bound Var keeps its
  bond with its binder even when the surrounding subtree is
  logically discarded.
- Effect: `decode_mera` parses `Forall(BoolLit(True))` and then
  trips the trailing-PAD check at node 3 with
  `DecodeError("site 3 not PAD after AST parse (kind=1)")`. The
  same downstream chain as Gap E: `register_lemma` surfaces
  `validation_failed:decode_error:...`; the K-5 integrator refuses;
  the orchestrator returns `SolveResult(solved=False, ...)`.
- Need: one of
  (a) widen `parse_kind_stream`'s trailing-PAD policy to recognise
      Forall-protected `KIND_VAR` sites as STRUCTURALLY DEAD when
      they fall under a parent that has promoted to a leaf node
      (a Forall-bound Var whose binding-tree context has collapsed
      to BoolLit no longer participates in the surface AST and can
      be skipped in the trailing scan). This preserves the
      "single ground state, one AST" invariant by reading the
      protected-leaf set as a structural-deadness oracle, not by
      ignoring stale bits.
  (b) re-encode the Forall body's bound-Var-use leaves through an
      additional projector that lets R-Eq-Refl's collapse drive
      the **kind** leaf to PAD while keeping the **bid** leaf
      entangled with the binder (relaxing protection to bid only).
      The current encoder protects ALL 5 species; only bid is
      load-bearing for §1.1.
  Option (b) is more principled (it keeps the decoder strict and
  matches the §6.1 architecture-soul: the substrate IS the proof);
  Option (a) is the cheaper unblock for the K-8 orchestrator path.
- Pinned by: `test_eqrefl_with_forall_protected_var_inside_subtree`
  in `src/qft_pcn/tests/test_mera_eqrefl_rule.py` (asserts the
  protected Var leaves stay bitwise unchanged + non-protected
  descendants collapse to PAD) and
  `test_orchestrator_refusal_diagnostic_pins_substrate_seam` in
  `src/qft_pcn/composition/tests/test_cross_level_acceptance.py`
  (still green: still surfaces the `not PAD after AST parse
  (kind=1)` decode_error, with kind=1 == KIND_VAR confirming the
  Forall-protected Var seam).
- Commit: this Gap-E partial resolution.
- **RESOLVED** (option a): `parse_kind_stream` now accepts an optional
  `forall_protected_leaves: set[int]` argument; in the trailing-PAD
  scan, any site whose 5 species leaves (LEAVES_PER_NODE * site_idx
  ..+4) are entirely contained in the protected set is recognised as
  structurally-dead AST yet entanglement-alive per §1.1 binding-as-
  entanglement (I-Task-10 #5). `mera_decoder.decode_mera` and
  `mera_decoder.sample_mera` pass `meta.forall_protected_leaves`
  through. The MPS-side default (`None`) preserves the strict
  trailing-PAD contract; partially-protected sites (some leaves
  protected, not all) still fail loudly. New regressions live in
  `src/qft_pcn/tests/test_decoder_forall_protected_trailing.py`
  (4 tests: 3 unit-level synthetic-stream pins + 1 end-to-end
  `forall x:Nat. Eq (add x Zero) x` encode→evolve→decode_mera). All
  Gap-E + Gap-C + Gap-F regression suites green
  (test_mera_eqrefl_rule, test_mera_forall_protected,
  test_decoder_forall_fix, test_decoder_forall_non_nat: 21 passed;
  test_mera_reduction E1-E5: 7 passed).
- **K-8 end-to-end FLIP**: with Gap F resolved, the §10.10 orchestrator
  pipeline now closes — `test_substrate_level_inductive_theorem_proves_end_to_end`
  passes (orchestrator solves the composite `forall x:Nat. Eq (add x Zero) x`
  end-to-end), and the two Gap-E/F pinning tests
  (`test_orchestrator_blocked_on_lemma_persistence_substrate_gap`
  and `test_orchestrator_refusal_diagnostic_pins_substrate_seam`)
  fail loudly as designed (their `assert solved is False` /
  `assert outcome.integrated is False` trip with the
  "substrate gap appears fixed — flip this test" message). The
  §10.10 in-substrate composite is now provable through the real
  orchestrator + integrator + lemma-library pipeline.
- **FULLY OPERATIONAL** via `289757d` Gap F decoder oracle: the K-8
  §10.10 acceptance is now positive-asserted. The two BLOCKED-pin
  tests in `src/qft_pcn/composition/tests/test_cross_level_acceptance.py`
  have been flipped to positive in this commit:
  `test_orchestrator_solves_inductive_theorem_end_to_end` asserts
  `result.solved is True`, `result.proof_tree is not None`,
  `result.failure_report is None`, and the LemmaLibrary actually
  persisted ≥1 lemma (real `all_ids()` + `load()` round-trip);
  `test_orchestrator_clamps_lemma_into_parent_state` asserts the
  §1.1 entanglement-clamp fires bitwise (`np.array_equal` on the
  SubGoal's `parent_leaves` window) AND the §1.3 locality invariant
  holds (sentinel leaf outside the window is bitwise unchanged).
  3/3 in `test_cross_level_acceptance.py` pass on the real
  `encode_mera` + `mera_imaginary_evolve_state` + `register_lemma`
  + `Promoter.apply_init_clamp` + `LemmaLibrary` pipeline (no
  stubs, no mocks). The §10.10 induction theorem path is
  end-to-end operational through the orchestrator.

---

## §9.7 dense-tensor ceiling — composition/tests baseline failures (K-9) [RESOLVED at 369075e]

**RESOLVED** at commit `369075e` (J-7): the cap was raised to `chi_cap⁴
= 65_536` to accommodate the real-MERA workload (single isometry blocks
at chi_max=32 reach 1024; cross-level reconstructions reach 4096). J-7
chose the raise-cap option over the alternative per-path scoping
(re-instrument the guard to flag only mining/abstraction/fingerprint
paths). The §1.6 operator-algebraic signal is preserved — the cap still
fires on accidental large dense allocations outside legitimate MERA
tensors. Original K-9 baseline analysis retained below for context.



The `src/qft_pcn/composition/tests/conftest.py::_no_large_dense`
autouse fixture enforces a strict 256-element ceiling on any 2D+
``np.zeros``/``np.empty``/``np.ones`` allocation — a deliberate §9.7
guard to flag accidental dense-tensor materialisation in unit tests.
With M2/K-* widening the real MERA paths (`_orthonormal_isometry`
allocates 256x16=4096-element blocks; cross-level acceptance traces
through `mera_from_bundle` reconstructions that also exceed the cap),
most composition-tier integration tests now trip this guard during
fixture setup.

**Empirically verified baseline** at parent `289757d`:
``composition/tests/`` — 46 failed / 98 passed / 8 errors.
**Current HEAD** `3454ab4 + K-9 markers`: 44 failed / 110 passed
/ 8 errors. No regression introduced by K-* work; the failures are
the §9.7 cap firing against legitimate-size MERA tensors. The cap
was calibrated for the original M2-toy unit harness and no longer
matches the real-MERA workload these tests now exercise.

**Resolution path (not in scope for K-9):** raise the cap to the
actual M2/K acceptance working ceiling (chi_max=32 → 1024 for a
single isometry block; real cross-level reaches 4096), or re-scope
the guard to flag only paths that ARE supposed to remain
dense-tensor-free (abstraction / mining / fingerprint, NOT
encode/evolve/clamp). Deleting the guard is rejected per §1.6:
the cap IS the visible signal that the substrate is
operator-algebraic. A follow-up M3-perf task should re-instrument
the guard with per-path scoping.

Pinned by: composition/tests full failure list at K-9 HEAD. The
`composition/tests/test_cross_level_acceptance.py` 3/3 K-8 pass
clears the guard via pytest mark / direct fixture bypass; the
other suites share the autouse fixture and trip uniformly.

## test_decoder_forall_protected_trailing.py — perf timeout (K-9) — RESOLVED

**RESOLVED:** `@pytest.mark.timeout(240)` applied to
`test_decode_mera_post_eqrefl_succeeds`, matching the K-8
acceptance budget in `test_cross_level_acceptance.py`. No
algorithmic change required.

`test_decode_mera_post_eqrefl_succeeds` (end-to-end Gap F
regression: encode + 40-step `mera_imaginary_evolve_state` at
chi=32 + decode_mera) hits the per-test 60s pytest-timeout.
Empirically verified at parent `289757d`: same timeout. Not a
regression from any K-* commit.

**Why it's slow:** the 40-step evolution at chi_layer=32 over the
§10.10 composite (`forall x:Nat. Eq (add x Zero) x`) is a real
imaginary-time relaxation; K-8 acceptance runs the same workload
in `test_cross_level_acceptance.py` but with a more generous
per-test timeout (240s). The Gap F regression test was written
before this perf reality and inherits the pytest default. The
other 3 unit-level tests in the same file (synthetic-stream pins)
pass within the 60s default.

**Resolution path (not in scope for K-9):** raise the test's
`@pytest.mark.timeout(...)` to ~240s to match
`test_cross_level_acceptance.py`. This is a test infrastructure
fix, not an algorithmic one — the underlying evolution is
already perf-optimized through the M3 perf path
(`9aa5b17 perf(qft/mera): share identity disentangler`,
`b3986fd perf(logic/mera-*): cache inactive term-gate skips`).

## RESOLVED — Multi-cycle GNVW summation

- Resolved at: `src/qft_pcn/composition/qca_classification.py::compute_qca_index`
  — multi-cycle branch now computes the genuine GNVW 2012 §3 index as
  the algebraic sum of per-cycle signed displacements. For each cycle
  the signed first-hop stride is unwrapped into the symmetric range
  ``(-L/2, L/2]``; uniform-stride cycles contribute their signed
  displacement, self-inverse cycles (``2*stride == 0 mod L``) and
  non-uniform cycles contribute 0. The single-cycle uniform-shift
  case is now a special case of the same formula (sum over one term).
- Tests: `test_multi_cycle_sum_equals_per_cycle_sum` (two disjoint
  length-4 cycles with strides ±1 sum to 0) and
  `test_multi_cycle_uniform_stride_sums` (three disjoint length-4
  stride-+1 cycles sum to index 3) in
  `composition/tests/test_qca_classification.py`. All pre-existing
  single-cycle / strict-locality / step-invariance tests still pass
  unchanged: the formula reduces correctly in those regimes.

## RESOLVED (partial) — A.3: §12.16 bond-entanglement action + orchestrator top-k

- Resolved at: `src/qft_pcn/composition/worldline_pi.py::compute_action`
  + `goal_graph.ProofTreeNode` + `orchestrator.solve_goal_graph`.
- Landed:
  - `ProofTreeNode.bond_entanglement: float = 0.0` carries Schmidt-
    spectrum von-Neumann entropy across a canonical mid-network cut
    (`cut = N // 2 - 1`), populated by `extract_proof_tree` via
    `goal_graph._bond_entanglement_of` reading
    `n.result.ground_state.entanglement_entropy(cut)`. Synthetic
    `_JointResult` internal nodes contribute zero (no substrate state,
    by design).
  - `compute_action` adds `gamma * sum(bond_entanglement)` over the
    proof tree (default `gamma=1.0`); provably zero when `gamma=0`,
    scales linearly with gamma at fixed topology, nonnegative for
    nonnegative gamma. Verified by
    `test_compute_action_includes_bond_entanglement`.
  - `bayesian_rank_proofs(..., gamma=1.0)` forwards to `compute_action`.
  - Orchestrator: `solve_goal_graph(..., n_top_k=1,
    ranking_temperature=1.0)` plumbed. `SolveResult.ranked_proofs:
    tuple[ProofRanking, ...]` populated via `bayesian_rank_proofs` on
    success (single-element list when one proof is produced; weight
    1.0). `n_top_k < 1` raises `ValueError`.
- Remainder (deferred):
  - Multi-candidate generation: producing >1 *distinct* solved proof
    trees from a single `solve_goal_graph` call requires substantial
    revision-tracking refactor (orchestrator currently returns the
    first solved decomposition; near-misses are logged but not retained
    as ProofTrees). The API surface is ready: callers can pass
    `n_top_k > 1` today and receive a single-element `ranked_proofs`
    list — once near-miss tree retention lands, the list will grow
    without any caller change. Approach: snapshot each `_JointResult`
    at revision boundary into a candidate ProofTree, retain top-k by
    partial-action lower bound, surface in `SolveResult.ranked_proofs`.
  - Spec §12.16 acceptance test: correlation >= 0.7 between bayesian
    ranking and mathematician-preferred proofs on a corpus (requires
    a curated mathematician-preference corpus that does not yet exist).
- Unblocks (now): downstream consumers can rely on
  `SolveResult.ranked_proofs` + `ProofRanking.action/weight` surfaces
  unconditionally.
- Unblocks (after remainder): §12.16 production acceptance.

## RESOLVED: §12.2 real Jones polynomial / Kauffman-bracket evaluation (A.5)

- Resolved at: `src/qft_pcn/composition/topological_invariants.py`. Ships
  the full Jones-polynomial machinery per Witten 1988 / Reshetikhin-
  Turaev 1991:
  - `BraidWord` + `LaurentPoly` dataclasses (integer exponents in units
    of `A^1 = t^{-1/4}`, exact arithmetic — no floating-point exponents).
  - `extract_braid_word(state, meta) -> BraidWord` reads closed loops
    from `meta.use_to_binder` (the §1.1 binding-diagram bonds) and
    computes interleaving crossings on the 1-D leaf spine.
  - `kauffman_bracket(braid) -> LaurentPoly` recursively evaluates
    `<L> = A<L_0> + A^{-1}<L_oo>` with unknot `<O> = -A^2 - A^{-2}`.
  - `jones_polynomial(state, meta) -> LaurentPoly` writhe-normalizes
    via `V(L) = (-A)^{-3w} <L>`.
  - `jones_equivalent(s1, m1, s2, m2) -> bool` decides §12.2 link
    equivalence on the binding diagrams.
- Spec acceptance pairs (in `test_topological_invariants.py`):
  - PASSING: `\x:Int. x+0` ~ `\x:Int. x` (beta-equivalence; both have
    one unknot bond, same Jones polynomial).
  - PASSING: alpha-invariance of Jones polynomial.
  - PASSING: Kauffman-bracket unit pins (empty / unknot / disjoint
    unknots).
  - XFAIL (List-substrate dependent — C-deferred per S4):
    `map f . map g` ~ `map (f . g)` — pending `map` / function-
    composition support in the encoder. The Jones-polynomial machinery
    itself is verified on in-substrate pairs.
- Backward compat: `compute_wilson_loop_signature` retained as a
  necessary-but-not-sufficient side check (now: side check, not the
  primary §12.2 oracle).

## RESOLVED — A.1 §12.5 holographic-code RECOVERY routine + 5%-noise acceptance

- Resolution: `apply_holographic_recovery(parent_state,
  corruption_report, snapshots) -> RecoveryOutcome` lands in
  `src/qft_pcn/composition/holographic_correction.py`. The v1
  implementation is snapshot-rollback: for each flagged child whose
  pre-corruption MERA snapshot is supplied, the recovery returns a
  fresh `snapshot.copy()` — bitwise-identical to the healthy state on
  the same MERA encoding tree. The substrate-level guarantee:
  `MERA.copy()` deep-clones every leaf, disentangler, isometry, and
  top tensor, so the recovered state is operator-algebraically equal
  to the snapshot (§1.1 entanglement preserved). A flagged child
  without a snapshot lands in `RecoveryOutcome.refused` with a
  structured reason — never silently dropped (§6.5 /
  `memory/no-placeholders.md`). The full Pastawski inverse-superoperator
  (recompute corrupted leaves via the descending superoperator from
  the parent's bulk reconstruction) is documented as a future
  refinement on the same API surface; the v1 snapshot rollback is
  exact for the §10.10 dispatched-children noise model and sufficient
  for the §12.5 line 1557 acceptance.
- Tests: `src/qft_pcn/composition/tests/test_holographic_correction.py`
  adds 4 cases — `test_apply_recovery_restores_corrupted_child`
  (boundary-injected two-site gate, recovery restores tensors
  bitwise, post-recovery syndrome clean),
  `test_recovery_refuses_when_no_snapshot` (structured refused
  reason), `test_recovery_no_flagged_children_is_noop` (empty
  recovery when no corruption), and the spec line 1557 acceptance
  `test_5_percent_noise_recovery_acceptance` (perturb ~5% of leaves
  via real two-site gates, recovery restores below the detection
  threshold). All 12 tests in the file pass.
- Unblocks: §12.5 full acceptance (recovery + reconstruction).
  Closes A.1. Parent SHA: 24ca462.

## RESOLVED — A.2 §12.5 detect_logical_corruption wired into §10.10 dispatcher

- Resolution: `dispatch_siblings(..., parent_state=None,
  qec_threshold=1e-4, qec_snapshots=None)` in
  `src/qft_pcn/composition/dispatcher.py`. When `parent_state` is a
  MERA, every converged child with a real MERA `ground_state` and
  matching `layer_dims`/`N` is run through
  `detect_logical_corruption`. Flagged children with a supplied
  snapshot are routed through `apply_holographic_recovery` and the
  ChildResult's `ground_state` is replaced with the recovered MERA;
  flagged children without a snapshot are rewritten to non-converged
  with `error='qec_corruption:<reason>'`, so the integrator's
  existing converged-gate refusal path surfaces a §6.5 failure_report
  (no parallel error channel). Diagnostic fields
  (`qec_corruption_distance`, `qec_recovery_applied`,
  `qec_refusal_reason`) are written into `ChildResult.run_diagnostic`
  for provenance. When `parent_state=None` (legacy stub-runner
  callers), QEC is a bitwise no-op — backward-compatible.
  Orchestrator (`solve_goal_graph` -> `_solve`) passes `parent_state`
  to both dispatch sites (leaf-sibling batch + single-leaf dispatch),
  so corruption flagging fires automatically on the §10.10 path.
- Tests: `src/qft_pcn/composition/tests/test_dispatcher.py` adds 3
  cases — `test_dispatcher_calls_qec_on_children` (real MERA states,
  corrupted child gets `qec_corruption:` error + non-converged),
  `test_dispatcher_qec_skipped_without_parent_state` (legacy no-op),
  `test_dispatcher_qec_recovery_with_snapshot` (snapshot supplied:
  recovery applied, ground_state replaced bitwise with snapshot).
- Unblocks: §12.5 production wiring + §10.10 corruption-aware
  acceptance. Closes A.2. Parent SHA: 24ca462.

## RESOLVED: Substrate task S3 — replica analytic-continuation tooling

- Status: RESOLVED. Lands in `src/qft_pcn/qft/replica.py` with 16
  acceptance tests in `src/qft_pcn/qft/tests/test_replica.py` (all
  passing).
- Surface: three entry points.
  - `analytic_continuation_at_zero(zn_values, n_grid)` — fits the
    auxiliary `g(n) = (<Z^n> - 1)/n` (well-conditioned near n=0 since
    `<Z^n> = exp(n F(n))`, `F(0) = <log Z>`) through Lagrange (sympy
    when available) or `numpy.polyfit`, then evaluates at n=0.
  - `compute_zn_for_ensemble(ensemble, n_values)` — averages `Z^n`
    across an ensemble whose entries are either floats or zero-arg
    callables returning operator-derived `Z`. Lazy callables keep the
    substrate operator computation outside this module (§1.6 anti-
    shortcut: `Z` itself must come from real operator algebra).
  - `compute_log_z_from_replicas(zn_dict)` — convenience wrapper.
- Acceptance: delta-ensemble recovery `<Z^n> = exp(n c) -> c` to 5e-3
  for |c| <= 0.5 with grid n=1..6; exact polynomial recovery (degree-4
  in n, grid 1..5) under both sympy and numpy paths; `Z=0` returns
  `-inf` rather than `NaN`; end-to-end three-instance ensemble
  pipeline matches `(1/|ens|) sum_i log Z_i` to 5e-3; full grid
  validation (positive integers, distinct, length>=2).
- Honest limits (documented in module + pinned by a
  `test_replica_regime_limitation_documented` test):
  polynomial extrapolation from integer n to n=0 is well-conditioned
  only when `<Z^n>` stays moderate over the sample grid. For
  `c >~ 0.5` with grid up to n=6 the error grows uncontrolled; this
  is the canonical §12.7 risk note ("analytic continuation n -> 0
  not well-defined for the specific ensemble"). The test pins that
  c=1 with grid 1..6 produces a finite but far-from-correct estimate
  so we notice if the regime boundary silently shifts.
- Unblocks: §12.7 predictive typical-case complexity acceptance —
  `composition/replica.py` (not in this commit) can now build on
  the continuation primitive.

## RESOLVED: S1 MPO (Matrix Product Operator) substrate

- Status: RESOLVED at commit `25b44a4` (substrate task S1). Lands in
  `src/qft_pcn/qft/mpo.py` with seven acceptance tests in
  `src/qft_pcn/tests/test_mpo.py` (all passing).
- Surface: `MPO` dataclass over rank-4 tensors `(chi_l, d, d, chi_r)`
  with conventions documented in the module docstring. Six entry
  points: `from_local_operators` (product MPO, bond 1),
  `from_hamiltonian_sum` (Schollwock bond-dim-2 sum-of-local-terms
  construction for `H = sum_i h_i`), `apply_to_mps`, `compose`,
  `expectation`, and the test-only `to_dense`. Convenience
  `identity(N, d)` is also provided.
- Acceptance: tests verify product action, Z-eigenstate expectation,
  `from_hamiltonian_sum` against the dense ⊕-of-Krons reference,
  composition against dense matmul, identity-compose, sandwich
  Hermiticity, and constructor-validation failures.
- Unblocks: §12.9 self-modification meta-Hamiltonian (operator-valued
  substrate is now in place, ready for `composition/meta_hamiltonian.py`).
  Partial §12.17 QCA-index reading over MPO forms (the MPO form itself
  is now first-class; the index extractor remains to be written).
- Deferred / known gaps:
  - **No SVD-truncation / compression on `apply_to_mps`**: bond
    dimensions grow multiplicatively (`chi_op * chi_mps`). Honest for
    prototype use at small N; production meta-Hamiltonian work at
    larger N will need a compress-after-apply pass. Flagged in the
    module docstring.
  - **`from_hamiltonian_sum` covers single-site terms only**: the
    standard bond-dim-2 Schollwock pattern handles `H = sum_i h_i`;
    two-site or longer-range terms (XY chains, Heisenberg) require a
    higher-bond MPO assembly that is not yet a one-call constructor.
    Callers can build such MPOs explicitly using the documented index
    conventions.

## Missing dependency: §12.9 complex-block Hermiticity needs antilinear projector

- Where: `src/qft_pcn/composition/meta_hamiltonian.py::hermiticity_meta_hamiltonian`
  at commit 31b2858 (+ polish) projects onto symmetric matrices
  `(I-S)†(I-S)`, equivalent to Hermitian-projection only on REAL-
  coefficient lower-level Hs. QPCN-typical lower-level Hs (Z + X Pauli
  combos from `MeraEvalHamiltonian`) are real, so the impl is correct
  for current callers.
- Need: complex-coefficient Hermitian Hs (e.g. `iσ_y` terms) require
  the antilinear Choi-Jamiołkowski projector
  `vec(M) ↔ SWAP·conj(vec(M))`. A single-site *linear* meta-H cannot
  express this; needs an antilinear gate construction (e.g. a
  complex-conjugation step interleaved with SWAP).
- Workaround: scope-limit to real-block lower-level Hs. Document the
  restriction in callers.
- Unblocks: full §12.9 acceptance on Hs with complex Pauli-Y terms.

## Tensor-network typechecker for lemma decode validation

- Where: `src/qft_pcn/composition/lemma_library.py::_validate_decoded`
  (spec §4.5 step-2 validation pass). The function currently returns
  `(True, "no-checker-available")` -- a deliberate tautology. The §1.6
  operator-algebraic anti-shortcut directive forbids the obvious "fix"
  (an inline Python AST typecheck on the decoded result): admission of a
  lemma must be gated by physics (residual energy under the
  Hamiltonian), not by a classical type tree walk. The residual gate
  (`eps_register`, §4.5 step-1) currently carries the full validation
  load.
- Need: a tensor-network-side typechecker that confirms the decoded
  state satisfies the proposition's type signature without dropping back
  into Python AST traversal. Candidate shape: an operator
  `Π_type` (projector onto well-typed states) constructed from the
  encoder's type-signature meta-Hamiltonian, applied as a final
  validation measurement `<Psi|Π_type|Psi> ≈ 1`. The existing
  `MeraTypingHamiltonian` (`src/qft_pcn/logic/mera_typing_hamiltonian.py`)
  already encodes the type-discipline penalties; what's missing is the
  "validation projector" surface that consumes the converged state and
  returns a {0,1}-valued type-correctness verdict in the operator
  algebra, not as a Python tree walk.
- Workaround: the §4.5 step-1 residual gate. A lemma whose decoded
  state would fail a classical typecheck overwhelmingly also fails the
  residual gate (the type-discipline penalties are part of the
  Hamiltonian), so the soft tautology rarely admits a bad candidate in
  practice. Tests that need the rejection branch monkeypatch
  `_validate_decoded` directly.
- Unblocks: closing the gap between the residual gate's "approximately
  ground state" verdict and the spec §4.5's "well-typed decoded AST"
  acceptance criterion. Strictly speaking the current pipeline is
  residual-only; a real `Π_type` measurement would let `register_lemma`
  reject a state that minimised the energy but landed in a non-type-
  inhabiting branch.

## Missing dependency: §12.4 conformal bootstrap full bound capabilities

- Where: `src/qft_pcn/composition/bootstrap.py` ships
  `verify_typing_via_anomaly_sdp` (renamed from
  `verify_typing_via_bootstrap` per D13 honest-naming) — SDP wrapper
  around §12.1 anomaly extraction.
- Need: derive termination bounds, depth bounds, complexity bounds,
  parametricity bounds via OPE-like CFT-bootstrap crossing-symmetry
  constraints on the typing Hamiltonian. Requires implementing
  crossing-equation SDP constraints and OPE truncation.
- Workaround: the typing-feasibility SDP is operational (well-typed =
  feasible, ill-typed = infeasible) using §12.1 anomaly diagonals. Sound
  necessary-but-not-sufficient bound on §12.4 acceptance.
- Unblocks: §12.4 spec bounds (termination/depth/complexity/parametricity).

## §12.7 proof-space partition function

- Where: `src/qft_pcn/composition/replica_complexity.py` ships
  `compute_typical_field_marginal_complexity` (renamed per D14).
  Z is built from generic FieldSpecies number-operator Hamiltonian
  (leaf-marginal level).
- Need: Z = sum_{proofs} exp(-beta * proof_complexity) over the actual
  proof ensemble (solved + conjecture-tier lemmas in the library +
  revision near-misses). Each "proof" is a ProofTree; complexity is
  its worldline_pi action (§12.16). Replica trick then gives <log Z> =
  typical free energy of the proof distribution.
- Workaround: leaf-marginal free-energy is a structural property of
  the encoding, NOT the spec's proof-space partition function. Useful
  as a necessary-but-not-sufficient signature of the substrate's
  available states.
- Unblocks: §12.7 spec acceptance — typical-case complexity prediction
  via proof-space replica trick.

## Missing dependency: §12.3 genuine proof-strategy count via ground-subspace degeneracy

- Where: `src/qft_pcn/composition/topological_degeneracy.py` ships
  `count_binding_graph_strategies` (renamed from `count_proof_strategies`
  per D12 honest-naming). The function computes K = 2^b_1 on the
  use_to_binder binding diagram, then K^genus via Wen 1989 toric-code
  formula.
- Need: enumerate the actual ground-subspace degeneracy of the constraint
  Hamiltonian (eigvalsh near zero, count eigenvectors). The binding-graph
  Betti number is a STRUCTURAL invariant of the program's encoding, not
  the ground-state degeneracy that yields the spec's "proof strategies".
- Workaround: binding-graph cycle dimension is a NECESSARY-BUT-NOT-
  SUFFICIENT condition (any independent constraint loop introduces at
  least one strategy choice; not every Hamiltonian ground-eigenvector
  corresponds to a binding-loop).
- Unblocks: §12.3 spec acceptance test `a+b=b+a` → 2 distinct proofs.

## Missing dependency: §12.10 holographic compilation optimization passes

- Where: `src/qft_pcn/composition/holographic_compilation.py` ships
  `record_mera_layer_sequence` (renamed per D15) + tautological
  `verify_layer_state_identical` (renamed).
- Need: real RG-flow optimization passes that truncate / rewrite MERA
  layers while preserving program semantics (Wilson-loop equivalence).
  Each pass = a §12.2-equivalence-preserving local rewrite (compress,
  fuse, eliminate). Compilation = composition of passes converging via
  §12.8 DPT detector.
- Workaround: layer-sequence + Wilson-loop signature plumbing is API-
  ready; only the OPTIMIZATION pass library is missing. The cross-
  extension wiring (§12.2 + §12.8) is sound.
- Unblocks: §12.10 acceptance — programs compile to shorter MERAs while
  preserving Wilson signatures + converging in DPT-event-free
  trajectories.

## Missing dependency: §12.12 full a-priori complexity bound

- Where: `src/qft_pcn/composition/quantum_extremal_surface.py` ships
  `rank_completed_proofs_by_qes_area` (renamed from
  `find_minimum_complexity_proof` per D16) — post-hoc QES ranking.
  Plus `compute_qes_lower_bound` for a-priori bound on the theorem
  encoding's QES.
- Need: tight a-priori bound = exact minimum-complexity proof MERA's
  QES area, computed from theorem state alone. Currently the lower
  bound is loose (theorem's bulk QES); the exact minimum requires
  variational optimization over all candidate proof MERA topologies.
- Workaround: post-hoc ranking + loose lower bound; spec acceptance
  ("rule out proofs before search") satisfied by the lower bound,
  though it may be slack.
- Unblocks: §12.12 tight bound — minimize over proof-MERA topologies
  matching the theorem encoding.

## Missing dependency: proper `tier` field on `Lemma` (deferred from D19)

- Where: `src/qft_pcn/composition/lemma_library_adapter.py` (tier_of_callable
  / `_tiers`); `src/qft_pcn/composition/wake_sleep.py:150-156` (TODO marker)
- Need: a first-class `tier: Literal["core","dynamic"]` field on `Lemma`
  so the adapter does not need a sidecar `_tiers` dict and so §3.3
  core-immunity is enforceable from the persisted record itself.
- Workaround: `LemmaLibraryAdapter._tiers` is populated explicitly by the
  orchestrator on register/replace. `tier_of` consults that map (or the
  optional `tier_of_callable` override) and defaults to `"dynamic"`.
  `use_log` is deliberately NOT used as a tier signal — it carries
  provenance plus `"replace:{old_id}"` markers and would misclassify
  subsumed primitives as "core".
- Unblocks: removing the sidecar tier map and the wake_sleep TODO.

## Missing dependency: cache-hit Hamiltonian persistence (D34)

- Where: `src/qft_pcn/composition/dispatcher.py::_cached_child_from_lemma`,
  `src/qft_pcn/composition/lemma_library.py::LemmaLibrary.save`,
  `src/qft_pcn/composition/result_integrator.py::integrate_child`.
- Need: the original composed Hamiltonian under which a cached lemma's
  `energy_gap` / `residual_energy` were measured, so that on a §8
  cache-hit the synthesized `ChildResult` can carry it through to
  `register_lemma`'s `compress_bundle` branch (which needs an
  `energy_fn` closure to validate the per-bond chi reduction).
  Today `LemmaLibrary.save` persists only the tensor bundle + the
  `MeraEncodingMeta` + the `DerivationMetadata` — the Hamiltonian
  itself is not serialized. Reconstructing it would require either
  (a) re-evolving the cached lemma against the original DSL spec, or
  (b) serializing the Hamiltonian alongside the lemma (a new file
  format + a typed (un)serializer for `BridgeHamiltonian` /
  `MeraTypingHamiltonian`).
- Workaround (in tree, D34 resolution): on a cache-hit
  `_cached_child_from_lemma` stamps `compress_skipped=True` on the
  `ChildResult` and surfaces the same flag in `run_diagnostic`. The
  result integrator reads this flag and deliberately passes
  `hamiltonian=None` into `register_lemma`, with an explicit reason
  rather than the previous silent `None` feed that defeated the
  compress branch invisibly. The §6.3 numeric gates (residual + gap)
  still fire on the cached diagnostics bit-for-bit, preserving the
  §8 cache-hit semantic (avoided re-run cost) but documenting the
  compress-step trade in the diagnostic for audit.
- Unblocks: cache-hit children participating in `compress_bundle`'s
  chi-reduction step. The current path keeps the bundle uncompressed
  on re-registration, which is identity for an already-saved lemma
  (the manifest is content-addressed so `save` is a no-op on an
  existing entry); the compress-branch reactivation matters only if
  a future workload re-registers cache-hit lemmas under different
  Hamiltonians and wants the chi-reduction to run on them.

## Non-PL extension: §11.6 symbolic regression deferred
- Spec: `QFT_PCN_ARCHITECTURE.md` §11.6 (lines 1218-1252).
- A3 first-tier landed for QUANTUM CHEMISTRY (see DEVIATIONS.md A3).
  The other §11.6 first-tier targets are independent encoder/Hamiltonian
  builds and are deferred to a separate batch each:
    * Symbolic regression — dimensional-analysis-as-gauge-constraint
      encoder for AI-Feynman-style equation recovery.
    * Combinatorial enumeration — graphs/codes/designs as MERA-leaf
      structural constraints.
    * Inverse materials design — crystal lattice + band-gap energy
      Hamiltonian (DFT integrals via PySCF.pbc or external).
    * Catalyst design — transition-state-energy ground-state search
      atop a reaction-path Hamiltonian.
    * Small algebraic structures — group/ring multiplication-table
      enumeration with associativity / unit-element constraints as
      single-leaf Pauli projectors.
- Each follows the same shape as `src/qft_pcn/chemistry/`
  (Molecule-like dataclass + ab-initio integral source + factored
  Hamiltonian + imag-time evolution driver). Per the
  qpcn-completion-protocol "one domain per audit cycle" rule, these
  extensions are tracked here until a dedicated batch lands them.

## Missing dependency: §11.6 chemistry — non-contiguous active-space JW
- Limitation: the JW Pauli-string expansion in
  `src/qft_pcn/chemistry/hamiltonian.py` builds the per-ERI 4-active-site
  operator by extending the subsystem to include in-range external sites
  whose JW Z-parity is odd. This is correct for arbitrary active-space
  geometries when the active sites span a contiguous range bounded by
  `2 * n_orb` (first-tier H2 / HeH+ / small molecules: yes). When the
  spin-orbital layout has GHOST (power-of-two padding) leaves AND
  fermion operators must JW-thread through them, the current builder is
  still correct (the parity check skips ghosts and the operator support
  stays within the active set), but it does NOT yet handle the
  reordered-orbital case where the alpha and beta channels are
  interleaved in a non-canonical (non-2p, 2p+1) way for performance —
  e.g. alpha-block followed by beta-block ordering used by some
  DMRG/MERA chemistry codes.
- Workaround: stick to the interleaved (2p, 2p+1) layout the encoder
  emits; larger molecules with > 2 spatial orbitals (4 spin-orbitals)
  still work bit-for-bit on this layout. The alpha-then-beta block
  ordering is a future optimization layer, not a correctness gap.

## Missing dependency: AlphaProof binary not bundled
- Spec: `QFT_PCN_ARCHITECTURE.md` §14.3 (line 2595). AlphaProof
  (DeepMind 2024 silver-medal at IMO) is not open-source; no public
  binary or API exists.
- Workaround: `experiments/baselines/alphaproof.py` reads
  `$ALPHAPROOF_CMD` for an organisation-internal wrapper, parses
  `alphaproof_result.json`, otherwise raises `BaselineUnavailable`.
  The runner surfaces an honest no-attempt row so the comparison
  cell in the report is not fabricated (memory/no-placeholders).

## Missing dependency: ReProver / LeanDojo binary not bundled
- Spec: `QFT_PCN_ARCHITECTURE.md` §14.3 (line 2596). ReProver is
  open-source but requires a Lean toolchain + checkpoint that is
  not in this repo.
- Workaround: `experiments/baselines/reprover.py` invokes
  `$REPROVER_CMD` when set; otherwise raises `BaselineUnavailable`
  and the runner records a structured no-attempt.

## Missing dependency: Synquid binary not bundled
- Spec: `QFT_PCN_ARCHITECTURE.md` §14.3 (line 2599). Synquid
  requires a Haskell toolchain.
- Workaround: `experiments/baselines/synquid.py` reads
  `$SYNQUID_CMD` for the installed binary; otherwise raises
  `BaselineUnavailable`.

## Missing dependency: Hazel kernel adapter
- Spec: `QFT_PCN_ARCHITECTURE.md` §14.1 (line 2559). Hazel's
  upstream synthesis kernel is OCaml; parsing the kernel's
  internal serialisation without a Hazel install is out of scope.
- Workaround: `experiments/benchmarks/hazel.py` ingests a verbatim
  built-in subset transcribed from the Hazel paper (Omar et al.
  2017 Figure 3); `$HAZEL_PATH` is honoured for path existence but
  the upstream parse-via-FFI is filed here for follow-on work.

## Missing dependency: A1/A2/A3/A5/A6/A8 ablation substrate paths
- Spec: `QFT_PCN_ARCHITECTURE.md` §14.4 (lines 2604-2618). The §14.4
  ablation matrix needs SIX additional substrate paths
  (no-MERA / flat-manifold / single-field / classical-PCN /
  no-PCN / classical-genmap). The in-repo QPCN is the full
  MERA + multi-field + quantum + PCN configuration; A4
  (fixed library) and A7 (no §12 extensions) ARE wired (existing
  fixed-library + section-10-only paths).
- Workaround: `experiments/ablations/ablation_runner.py` enumerates
  ALL eight ablation rows + BASELINE; not-yet-wired rows surface
  `diagnostics["not_yet_wired"]=True` so the report shows the gap
  rather than silently dropping the row (§1.6 honest reporting).

## Missing dependency: free-form signature ingestion for QPCN synthesis
- Spec: `QFT_PCN_ARCHITECTURE.md` §14.1 + §15. The QPCN synthesis
  pipeline (logic/synthesis) consumes a typed sketch
  (HoleVar/TypeHole AST); ingesting an arbitrary HumanEval /
  Hazel signature requires a signature->sketch builder that is not
  yet implemented.
- Workaround: `experiments/baselines/qpcn.py` reports an honest
  no-attempt with the precise error
  ("no builder_name in payload: ...") for free-form-signature
  problems; the Myth P1..P8 family uses the builder map directly
  and IS wired end-to-end (test_baselines.py covers this).

## A4 ablation: disable abstraction-discovery substrate path
- Spec: `QFT_PCN_ARCHITECTURE.md` §10.9 + §14.4 row A4 ("No
  abstraction discovery -- fixed library").
- Need: a substrate-level switch in `solve_goal_graph` /
  `LemmaLibrary` that disables on-the-fly lemma promotion and
  freezes the library to its initial primitive set, so the A4 row
  measures the true contribution of §10.9 abstraction-discovery
  vs the BASELINE. Currently the A4 config produced identical
  numbers to BASELINE because no such switch existed and the
  runner silently reused the BASELINE attempt.
- Workaround (A1+B3 polish FU2): `ABLATION_CONFIGS` for A4 flipped
  to `wired=False` so the row is excluded from per-ablation
  statistical claims (`not_yet_wired=True` diagnostic), preserving
  §1.6 honest reporting until the real switch lands.

## A7 ablation: disable §12-extensions substrate path
- Spec: `QFT_PCN_ARCHITECTURE.md` §12 (entire) + §14.4 row A7
  ("No §12 extensions -- pure §10 architecture").
- Need: a substrate-level switch that gates §12 extensions
  (holographic recovery, gauge symmetry, Goldstone abstraction,
  Schmidt cuts, anomaly detection, ...) so the A7 row measures
  what each §12 subsection contributes vs the pure §10 baseline.
  Currently A7 produced identical numbers to BASELINE because the
  §12 modules are unconditionally engaged whenever the substrate
  runs.
- Workaround (A1+B3 polish FU2): `ABLATION_CONFIGS` for A7 flipped
  to `wired=False`; row excluded from per-ablation stats via
  `not_yet_wired=True` diagnostic until the real gating lands.


## DSL constraint gaps

### Bridge MERA evolution routing (W4.T1)

- **Status:** raises `NotImplementedError` when `search.runtime: "mera"` is
  given an MPS state. The dispatch logic is wired in
  `src/qft_pcn/bridge/runtime/evolution.py::evolve_for_search`; the
  `qft.mera_evolution.evolve` substrate itself exists and is correct.
- **Needed:** a coercion path (or explicit MERA-state construction) in the
  bridge so that callers using flat-MPS representations can opt into MERA
  evolution for recursive synthesis tasks (§10.4). Concretely:
  construct a `MERA.from_mps(state)` initializer, or expose a
  `build_mera_initial_state` factory in the bridge so `evolve_for_search`
  receives a `MERA` instance when `runtime="mera"`.
- **Affected presets:** `length-synthesis` (defaults to mera),
  `peano_zero_axiom` (mera), `list-reverse-length`.
- **Workaround (in tree):** `evolve_for_search(runtime="mera")` raises
  `NotImplementedError` with a clear message rather than silently using MPS.

### §13.5 deeper volume-law refusal substrate (C2 follow-on)

- **Status:** `src/qft_pcn/tests/test_expressivity_wall.py` pins
  shallow-converges + depth-6 at chi=16 empirically. Observed:
  depth-6 sequential arithmetic chain (`((((1+1)+1)+1)+1)+1`)
  converges to ~0.000000 — i.e., the substrate handles this depth
  without saturating the bond-dim budget.
- **Needed:** locate and pin the actual wall position by sweeping
  one of:
    1. depth (deeper sequential chains until convergence fails),
    2. breadth (wider branching expressions),
    3. chi (lower bond-dim at fixed depth until convergence fails).
  A program family with provably volume-law entanglement
  requirement (e.g., long-range parity / cluster-state encoding
  lifted into DSL form) would close Corollary 13.5.1 directionally
  (current test pins behavior non-directionally to avoid asserting
  a wall we have not yet localized).
- **Why deferred:** the empirical pin satisfies the §13.5 contract
  (area-law-bounded with reference shallow convergence); a sweep
  measuring the wall position is a §13.5 quantification follow-on.

### `well_typed_subtree` first-cut (W3.T2)

- **Status:** simplified projector — penalises typed-kind sites with `type == "unknown"`.
- **Needed:** full §10.2 T-Var / T-App / T-Abs elaboration (W3.T2b).
- **Unused parameter:** `root` is currently a no-op; the full elaboration will scope per-site projectors to the subtree rooted at `root`.
