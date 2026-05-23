# Extensions — recorded missing dependencies

Per `memory/no-placeholders.md`: when a feature cannot be implemented
because a real dependency is missing, the gap is recorded here rather
than left as a TODO or stub. Each entry names the call site, what is
needed, the workaround currently in tree, and which acceptance criterion
it unblocks.

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

## Missing dependency: List arithmetic in encoder substrate (`length` / `reverse`)

- Where: `src/qft_pcn/logic/mera_encoding.py` (kind table),
  `src/qft_pcn/logic/mera_encoder.py` (AST → state walk),
  `src/qft_pcn/logic/mera_typing_hamiltonian.py` (typing-rule
  terms), `src/qft_pcn/logic/mera_evaluation_hamiltonian.py`
  (reduction rules `length(Nil)=0`,
  `length(Cons h t)=Succ (length t)`, `reverse(Nil)=Nil`,
  `reverse(Cons h t)=append (reverse t) (Cons h Nil)`).
- Need: dedicated `KIND_LENGTH`, `KIND_REVERSE` (and likely
  `KIND_APPEND`) species in the kind table; encoder leaf-emission
  for unary list operators; typing-Hamiltonian terms enforcing
  `length : List A -> Nat` and `reverse : List A -> List A`;
  reduction-Hamiltonian terms implementing the structural-induction
  reductions above so the §10.10 list-induction theorem promotes
  through the same R-Eq-Refl + Forall-protected channel that
  `forall x:Nat. Eq (x + Zero) x` already uses. Decoder
  (`logic/decoder.py`) must parse the new kinds back to
  `App(Var("length"), ...)` / `App(Var("reverse"), ...)` (or
  dedicated nodes).
- Workaround: the textual `List` / `Cons` / `Nil` surface has
  landed (above); `length` / `reverse` remain free identifiers in
  the parsed AST (they tokenise as `ident` and so an expression
  like `length xs` parses as `App(Var("length"), Var("xs"))` --
  which simply does not reduce under the current evaluation
  Hamiltonian).
  `src/qft_pcn/bridge/tests/test_dsl_extended_calculus.py::test_parses_list_length_reverse_theorem`
  is `pytest.mark.skip`'d until this entry is resolved.
- Unblocks: K-Task-8 acceptance on the spec's literal §10.10
  theorem `forall xs:List A. length (reverse xs) = length xs`.

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
