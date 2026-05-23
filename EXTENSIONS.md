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

## Missing dependency: SubGoal.parent_site is a single int, not a leaf tuple

- Where: `src/qft_pcn/composition/goal_graph.py:36` (`SubGoal.parent_site:
  int | None`) and `result_integrator._resolve_host_leaves`.
- Need: a parent-aware decomposer that publishes the *full* host-leaf
  window the child lemma is meant to occupy, e.g. `parent_leaves:
  tuple[int, ...]`. The current single-int field underspecifies the
  clamp: lemma footprints span `n_leaves > 1` host sites.
- Workaround: `_resolve_host_leaves` extends `parent_site` to the
  contiguous window `[parent_site, parent_site + child_meta.n_leaves)`.
  This is the principled one-shot expansion for a child that decomposed
  out of an isomorphic parent region and is what the integrator tests
  exercise; a non-contiguous or species-permuted layout would need the
  richer SubGoal field.
- Unblocks: integrator's call to `Promoter.compile_constraint(... leaves=
  [...])` for non-trivial decomposers (J-Task / decomposer follow-on).

## Missing dependency: orchestrator does not own a parent MERA

- Where: `src/qft_pcn/composition/orchestrator.py:85-86` (`parent_state`,
  `parent_meta` default to `None`) and the resulting
  `integrate_child(None, None, ...)` call.
- Need: the orchestrator should own (or accept) a parent MERA + meta so
  every integrated child's lemma is genuinely clamped into the parent's
  tensor network -- the §6.1 acceptance.
- Workaround: `integrate_child` registers the lemma (real I-Task-7
  surface) and marks the node SOLVED, but skips the
  `Promoter.apply_init_clamp` step when `parent_state is None`. The
  orchestrator's existing tests rely on this graceful skip; promoting
  the orchestrator to own a parent MERA is its own follow-on.
- Unblocks: §6.1 / §8.6 end-to-end clamp via the orchestrator entry
  point (currently only exercised by the unit tests in
  `test_result_integrator.py`).

## RESOLVED: `decoder.parse_one` rejects `KIND_FORALL` / `KIND_FIX` (Gap C)

- Status: RESOLVED on branch `claude/qft-pcn-hybrid-architecture-ihCIR`
  (pending commit). The Part-2 stub in `parse_kind_stream._parse_one`
  has been replaced with proper binder branches that mirror the
  `KIND_LAM` machinery: push a freshly-named `Forall` / `Fix` onto
  `binder_stack`, descend into the body, pop, and patch the body.
  `param_ty` recovery: Fix uses `_extended_type_from_tag(ti, ...)`
  because a `Fix`'s site type tag IS its `param_ty` tag; Forall
  defaults to `TNat()` because the site type tag is `TYPE_PROP`
  (Forall returns Prop), so the param type is not directly recoverable
  from the leaf — `TNat` is the canonical §10.10 lemma quantifier.
- Pinned by: `src/qft_pcn/tests/test_decoder_forall_fix.py`
  (`test_decode_forall_identity_body`, `test_decode_forall_with_eq_body`,
  `test_decode_fix_nat_body`) — three round-trips through `encode_mera`
  → `decode_mera` confirm the AST shape and binder-bound `Var`
  resolution.
- Downstream effect on the K-8 acceptance pin
  (`test_orchestrator_refusal_diagnostic_pins_substrate_seam`): the
  refusal reason flips from `validation_failed:decode_error:...
  Forall/Fix binder decoding is Part-2 scope` (Gap C) to the
  `_meta_to_json` `set is not JSON serializable` (Gap D). The pinning
  test still passes because its assertion accepts either Gap C OR
  Gap D as the surfacing reason; once Gap D lands the orchestrator
  path can persist Forall-rooted proofs end-to-end.

## Missing dependency: `_meta_to_json` does not handle `set` fields

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
- Workaround: none in-tree. Every call to `LemmaLibrary.save(...)`
  on a current encoder's meta raises before the lemma is persisted.
  The K-5 integrator's `register_lemma` call therefore cannot
  complete on any encoded AST today (Forall-rooted or not); the
  failure mode for non-Forall ASTs surfaces as a `TypeError`
  bubbling up to `integrate_child`'s `except Exception` guard, while
  Forall-rooted ASTs short-circuit on the decoder gap above first.
- Unblocks: K-Task-8 §10.10 acceptance end-to-end (paired with the
  decoder Part-2 fix above) AND any K-5 / I-7 integration test that
  exercises a real `LemmaLibrary` via `encode_mera`. Until this gap
  closes, the lemma-persistence path is exercised only by tests that
  construct meta objects by hand without `forall_protected_leaves`,
  or by tests that mock `_meta_to_json` directly.

## Bridge DSL: no `forall` / `Eq` / `Nat` / `List` surface (K-8 Blocker B)

- Where: `src/qft_pcn/bridge/dsl/schema.py` + the bridge DSL parser
  extension `52e9387` -- the AST gained `forall`, `Eq`, `Nat`, `add`,
  `Zero`, `Succ`, `NatLit` keywords, but the bridge DSL surface that
  `bridge.runtime.run_problem` consumes (the dispatcher's default
  child runner) has not been extended in parallel. There is no `List`
  surface anywhere yet.
- Need: bridge DSL schema entries for the extended-calculus
  vocabulary (`forall`, `Eq`, `Nat`, `add`, `Zero`, `Succ`, `NatLit`
  at minimum; `List`/`Cons`/`Nil`/`length`/`reverse` to land the
  spec's literal §10.10 theorem `forall xs : List A. length (reverse
  xs) = length xs`).
- Workaround: K-Task-8 acceptance (`test_cross_level_acceptance.py`)
  drives child runs via `encode_mera` + `mera_imaginary_evolve_state`
  directly, bypassing the bridge. This is principled for the
  inductive-theorem PATH (the substrate proof itself does not go
  through the bridge), but the spec's literal list-induction example
  is deferred until the bridge DSL grows the `List` surface.
- Unblocks: K-Task-8 acceptance on the spec's literal theorem (list
  induction). The §10.10 inductive-theorem PATH itself is exercised
  by the in-substrate composite `forall x:Nat. Eq (add x Zero) x`
  per the K-8 retry directive's Step 2 fallback.
