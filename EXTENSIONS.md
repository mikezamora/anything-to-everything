# Extensions — recorded missing dependencies

Per `memory/no-placeholders.md`: when a feature cannot be implemented
because a real dependency is missing, the gap is recorded here rather
than left as a TODO or stub. Each entry names the call site, what is
needed, the workaround currently in tree, and which acceptance criterion
it unblocks.

## Missing dependency: bridge RunResult does not surface MeraEncodingMeta / ground_state / solved_ast

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
