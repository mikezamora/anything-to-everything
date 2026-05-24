<!-- src/qft_pcn/viz/EXTENSIONS.md -->
# Viz Deferred Extensions

Features the visualizer would render if the substrate exposed the needed
hook. Listed so we can wire them when the dependency lands — never stubbed
in code, never faked in the UI.

Each entry: **What's needed**, **Why deferred**, **Wire-up when ready**.

---

## RESOLVED

- **mera_relax — MERA imag-time relaxation panel** (resolved via blockers
  #1, #3, #4, #5, #6 + Gap C + 6a455ad): `_build_mera_relax` rebuilds the
  encoder/Hamiltonian pair via `encode_mera(parse(expr), n_nodes_max,
  chi_layer) -> (state, meta)` + `MeraEvalHamiltonian(meta=meta)`; per
  step `mera_trotter_step(..., frozen_leaves=meta.forall_protected_leaves)`
  drives relaxation while ∀-protected leaves stay bitwise stable;
  `snapshot_mera_relax` emits `total_energy`, `residuals`,
  `forall_protected_leaves`, `layer_bond_dims`, and a `decode_mera`-derived
  `ast_text` round-trip. `mera_relax.forall-add-zero` preset is the live
  §10.10 induction-theorem demo.
- **bridge — RunResult inspector** (resolved alongside mera_relax):
  `_build_bridge_problem` calls `bridge.runtime.run_problem(problem)` once
  at build time; `snapshot_run_result` surfaces `trotter_steps`, `energy`,
  `converged`, the M2 composition-layer additions (`ground_state`,
  `hamiltonian`, `meta`, `solved_ast`) as miniatures + a `<pre>` block.
  `bridge.physics-relax` preset runs the canonical demo DSL
  (single-field, two sites, occupation observable).
- **vqc — live training panel** (resolved 3e5ec44): `_build_vqc` rebuilds a
  `QuantumGenerativeMap`; per-step parameter-shift training toward a fixed
  Z-target drives `theta`. `vqc.parameter-shift-train` preset added.
- **logic — live relaxation panel** (resolved 974507d): `_build_logic`
  rebuilds an `EvalHamiltonian` + a logic-encoded MPS; per-step
  imaginary-time `factored_trotter_step` drives relaxation, and
  `snapshot_logic(enc, state)` now emits per-term `residuals` plus
  `total_energy`. `logic.beta-reduce` preset added.
- **qpcn — live parameter editing during pause** (resolved 254ee0b):
  `RunController._substrates` captures live substrate handles via a new
  `run_simulation(on_build=...)` callback; `POST /runs/{id}/params`
  forwards updates to `Hamiltonian.update_param`. `QpcnPanel` renders
  sliders for writable params while paused on an active run.
- **mera — isometry-violation indicator** (resolved aab100d):
  `snapshot_mera` emits `iso_residuals` (per-layer mean
  `‖W W† − I‖_F`); `MeraPanel` renders the residual sparkline + an
  `iso err (max)` readout.

---

<a id="logic-parse-only-presets-for-bridge-dsl-keywords"></a>
## logic — parse-only presets for bridge-DSL keywords (forall/Eq/Nat/List/Cons/Nil)

- **What's needed:** `logic.encoder.encode(ast, N, chi_max)` to support
  the extended-calculus node kinds (`Forall`, `Eq`, `Zero`, `Succ`,
  `NatLit`, `Nil`, `Cons`). Today the flat `_tensors.py` index space is
  capped at 65536, but encoding any of those nodes immediately overflows
  it (e.g. `Zero` requests index 73728, `Forall` 125952, `Cons` 107520).
- **Why deferred:** Adding presets that fail at substrate-encode time
  would ship a broken UX and break `test_viz_presets`. The §1.1 / §10.10
  induction-theorem demo for these node kinds already lives on the
  MERA-based `mera_relax` layer, where `encode_mera` does support them
  in full. The flat-MPS `logic` layer remains restricted to the
  arithmetic / if / lambda subset.
- **Wire-up when ready:** Expand `_tensors.py`'s leaf basis to cover the
  extended-calculus kind/value enumerations, then add the two presets
  (`logic.forall-add-zero`, `logic.list-cons-nil`) to `presets.py`.
  `_build_logic` already honours `params["logic"]["expr"]` and the
  PARAM_SCHEMA already documents the `expr` key.

<a id="hamiltonian-term-list-with-active-term-highlighting"></a>
## hamiltonian — term list with active-term highlighting

- **What's needed:** `snapshot_hamiltonian` to enumerate active terms with
  rule_id / site / coefficient (the logic snapshot already does this).
- **Why deferred:** Not exposed today.
- **Wire-up when ready:** `HamiltonianPanel` already has the layout slot;
  drop the surrounding `null` guard once `terms` is present.

<a id="pcn-layer-kl-divergence"></a>
## PCN — per-layer KL divergence

- **What's needed:** A `QFTPCNLayer.kl_divergence() -> float` method exposing
  the per-layer KL term so `PcnDynamicsPanel` can decompose F = accuracy + KL.
- **Why deferred:** Substrate currently exposes only the aggregate
  `free_energy(phi_below, kappa_R)`.
- **Wire-up when ready:** Snapshot `per_layer_kl` next to `per_layer_free_energy`
  in `snapshot_pcn_dynamics`; the panel already has the chart slot.

<a id="lossless-dsl-runspec-round-trip"></a>
## DSL — lossless RunSpec round-trip

- **What's needed:** `RunSpec` to carry an optional `dsl: dict | None` companion
  so `runspec_to_dsl` can return the original DSL verbatim.
- **Why deferred:** `RunSpec.params` is intentionally flat for fast dict-merge
  semantics; structured DSL is currently dropped on `dsl_to_runspec`.
- **Wire-up when ready:** Store the original DSL on `RunSpec.dsl` in
  `dsl_to_runspec`; `runspec_to_dsl` returns it verbatim when present.

<a id="llm-retry-with-validation-feedback"></a>
## DSL — LLM retry on validation failure

- **What's needed:** A retry loop in `generate_dsl` that feeds the validation
  errors back into the LLM as a follow-up turn.
- **Why deferred:** v1 surfaces raw output to the chat pane for manual repair.
- **Wire-up when ready:** Add `max_retries: int = 0` param; on validation
  failure, append a "this DSL failed validation: <errors>; fix and re-emit"
  message and retry up to `max_retries` times.

<a id="step-replay-mera-relax"></a>
## mera_relax — step-replay annotation for the §10.10 induction-theorem demo

- **What's needed:** Frame-by-frame "this leaf moved because of rule R-AddZero at site 4" overlays on `MeraRelaxPanel`. Snapshot already emits per-term residuals; the annotation logic is its own design problem.
- **Why deferred:** v1 ships the relaxation panel; the annotation system is out of scope.
- **Wire-up when ready:** A new `lib/mera-step-annotator.ts` that diffs adjacent frames' residuals and emits per-frame "what fired" labels for an overlay.

<a id="learn-heavy-miniviz"></a>
## learn — live mini-viz embedded in heavy-panel articles

- **What's needed:** A lightweight "shrunken-render" mode for R3F (manifold, multifield, MERA) and Three.js (manifold) panels so they can mount multiple times in a scrolling Learn article without tanking page perf.
- **Why deferred:** v1 embeds them as `miniViz` placeholder sections; cheap panels (PcnDynamicsPanel, PcnCouplingPanel) could already embed but the panel itself must opt in to a smaller default size.
- **Wire-up when ready:** Add a `<MiniPanel layer="…" fixtureFrameId="…" />` component; cheap-panel paths render live; heavy panels render an SVG snapshot.

<a id="cross-panel-concept-search"></a>
## viz — cross-panel concept search

- **What's needed:** Full-text search across Learn articles + ExplainerPane tab content + EQUATIONS glosses.
- **Why deferred:** Bundle the index at build time; v1 doesn't have the ergonomic bar to host the search UI.
- **Wire-up when ready:** A `lib/search-index.ts` plus a header-mounted command palette (`Cmd+K`).
