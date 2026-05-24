<!-- src/qft_pcn/viz/EXTENSIONS.md -->
# Viz Deferred Extensions

Features the visualizer would render if the substrate exposed the needed
hook. Listed so we can wire them when the dependency lands — never stubbed
in code, never faked in the UI.

Each entry: **What's needed**, **Why deferred**, **Wire-up when ready**.

---

## RESOLVED

- **mera_relax — step-replay annotation for the §10.10 induction-theorem
  demo** (resolved dd03df0): `lib/mera-step-annotator.ts` diffs adjacent
  frames' `layer_states.mera_relax.residuals`; any (rule_id, site) whose
  absolute residual drops by more than 0.05 emits a `StepLabel` with the
  drop magnitude and a `"R-${rule_id} fired at site ${site}"` description.
  `MeraRelaxPanel` consumes the active run's last 3 frames via the
  zustand store and renders a "Recent firings" list (one strongest label
  per frame) below the residual table.
- **learn — live mini-viz embedded in heavy-panel articles** (resolved
  a3297ed): `components/MiniPanel.tsx` is the `miniViz`-section component.
  Cheap layers (`pcn-dynamics`, `pcn-coupling`) live-render the real
  panel component inside a `transform: scale(0.5)` wrapper, fed a
  minimal synthetic Frame; heavy layers (everything else) render a
  static SVG placeholder with the layer name and a "Open the Viz route
  for live rendering" caption. `LearnArticle.tsx` dispatches `miniViz`
  sections to `<MiniPanel layer={...} fixtureFrameId={...} />` in place
  of the previous placeholder paragraph.
- **viz — cross-panel concept search** (resolved 2f0a5ae):
  `lib/search-index.ts` builds an in-memory inverted index at module
  load over EXPLAINERS (overview prose + element meanings + tab labels),
  EQUATIONS (gloss + per-symbol glosses), and ARTICLES (title + prose
  section bodies). Tokeniser is lowercased `/\W+/` with sub-3-char
  tokens dropped; scoring is matching-token count with source-order
  tiebreak; snippet is the first paragraph that hits the query,
  truncated to ~140 chars. `components/CommandPalette.tsx` mounts a
  Cmd+K (or Ctrl+K) modal at the App root that surfaces these results;
  layer-result clicks call `selectLayer(id) + setRoute('viz')`,
  article-result clicks set the URL fragment and `setRoute('learn')`,
  equation-result clicks switch to the Learn route as a best-effort
  navigation. Closes on Escape or overlay click.
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
- **PCN — per-layer KL divergence** (resolved af0dc57):
  `QFTPCNLayer.kl_divergence(phi_below)` returns the entropy-like
  portion of the per-layer free energy (drops the `kappa_R * R`
  geometric regulariser). `snapshot_pcn_dynamics` emits `per_layer_kl`
  alongside `per_layer_free_energy`; `PcnDynamicsPanel` adds an opt-in
  KL column to the per-layer table.
- **hamiltonian — term list with active-term highlighting** (resolved
  e44ae4b): `Hamiltonian.__init__` builds a parallel `terms` list of
  `{kind, species, site, coeff}` dicts (mass / curvature / source /
  quartic / density / yukawa / kinetic). `snapshot_hamiltonian`
  serialises and surfaces it; `HamiltonianPanel` renders an
  Active-terms table when `st.terms` is non-empty.

---

## RETIRED

- **logic — parse-only presets for bridge-DSL keywords** (retired —
  superseded by `mera_relax`): bumping `KIND_CUTOFF` from 8 to 16 in
  `logic/encoding.py` is necessary but not sufficient to make the
  proposed presets work. The widening (a) requires hand-updating
  several test-contract assertions that hard-code `KIND_CUTOFF=8` /
  `D_LOCAL=65536` (legitimate test-suite churn), and (b) doubles the
  per-site Hilbert dim from 65536 to 131072 (still factored, so it
  fits). However, a parallel encoder bug surfaces for `Forall` nodes
  in `_tensors.py:_bid_bond_tensor_at_site` ("VAR site k: candidate
  binder at lam_site=0 not in left_live (bookkeeping bug)"), so
  `forall x:Nat. Eq (x + Zero) x` still fails to encode on the flat
  MPS path even with the cutoff bump. The `mera_relax.forall-add-zero`
  preset already provides a working live demo for the same proposition
  via the MERA-native encoder, which is the §1.1 / §10.10
  induction-theorem demo's intended path. Retiring this entry.

---

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

