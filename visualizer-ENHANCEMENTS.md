# Visualizer — Enhancements

Refreshed 2026-05-23 after the PCN + narrative + DSL plan landed. Existing
entries E-1 … E-18 from the prior pass remain accurate; new entries E-19 …
E-30 cover surface area that did not exist when the previous list was
written.

## E-1: Surface a true free-energy time series, not just `mean_abs_ricci`
The manifold snapshot honestly notes "a true free energy needs an observation" and falls back to `mean(|R|)`. `snapshot_pcn_dynamics` now exposes `total_free_energy` and per-layer F — wire this scalar into the Manifold panel's MetricsStrip too so the geometric and dynamical views share one objective.

## E-2: Add a "stress-energy → metric" causal overlay on the Manifold panel
§3.2 (`dh_μν/dt = κ T_μν[E] − γ h_μν`) makes the metric *causally driven* by error stress-energy. With `snapshot_pcn_coupling.mean_abs_stress_energy` already available, render a small `T_μν[E]` inset next to `h_μν` on the Manifold panel with an arrow hinting at the sourcing relationship.

## E-3: Render per-bond truncation error history on the MPS panel
`evolution.trotter_step` returns the per-step truncation error (§3.3.5 / §4.7.4). Capturing and surfacing it on the MPS panel as a faded band overlaid on the entropy curve would make the panel a real "is χ_max too small?" diagnostic.

## E-4: Add a real-time vs imaginary-time toggle indicator
The QPCN does both inside one `observe` call (§4.7.5); the viz never tells the user which mode produced the current frame. A small chip ("τ-step 3 of 4, dt_imag = 0.05") would make the dynamics legible.

## E-5: Per-term residual breakdown for the Logic panel (delivered)
Already shipped in D-4 / `snapshot_logic.residuals`; the LogicPanel colours each term by residual intensity. Promote the `lambda_*` legend into a real "constraint debugger" table that lists rule_id / site / residual sorted descending — same pattern the MeraRelaxPanel uses for its 360-term enumeration. Directly delivers §8.2 + §10.6.

## E-6: Live entanglement-entropy → log(χ) saturation badge
The MPS panel draws the `log χ` ceiling but doesn't summarize how close the system is. A scalar `max(S(bond) / log χ_bond)` plus a coloured chip at > 0.9 would tell the user "bond dimension is your bottleneck now" — a tunable the architecture flags repeatedly (§2.3 / §3.3.3 / §10.7).

## E-7: Species-aware MPS chain
Both the MPS and Logic panels render the site chain monochrome. The Hamiltonian snapshot carries `species_dims` and `species`. Colour-coding each site by dominant species (or splitting the glyph into a per-species stack) would make multi-species runs (§3.3.2) immediately visible.

## E-8: Connect the QPCN panel to the Hamiltonian panel via a shared parameter view
Today, learnable params are shown only in the QPCN panel. The Hamiltonian *is* those parameters — they should also be visible next to the Hamiltonian's species/coupling readout. Hovering one should highlight the other.

## E-9: Goal-graph / hierarchical composition view
§10.10 / §11.2 describe a DAG of child QPCN runs. The viz currently shows only a single run. A new panel listing parent/child runs (read from the dispatcher's output) with edges and per-node residual energy would surface §10.8-§10.10 the moment the underlying machinery emits it. The bridge panel is the natural place to graft this on.

## E-10: Lemma library browser
§10.8 (lemma promotion) describes a persistent store of `(MPS tensors, proposition_type, derivation_metadata)`. A small panel listing registered lemmas with their type signatures and one-click "clamp into current run" would make compounding capability visible.

## E-11: Page-curve overlay enrichment
The MPS explainer mentions the Page curve but the panel draws only `log χ`. Add the true `S_page(L) = log d · min(L, N−L) − ½ · 1[L=N/2]` so the user can see how close the state is to maximally entangled.

## E-12: KaTeX/explainer fact-check pass
Each panel's `EXPLAINERS` entry should cite a *specific* anchor (e.g. `#33-quantum-field-theory-primitives`) instead of the generic top-of-document link.

## E-13: Curvature ξ control & coupling visible on the QPCN panel
`H.curvature_xi` (§3.3.4 / §4.7.5) is the bidirectional manifold↔QFT switch — turning it off should noticeably change run dynamics. Expose it as a paused-mode slider with a spark line showing manifold→QPCN feedback strength (e.g., `corr(R, ⟨n⟩)`).

## E-14: Compare-mode for per-term residuals (logic + mera_relax)
Compare-mode works for geometric panels but the LogicPanel and the new MeraRelaxPanel both ignore `baselineFrame`. A per-rule residual-delta column or stacked-bar (baseline vs current) view would make A/B comparison of e.g. different `λ_β` settings concretely useful.

## E-15: Polish — accessible colour choices
The diverging ramp on the manifold and the inferno scale on the Hamiltonian heatmap are not colourblind-safe. Adopt Viridis / Cividis (sequential) + ColorBrewer RdBu (diverging) and add legends with numeric extents.

## E-16: Replay scrubbing across recorded runs
`recorder.py` writes per-step frames; `runs.py` lists them. A scrubber + per-step diff between any two recorded runs (not just live vs single baseline) supports §14's iterate-and-compare workflow.

## E-17: Show the conservation-law signal
§1.1 calls out conservation laws as the architecture's inductive-bias source. For any species with U(1)-like number conservation, plot `total ⟨N⟩ = Σ_k ⟨n_k⟩` over time — `occupations_n` is already in `snapshot_qpcn`, so the metric is one reduce away.

## E-18: Things the viz already does well (worth preserving)
- Non-invasive snapshot extractors with defensive `_safe` semantics — recordings cannot perturb the simulation.
- Compare-mode side-by-side for MERA (`mera-disk-active` / `mera-disk-baseline`) is a clean pattern other 3D panels should copy.
- `MetricsStrip` cleanly factors scalar time series out of per-panel code.
- The per-rule colour scheme in `LogicPanel.ruleColor` is consistent and worth extending across panels.
- KaTeX-rendered formulas in the explainer (when correct) anchor each panel to a piece of math.

## E-19: Intro panels could embed live mini-readouts (new)
`IntroQftPanel` / `IntroPcnPanel` / `IntroQpcnPanel` are static prose right now. Each section has at least one obvious "section health" scalar (mean bond χ, total F, mean |T|↔|R| ratio). Surfacing those next to the section tagline would make the narrative pages actually live during a run, not just a static intro screen.

## E-20: Per-section MetricsStrip in the intro panels (new)
Once E-19 is in, a 3-line strip (one trace per section) would let a user compare "is the QPCN converging faster than the PCN is settling?" from a single screen — exactly the cross-substrate observation the architecture's "bidirectional coupling" claim (§3.4) is trying to demonstrate.

## E-21: PcnFieldsPanel — render the full hierarchy as a stacked column (new)
The current per-layer card row is good. An extra "stacked column" mode (all layers' Φ rendered as transparent contour planes stacked along z) would visually deliver §2.1's claim that the hierarchy *is* a vertical hierarchy of beliefs sharing one manifold.

## E-22: PcnDynamicsPanel — surface free-energy *gradient* per layer (new)
The current panel shows F, ‖E‖, mean Π per layer. Adding the per-layer step-to-step `ΔF / Δt` (one bar per layer) would isolate which layer is currently learning vs which has converged — §3.1's gradient-flow story made visible.

## E-23: PcnCouplingPanel — animate the two arrows with a phase offset (new)
The current static-width arrows are good. Adding a slow pulsation tied to `corr(T_μν, h_μν)` (with `T_μν` peaks leading by one frame) would make the *causal direction* of the bidirectional bridge visible — currently the user can tell magnitudes but not which side is leading.

## E-24: DslEditor — schema-driven autocompletion (new)
`@monaco-editor/react` + the served `/dsl/schema` are both wired. Hooking Monaco's JSON schema validation to the served schema (via `monaco.languages.json.jsonDefaults.setDiagnosticsOptions`) would give the LLM operator real autocomplete and inline error squiggles — turning the editor into a real DSL workbench (§9.2).

## E-25: ChatPane — render the raw LLM text and DSL artifact side by side (new)
Beyond fixing D-9 (showing `result.raw`), the artifact (parsed DSL) and the raw text should appear side by side in each assistant turn. The user can then compare the LLM's NL explanation against the structured DSL — the §9.1 division of labour made concrete.

## E-26: SteppedFlowBar — show the "DSL → MPS → Hamiltonian → ⟨H⟩ → NL" pipeline as a 5-node diagram (new)
The current three-button bar is functional but doesn't communicate the §9.3 operational flow. A 5-node mini-diagram with the active step highlighted and arrows lighting up as each step completes would teach the architecture as it runs.

## E-27: RunOutputPane — frame-history scrubber (new)
The pane shows only the last frame. A small slider letting the user scrub through the recorded `run.frames` array would let them watch the relaxation in the output pane without leaving the DSL route.

## E-28: MeraRelaxPanel — render the ∀-protected leaves on a leaf-strip (new)
The panel lists protected-leaf *indices* as a comma-separated string. A leaf-strip (one cell per leaf, protected ones outlined green) would make the §10.10 "binder stays bitwise stable" invariant immediately visible at a glance — directly satisfying the architectural-soul (§1.1) display requirement.

## E-29: BridgePanel — surface per-observable measurements and residual energies (new)
The bridge `RunResult` has `observables` and per-constraint residual energies (§9.2's last bullet). Today the panel only shows total energy + convergence flag. Add a small table of `{ observable name, target, measured, residual }` so the bridge panel actually shows the DSL's promised outputs.

## E-30: Section reassignment after D-7 fix (new)
After D-7 lands, `mera_relax` and `bridge` will join the QPCN section. Reorder the section's layer list so the narrative flows: `qpcn → mps/hamiltonian context → mera_relax (composition, §10.10) → bridge (LLM↔QPCN runtime, §10.5) → logic (the calculus this all encodes)`. This mirrors §11.2's bottom-up mechanism list and turns the rail into a teaching path.
