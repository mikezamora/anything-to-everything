# Visualizer — Enhancements

## E-1: Surface a true free-energy time series, not just `mean_abs_ricci`
The manifold snapshot honestly notes "a true free energy needs an observation" and falls back to `mean(|R|)`. Once any observation source is bound to a run (QPCN errors, multifield free energy, PCN `1/2 Π E² − 1/2 log Π`), emit a real per-step `F` scalar and add it as the primary MetricsStrip series. The architecture (§3.1, §4.3) makes `F` the load-bearing objective — it deserves first-class status above curvature.

## E-2: Add a "stress-energy → metric" causal overlay on the Manifold panel
The architecture (§3.2: `dh_μν/dt = κ T_μν[E] − γ h_μν`) makes the metric *causally driven* by the error field's stress-energy. Render a small inset showing `T_μν[E]` next to `h_μν`, with an arrow or animation hinting at the sourcing relationship; ideally show both fields' time derivatives so the user can watch curvature catch up to error spikes.

## E-3: Render per-bond truncation error history on the MPS panel
`evolution.trotter_step` returns the per-step truncation error (§3.3.5 / §4.7.4). Capturing and surfacing this in the MPS panel — perhaps as a faded band overlaid on the entropy curve — would make the panel a real diagnostic for "is χ_max too small for this problem?", which is otherwise invisible.

## E-4: Add a real-time vs imaginary-time toggle indicator
The QPCN does both real- and imaginary-time evolution within one `observe` call (§4.7.5). The viz never tells the user which mode produced the current frame. A small chip on the QPCN panel ("τ-step 3 of 4, dt_imag = 0.05") would make the dynamics legible.

## E-5: Per-term residual breakdown for the Logic panel
`snapshot_logic` already emits the per-term `residuals` parallel array. The panel could color each term node by its residual energy (red = stuck, green = satisfied) and show a "constraint debugger" table — directly delivering the architecture's §8.2 "constraint debugging" promise and §10.6 debugger goal.

## E-6: Live entanglement-entropy → log(χ) saturation badge
The MPS panel draws the `log χ` ceiling but doesn't summarize how close the system is to saturating it. A scalar `max(S(bond) / log χ_bond)` value, plus a colored chip when it crosses 0.9, would tell the user "your bond dimension is the bottleneck right now" — a key tunable mentioned across §2.3, §3.3.3, §10.7.

## E-7: Species-aware MPS chain
Both the MPS panel and the Logic panel render the site chain monochrome. The Hamiltonian snapshot carries `species_dims` and `species`. Color-coding each site by its dominant species (or splitting the site glyph into a stack of per-species rings) would make multi-species runs (§3.3.2) immediately visible.

## E-8: Connect the QPCN panel to the Hamiltonian panel via a shared parameter view
Today, learnable params are shown only in the QPCN panel as a bar chart. The Hamiltonian *is* those parameters — the same numbers should also be visible next to the Hamiltonian's species/coupling readout, so the user can see "this knob lives in `H.A.mass`". Hovering one should highlight the other.

## E-9: Goal-graph / hierarchical composition view
§10.10 and §11.2 (the composition mechanisms) describe a goal graph DAG of child QPCN runs. The viz currently shows only a single run. A new panel listing parent/child runs (read from the dispatcher's output) with edges and per-node residual energy would surface §10.8-§10.10 the moment the underlying machinery emits it.

## E-10: Lemma library browser
§10.8 (lemma promotion) describes a persistent store of `(MPS tensors, proposition_type, derivation_metadata)`. A small panel listing registered lemmas with their type signatures and one-click "clamp into current run" would make the library learning compounding visible — the central architectural pitch.

## E-11: Page-curve overlay enrichment
The MPS explainer mentions "Page-curve reference, random-state entropy ceiling (visual guide)" but the panel currently draws only `log χ`. Add the true Page curve `S_page(L) = log d · min(L, N−L) − 1/2 · 1[L = N/2]` so the user can see how close the system is to a random/maximally-entangled state.

## E-12: KaTeX/explainer fact-check pass
Each panel's `EXPLAINERS` entry should cite a *specific* anchor (e.g., `#33-quantum-field-theory-primitives`) instead of "Architecture §2.3" generically. The references currently all point at the same href. Granular anchors make it cheap to verify the panel against the architecture during later audits.

## E-13: Curvature ξ control & coupling visible on the QPCN panel
`H.curvature_xi` (§3.3.4, §4.7.5) is the bidirectional manifold↔QFT coupling switch — turning it off should noticeably change run dynamics. Expose it as a paused-mode slider and as a readout chip, with a small spark line showing manifold→QPCN feedback strength (e.g., `corr(R, ⟨n⟩)` over recent steps).

## E-14: Compare-mode for per-term residuals (logic)
Compare-mode is implemented for the geometric panels (Manifold/QPCN/Hamiltonian/MERA). The Logic panel ignores `baselineFrame` (line 233 `_baselineFrame`). A per-rule residual delta column or a stacked-bar (baseline vs current) view would make A/B comparison of, say, different `λ_β` settings concretely useful.

## E-15: Polish — accessible color choices
The diverging ramp on the manifold and the inferno scale on the Hamiltonian heatmap are not colorblind-safe. Adopting Viridis / Cividis for sequential, ColorBrewer's RdBu for diverging, and providing legends with numeric extents would help — the architecture explicitly targets being readable by non-physicist reviewers.

## E-16: Replay scrubbing across recorded runs
`recorder.py` writes per-step frames; `runs.py` lists them. A scrubber + per-step diff between any two recorded runs (not just live vs single baseline) would let a researcher answer "what changed between run 27 and run 42?" — directly supporting the iterate-and-compare workflow §14 (Evaluation methodology) calls for.

## E-17: Show the conservation-law signal
§1.1 calls out conservation laws as the architecture's inductive-bias source. For any species with U(1)-like number conservation, plot `total ⟨N⟩ = Σ_k ⟨n_k⟩` over time as a metric — when it drifts under real-time evolution, that is *a bug or a coherent source firing*. Either is worth seeing.

## E-18: Things the viz already does well (worth preserving)
- Non-invasive snapshot extractors with defensive `_safe` semantics — recordings cannot perturb the simulation. This is the right architectural decision and the audit found no leakage.
- Compare-mode side-by-side for MERA (`mera-disk-active` / `mera-disk-baseline`) is a clean pattern other 3D panels should copy.
- `MetricsStrip` cleanly factors scalar time series out of per-panel code — adding a new tracked metric is a one-line change.
- The per-rule color scheme in `LogicPanel.ruleColor` is consistent and worth extending (same rule = same color across panels).
- KaTeX-rendered formulas in the explainer (when correct) anchor each panel to a piece of math; this is the right pedagogical pattern.
