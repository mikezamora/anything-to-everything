# Visualizer — Deviations from QFT_PCN_ARCHITECTURE.md

Refreshed 2026-05-23 after the PCN + narrative + DSL plan landed (T1–T20 + the
mera_relax / bridge wire-ups). Audit cap honoured: search stopped at five
findings of comparable severity.

## Findings (this pass)

| ID | Severity | Area | Summary |
| --- | --- | --- | --- |
| D-7 | high | Section navigation | `mera_relax` and `bridge` panels exist in `PANELS` but are not listed in any section of `sections.ts`, so they are unreachable from the `SectionedLayerSelector` left rail. |
| D-8 | medium | DSL stepped flow | After `runDsl` returns a `run_id`, `SteppedFlowBar` only appends a chat note; it never registers the run with the store or opens a WebSocket, so step 3 (Verbalize) silently reads stale or missing observations and the user has to switch routes manually to see anything. |
| D-9 | medium | Chat surface | `ChatPane.send` discards `result.raw` (the LLM's natural-language text) and only emits the static string `"Emitted DSL into the editor."`. §9.1 / §9.3 frame the LLM as the verbalizer; suppressing its raw reply hides part of the contract. |
| D-10 | medium | Section taxonomy | `sections.ts` puts `manifold` under the **QPCN Fusion** section. The architecture §2.1 places the dynamic Riemannian manifold squarely in the **classical PCN substrate** ("This is the geometric substrate that all higher layers attach to."). The bidirectional QFT↔PCN bridge belongs to QPCN, but the manifold itself belongs to PCN. |
| D-11 | low | PCN-coupling labelling | `PcnCouplingPanel`'s QFT→PCN arrow width is driven by `qpcn_observable_energy` (the QPCN's variational ⟨H⟩) and rendered with the label `⟨O⟩`. The architecture §3.4 / §4.7.5 says the QFT→PCN feedback is per-observable expectations `⟨Ô⟩`, not the scalar energy. The explainer (`pcn-coupling`) discloses this as a "proxy", which keeps it from being outright wrong, but the on-canvas label is still ⟨O⟩ — readers who look at the diagram alone will mis-attribute. |

### D-7 details (highest severity)
- `src/qft_pcn/viz/web/src/panels/index.ts:37-38` registers `mera_relax: MeraRelaxPanel` and `bridge: BridgePanel`.
- `src/qft_pcn/viz/runs.py:279-280, 308-311, 423-429` dispatches them on request.
- `src/qft_pcn/viz/web/src/lib/sections.ts:20-74` declares only `qft.layers = [mps, mera, vqc, hamiltonian]`, `pcn.layers = [pcn-fields, pcn-dynamics, multifield]`, `qpcn.layers = [pcn-coupling, qpcn, manifold, logic]`. Neither `mera_relax` nor `bridge` appears.
- `src/qft_pcn/viz/web/src/components/SectionedLayerSelector.tsx:46-57` iterates `sec.layers` — so two real, working panels are unreachable from the primary navigation.
- Fix: add `mera_relax` and `bridge` to the QPCN section (both belong to §10.10 / §10.5 fusion machinery), with matching `layerSummaries`.

### D-8 details
- `src/qft_pcn/viz/web/src/routes/dsl/SteppedFlowBar.tsx:31-40`: `onRun` posts to `/dsl/run`, receives `{ run_id }`, appends a chat line — and stops. The store's `activeRunId`/`runs` map is never updated, so the Verbalize button (`onVerbalize`, lines 42-50) reads `run.frames[run.frames.length - 1]` from `s.runs.get(activeRunId)`, which is `undefined` unless the user manually switched to the Viz route and reconnected. The architecture §1.3 / §9.3 calls the DSL "the entire contract" between LLM and QPCN; the round-trip needs to actually round-trip.
- Fix: in `onRun`, call the existing `connectRun(run_id)` / equivalent WS bootstrap path and set `activeRunId` so the stepped flow can read its own outputs.

### D-9 details
- `src/qft_pcn/viz/web/src/routes/dsl/ChatPane.tsx:40-50`: on success the assistant turn is `text: 'Emitted DSL into the editor.'` plus an opaque `artifact`. The translate API returns `{ dsl, error, raw, validation_errors }` (see `src/qft_pcn/viz/web/src/lib/dsl.ts:13-23`); `raw` is the LLM's actual text. Discarding it means the chat pane is a stub of the verbalizer described in §9.3.
- Fix: render `result.raw` (truncated) in the assistant turn alongside the artifact pill.

### D-10 details
- §2.1: "A standard Friston/Bogacz-style hierarchical PCN with the unique addition of a **dynamic Riemannian metric**. […] This is the geometric substrate that all higher layers attach to."
- §4.1: `Manifold2D` is filed under "Components As Built" alongside Fields, Layer, Network — all PCN-substrate components.
- `src/qft_pcn/viz/web/src/lib/sections.ts:65` lists `'manifold'` inside the `qpcn` section, alongside `pcn-coupling` / `qpcn` / `logic`.
- The IntroQpcnPanel then claims the manifold as part of "QPCN Fusion", which misallocates the load-bearing classical-substrate component.
- Fix: move `manifold` out of `qpcn.layers` into `pcn.layers` (alongside `pcn-fields`, `pcn-dynamics`, `multifield`), and update both Intro panels' layer-summary lists.

### D-11 details
- `src/qft_pcn/viz/snapshots.py:697-700` sets `qpcn_observable_energy = float(np.real(q._last_energy))`.
- `src/qft_pcn/viz/web/src/panels/PcnCouplingPanel.tsx:83-88` renders the bottom arrow labelled `⟨O⟩` with width driven by that scalar.
- §3.4 maps "Observation → Target expectation `⟨Ψ|O|Ψ⟩`" and §4.7.5 lists the per-observable predictions as the actual feedback into PCN. `⟨H⟩` is the *aggregate* variational energy, not a per-observable expectation.
- The explainer (`lib/explainer.ts:391, 395`) labels this honestly as a "proxy", which mitigates but does not erase the canvas-label deviation. Fix would either rename the arrow label to `⟨H⟩` (matching the readout) or wire a real per-observable expectation up through `snapshot_qpcn` (the `pred_errors` dict already carries them).

## Methodology

- Re-read `QFT_PCN_ARCHITECTURE.md` §§1.1–4.7, §8, §9, §10.1, §10.8–10.10, §11.2 / §11.4.
- Walked every TSX panel in `src/qft_pcn/viz/web/src/panels/` (16 panels including the three Intro panels, the three new PCN panels, and the mera_relax / bridge additions).
- Walked the DSL route stack: `routes/DslRoute.tsx`, `routes/dsl/{ChatPane,DslEditor,RunOutputPane,SteppedFlowBar}.tsx`, `lib/{dsl,llm}.ts`, `components/RouteSwitcher.tsx`.
- Walked the section + selector: `lib/sections.ts`, `components/SectionedLayerSelector.tsx`.
- Cross-checked snapshot semantics in `src/qft_pcn/viz/snapshots.py` and run dispatch in `src/qft_pcn/viz/runs.py` against the architecture's "where implemented" tables (§3.4, §4.x).
- Verified that each new explainer entry (`mera_relax`, `bridge`, `pcn-fields`, `pcn-dynamics`, `pcn-coupling`) cites a real section and that the math TeX matches §3.x / §10.10.
- Audit cap honoured: stopped after five findings of comparable severity.

## Audit history

- 2026-05-23 (initial): six deviations (D-1 … D-6); all fixed.
- 2026-05-23 (post-fix): audit clean.
- 2026-05-23 (this pass, post PCN+narrative+DSL): five new deviations (D-7 … D-11) introduced by the recent expansion; queued for follow-up.

## Notes for future audits

- When a new panel ships, three files must move in lock-step:
  `panels/index.ts` (registry), `lib/sections.ts` (navigation), `lib/explainer.ts` (semantics). D-7 is the consequence of skipping the second.
- The §1.1 invariant ("variable binding = bond entanglement, never classical lookup") remains satisfied across the viz layer — the LogicPanel's bond-entropy chart and the MeraRelaxPanel's ∀-protected-leaves display both encode binding as a real entanglement / freezing signal, not a classical lookup.
- The new DSL surface honestly reports validation errors and never fabricates a DSL when the LLM fails. The remaining gaps (D-8 / D-9) are surface-completeness issues, not fabrication.
