# Visualizer — Deviations from QFT_PCN_ARCHITECTURE.md

Refreshed 2026-05-23 after resolving D-7 … D-11 (commits 56d1ca4,
e5143f9, b1408f0, 131a9fc, 479ce7d).

## Findings (this pass)

None found. Audit clean on 2026-05-23.

## Resolution log (this pass)

| ID | Commit | Resolution |
| --- | --- | --- |
| D-7 | 56d1ca4 | `mera_relax` + `bridge` added to QPCN Fusion section in `lib/sections.ts`; both now reachable from `SectionedLayerSelector`. |
| D-8 | b1408f0 | Extracted `attachWs(runId)` helper in `lib/ws.ts`; `SteppedFlowBar.onRun` now opens the WebSocket + registers the run, so step 3 reads live frames. |
| D-9 | 131a9fc | `ChatPane.send` now appends `result.raw` to both success and failure assistant turns; on validation failure it also seeds the editor with `raw` so the user can repair inline. |
| D-10 | e5143f9 | Removed duplicated `manifold` entry from QPCN Fusion section; manifold lives only in PCN Substrate per §2.1. |
| D-11 | 479ce7d | Renamed down-arrow label from `⟨O⟩` to `⟨H⟩` in `PcnCouplingPanel`, matching the actual driver (`qpcn_observable_energy` = `q._last_energy`). Explainer's "proxy" caveat continues to disclose the §3.4 gap. |

## Methodology

- Re-read `QFT_PCN_ARCHITECTURE.md` §§1.1–4.7, §8, §9, §10.1, §10.8–10.10,
  §11.2 / §11.4.
- Walked every TSX panel in `src/qft_pcn/viz/web/src/panels/`.
- Walked the DSL route stack: `routes/DslRoute.tsx`,
  `routes/dsl/{ChatPane,DslEditor,RunOutputPane,SteppedFlowBar}.tsx`,
  `lib/{dsl,llm,ws}.ts`, `components/RouteSwitcher.tsx`.
- Walked `lib/sections.ts` and `components/SectionedLayerSelector.tsx`.
- Cross-checked snapshot semantics in `src/qft_pcn/viz/snapshots.py` and
  run dispatch in `src/qft_pcn/viz/runs.py` against the architecture's
  "where implemented" tables (§3.4, §4.x).
- Verified each explainer entry (`mera_relax`, `bridge`, `pcn-fields`,
  `pcn-dynamics`, `pcn-coupling`) cites a real section.
- §1.1 invariant ("variable binding = bond entanglement, never classical
  lookup") still satisfied across the viz layer.

## Audit history

- 2026-05-23 (initial): six deviations (D-1 … D-6); all fixed.
- 2026-05-23 (post-fix): audit clean.
- 2026-05-23 (post PCN+narrative+DSL): five deviations (D-7 … D-11) found.
- 2026-05-23 (this pass, post D-7..D-11 fixes): audit clean.

## Notes for future audits

- When a new panel ships, three files must move in lock-step:
  `panels/index.ts` (registry), `lib/sections.ts` (navigation),
  `lib/explainer.ts` (semantics). D-7 was the consequence of skipping
  the second; the audit checklist now explicitly walks all three.
- When extending `connectRun` / `attachWs`, the test suite covers the
  WS lifecycle (`lib/ws.test.ts`); extend those before adding new
  bootstrap paths.
- ⟨H⟩ vs ⟨O⟩ labelling: the QFT→PCN arrow currently shows the scalar
  variational energy. If a future iteration plumbs per-observable
  expectations through `snapshot_qpcn` (the `pred_errors` dict already
  carries them), the arrow can be relabelled ⟨O⟩ and the explainer's
  proxy caveat removed.
