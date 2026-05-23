# Visualizer — Deviations from QFT_PCN_ARCHITECTURE.md

None found. Audit clean on 2026-05-23.

## Audit history

- 2026-05-23 (initial): six deviations recorded (D-1 … D-6) in commit
  `1a54564`. See git history for the original entries and per-fix
  commits (`fix(viz): D-N …`).
- 2026-05-23 (post-fix): all six fixed. Re-audit found no new
  deviations.

## Methodology (this pass)

- Re-read snapshot extractors in `src/qft_pcn/viz/snapshots.py` and
  cross-checked against the architecture sections each panel maps to:
  §2.x / §3.2 / §3.3.4 / §8 / §10.1 / §11.4.
- Read every panel in `src/qft_pcn/viz/web/src/panels/`
  (Manifold, Multifield, MPS, Hamiltonian, QPCN, MERA, VQC, Logic).
- Verified each panel's rendered elements either (a) trace to a real
  snapshot field with semantics matching the architecture, or (b) are
  explicitly explainer-labelled as illustrative/structural with no
  claim of measurement.
- Verified the deviation-fix commits did not silently break adjacent
  panels via the full vitest + python viz test suite.

## Fixes landed in this pass

| ID | Panel / file | Fix | Commit |
| --- | --- | --- | --- |
| D-1 | QPCN — `snapshots.py` + `QpcnPanel.tsx` | Replace MPS tensor Frobenius norm (labelled "occupations") with real per-species ⟨n_k⟩ via `state.local_expectation(k, H.n(s))`. Keep tensor norm as `tensor_norms` sanity diag. | `a2c055a` |
| D-2 | Hamiltonian — `snapshots.py` + `HamiltonianPanel.tsx` | Surface real §3.3.4 coefficients (`per_species`, `density_couplings`, `yukawa_couplings`, `curvature_xi`); render 1D `curvature` as a strip aligned to the site axis; coupling matrix toggles g_ab / λ_ab. | `315e87a` |
| D-3 | Manifold — `ManifoldPanel.tsx` | Add height-channel toggle (h_xx default, h_xy, h_yy, tr(h)) so the rank-2 tensor structure of h_μν is visible. | `d03f322` |
| D-4 | Logic — `snapshots.py` + `LogicPanel.tsx` | Remove decorative λ-driven "binder-entanglement arcs"; colour terms by residual energy; add real per-bond `bond_entropies` chart from the logic MPS state — the load-bearing §1.1 binder-as-entanglement signal. | `8052d3c` |
| D-5 | Multifield — `MultifieldPanel.tsx` | Add per-pair signed `g[(a,b)]` readout cells + one MetricsStrip trace per coupling pair, so per-pair convergence is observable instead of being collapsed into mean \|g\|. | `193b459` |
| D-6 | MERA — `MeraPanel.tsx` | Drop the `kind = layer % 2 ? 'disentangler' : 'isometry'` scheme (no basis in the substrate). Every coarse node is now an isometry, matching what `snapshot_mera.isometries` actually exposes; explainer math updated to the real W_ℓ ∘ U_ℓ within-layer decomposition. | `e5313ed` |

## Notes for future audits

- Substrate is owned by parallel agents. Viz fixes here only touch
  `src/qft_pcn/viz/**`; if a future deviation requires substrate
  extension, add an `EXTENSIONS.md` entry instead of editing
  substrate.
- When a panel renders an element with a physics name (`occupation`,
  `binder`, `isometry`, `disentangler`), the rendered quantity MUST
  match the physics-name's meaning under the architecture. The §1.1
  invariant (binding = entanglement) is the load-bearing example.
- The `_safe(...)` defensive idiom in `snapshots.py` swallows
  `AttributeError | KeyError | TypeError` (optional attributes); it
  intentionally lets `ValueError`/`IndexError` propagate as real bugs.
  Don't widen the catch list when adding new fields.
