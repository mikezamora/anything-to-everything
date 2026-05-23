<!-- src/qft_pcn/viz/EXTENSIONS.md -->
# Viz Deferred Extensions

Features the visualizer would render if the substrate exposed the needed
hook. Listed so we can wire them when the dependency lands — never stubbed
in code, never faked in the UI.

Each entry: **What's needed**, **Why deferred**, **Wire-up when ready**.

---

## vqc — live training panel

- **What's needed:** `_build_vqc(spec)` + `snapshot_vqc(vqc)` re-added to
  `runs.py` / `snapshots.py`, plus a parameter-shift training step in the
  per-frame loop.
- **Why deferred:** Parallel substrate work removed both. `VqcPanel` is now
  fixture-only and renders an "extension pending" badge.
- **Wire-up when ready:** Re-add the builder + snapshot + training step,
  add a `vqc.*` preset to `presets.py`, drop the `isExtension` prop from
  `VqcPanel`.

## logic — live relaxation panel

- **What's needed:** `_build_logic(spec)` + `snapshot_logic(enc, state)`
  re-added, plus the `factored_trotter_step` call in the per-frame loop.
- **Why deferred:** Removed alongside vqc.
- **Wire-up when ready:** Same shape as vqc — rebuild + snapshot + add
  `logic.*` presets, drop the badge.

## qpcn — live parameter editing during pause

- **What's needed:** A substrate setter on `Hamiltonian` (e.g.
  `set_param(name, value)`) and a server route `POST /runs/{id}/params`
  that mutates the controller's active `QPCN` between steps.
- **Why deferred:** Snapshot exposes parameter values but no setter.
- **Wire-up when ready:** Expose `writable=true` for editable keys in
  `/params/schema`; `QpcnPanel` will show sliders for any key whose schema
  entry has `writable=true`.

## mera — isometry-violation indicator

- **What's needed:** `snapshot_mera` to return per-layer
  `‖U†U − I‖` (or similar) as `iso_residuals`.
- **Why deferred:** Not currently computed.
- **Wire-up when ready:** `MeraPanel` renders the residual sparkline next
  to the bond-dim readout.

## hamiltonian — term list with active-term highlighting

- **What's needed:** `snapshot_hamiltonian` to enumerate active terms with
  rule_id / site / coefficient (the logic snapshot already does this).
- **Why deferred:** Not exposed today.
- **Wire-up when ready:** `HamiltonianPanel` already has the layout slot;
  drop the surrounding `null` guard once `terms` is present.
