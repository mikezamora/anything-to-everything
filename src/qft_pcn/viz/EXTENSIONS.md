<!-- src/qft_pcn/viz/EXTENSIONS.md -->
# Viz Deferred Extensions

Features the visualizer would render if the substrate exposed the needed
hook. Listed so we can wire them when the dependency lands — never stubbed
in code, never faked in the UI.

Each entry: **What's needed**, **Why deferred**, **Wire-up when ready**.

---

## RESOLVED

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

<a id="hamiltonian-term-list-with-active-term-highlighting"></a>
## hamiltonian — term list with active-term highlighting

- **What's needed:** `snapshot_hamiltonian` to enumerate active terms with
  rule_id / site / coefficient (the logic snapshot already does this).
- **Why deferred:** Not exposed today.
- **Wire-up when ready:** `HamiltonianPanel` already has the layout slot;
  drop the surrounding `null` guard once `terms` is present.
