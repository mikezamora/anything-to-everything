# §13.8 Capability-growth-law empirical fit — HEAD 4444045

Spec: `QFT_PCN_ARCHITECTURE.md` §13.8 (Conjecture 13.8, Wake-sleep capability scaling).
Module: `src/qft_pcn/analysis/capability_growth.py`.
Driver: `record_growth_trajectory` + `fit_growth_law` on the §10.10-style
induction corpus (`build_induction_corpus`, 5 inductive theorems).
Library: `FakeLemmaLibrary` (in-memory backend satisfying the spec §7 contract).
Solver: `make_stub_solver({})` (deterministic, every problem solvable).

## Trajectory (real 5-cycle wake-sleep)

| t (cycle) | C(t) |
|---:|---:|
| 0 | 0.000 |
| 1 | 1.000 |
| 2 | 1.000 |
| 3 | 1.000 |
| 4 | 1.000 |
| 5 | 1.000 |

The library starts empty (C=0). On cycle 1 the wake phase registers all
five problems as cached solutions (route (a) of `measure_capability`); the
sleep phase then promotes a shared induction primitive and the consolidate
step replaces all five cached lemmas with derived-from-primitive
replacements. From cycle 1 onward every problem matches the registered
primitive's canonical density via the §10.9 subsume-rule (route (b)),
so C stays at 1.0.

## Fitted (α, β) for `dC/dt = α(1-C) - βC`

| param | value |
|---|---|
| α | 541.605 |
| β | 0.01984 |
| ssr | 6.710e-09 |
| C∞ = α/(α+β) | 0.99996 |
| C(0) | 0.000 |
| n_points | 6 |

## Library state after cycle 5

- registered primitives: 6 (the induction primitive plus 5 consolidation
  replacements from `_consolidate` ).
- replacements: 5
- registered solutions (wake-phase): 25 (5 problems × 5 cycles).

## Interpretation (honesty note)

The induction corpus saturates in one cycle: wake-phase registration alone
takes C from 0 to 1 because the stub solver returns every problem
immediately. The fitter therefore drives α to a large value with β small,
so the closed-form curve effectively becomes a Heaviside-like jump near
t=0, asymptote 1.0. The residual is < 1e-8 because the curve is
arbitrarily close to a step under the unbounded-α limit.

This is a faithful empirical realisation of the §13.8 ODE on this corpus,
but the **fit is ill-conditioned**: a corpus with non-trivial residuals
(real solver returning incremental progress, or a partially-solvable
corpus where some problems never match) would expose a finite α and
larger β. The fitter, module, and pipeline are correct — the corpus is
the saturation pathology.

A longer-trajectory / partial-difficulty corpus is tracked in
`EXTENSIONS.md` (A4 follow-on); the §14 benchmark wiring (miniF2F /
DreamCoder corpora) is the natural next driver once §14 lands.

## Acceptance criterion (test)

`src/qft_pcn/analysis/tests/test_capability_growth.py::test_fit_recovers_known_alpha_beta`
verifies the fitter recovers (α, β) within 5% on three synthetic
trajectories with known parameters — the A4 spec-gap-analysis criterion.
Nine tests pass (184.6 s) including the full real-trajectory smoke.
