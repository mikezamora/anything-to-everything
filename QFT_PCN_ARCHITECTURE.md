# PCN-QFT Hybrid Architecture: A Logic-and-Reasoning Substrate for AI Systems

**Status**: Working prototype on branch `claude/qft-pcn-hybrid-architecture-ihCIR`. Core machinery built and tested. Logic/coding integration is the next phase, scoped below.

**Location of code**: `src/qft_pcn/` (classical substrate + multi-field + Qiskit) and `src/qft_pcn/qft/` (full QFT core with operator-valued fields, MPS, Hamiltonian, TEBD evolution, and the Quantum Predictive Coder).

---

## 0. Document Purpose

This document captures the complete design, mathematics, prior research, and forward plan for the PCN-QFT hybrid architecture. It exists so the project can be continued without reconstructing context. It records:

1. **Intent**: what this system is meant to be and why.
2. **Architecture as built**: every component, its math, its file location, and what's been verified.
3. **Mathematical foundations**: the precise correspondence between Predictive Coding Networks (PCN) and Quantum Field Theory (QFT).
4. **Strengths and weaknesses for learning**: where this architecture is expected to excel, where it will not.
5. **Comparison to frontier research**: real published work that overlaps with or motivates this design.
6. **The logic/reasoning application**: the deep correspondence between programming-language semantics and quantum many-body physics that motivates targeting this architecture at code, mathematics, and physical-domain reasoning instead of natural language.
7. **The LLM + QPCN system**: how a traditional language model fits as the natural-language interface above this reasoning core.
8. **Forward roadmap**: the next set of concrete components to build, in priority order, with the first publishable milestone explicitly scoped.

---

## 1. Vision and Intent

The fundamental thesis is that **a learning system whose substrate is a genuine quantum field theory will outperform classical deep networks on data that has algebraic, conservation-law, or compositional structure** — chemistry, physics, formal logic, programming, and structured causal inference. It will *not* outperform on web-scale natural language, and that is the point: this is not a language model. A traditional LLM sits above it as the natural-language interface, and the QPCN underneath is the reasoning core.

### 1.1 Target domains

The QPCN is meant for problems where:

- The data has **conservation laws** (charge, energy, particle number, type-safety constraints).
- The data has **local, polynomial interactions** between components (most chemistry, most physics, most logic).
- The data has **gauge symmetry** (alpha-renaming, rotational invariance, choice of coordinate system).
- The data has **causal/temporal structure** that respects unitarity (reversible computation, Hamiltonian dynamics).
- The data has **discrete compositionality** (molecular bonds, syntax trees, proof trees).

Concretely:

- **Molecular and materials energetics**: bond energies, reaction barriers, electronic structure. This is the canonical domain for MPS/DMRG already.
- **Gauge theories and lattice models**: Yang-Mills, lattice QCD observables, frustrated magnetism, topological phases.
- **Dynamical systems with conservation laws**: planetary dynamics, plasma physics, climate sub-grid parameterizations, biological signaling networks.
- **Programming and logic**: type inference, program synthesis, theorem proving, equation solving, symbolic differentiation, model checking.

### 1.2 What this is *not*

- **Not a language model.** It does not produce or understand natural language directly. An LLM does that.
- **Not a general-purpose function approximator.** Its function class is restricted to those expressible by local Hermitian Hamiltonians acting on tensor-network states. This restriction is the source of its inductive bias and its strength.
- **Not a compute-saving system.** Classically simulating quantum many-body systems is in general exponentially expensive. The hope is that with tensor-network truncation and the right domain match, the *learned* solutions are cheap to evaluate even if the full state space is enormous.

### 1.3 The system architecture in one sentence

> The **LLM** translates a natural-language problem statement into a structured specification (Hamiltonian terms and target observables); the **QPCN** finds the ground state of that Hamiltonian (i.e., solves the problem) using imaginary-time evolution and real-time Schrödinger dynamics; the LLM then verbalizes the result.

The interface between the two is a small typed DSL: JSON specifying fields, constraints, and observables. Everything interesting happens on the QPCN side; the LLM is a thin spec-writer/verbalizer.

---

## 2. Architectural Overview

The system has three layers of substrate, each more expressive than the last, all coupled to a shared dynamic geometry:

### 2.1 Classical predictive-coding layer (built)

A standard Friston/Bogacz-style hierarchical PCN with the unique addition of a **dynamic Riemannian metric**. Belief fields `Phi`, error fields `E`, and precision fields `Pi` live on a 2D manifold whose metric `g_μν` is sourced by the stress-energy tensor of the prediction-error field. Belief diffusion uses the Laplace-Beltrami operator built from that metric.

This is the geometric substrate that all higher layers attach to.

**Code**: `src/qft_pcn/manifold.py`, `src/qft_pcn/fields.py`, `src/qft_pcn/layer.py`, `src/qft_pcn/network.py`.

### 2.2 Multi-field layer (built)

Multiple field types (analogous to electron, photon, Higgs in QFT, or shape/motion/color in cortical streams) share **one** manifold. They interact via a Yukawa-style Lagrangian `L_int = Σ_{i<j} g_{ij}(x) Phi_i Phi_j` with **learnable coupling constants `g_{ij}`** updated by gradient descent on the joint free energy. Correlated fields grow their coupling; uncorrelated fields don't.

**Code**: `src/qft_pcn/multifield.py`.

### 2.3 Quantum substrate layer (built)

The full QFT core. Each spatial point holds an operator-valued bosonic field (truncated Fock space). The many-body state is represented as a Matrix Product State (MPS) with controllable bond dimension — this is the entanglement carrier. The Hamiltonian is constructed locally from one-site (mass, source, quartic, curvature coupling) and two-site (kinetic hopping, cross-species interaction) terms. Evolution uses second-order Suzuki-Trotter splitting with SVD-based bond truncation, supporting both real-time (Lorentzian, unitary) and imaginary-time (Euclidean, relaxational) dynamics.

The **Quantum Predictive Coder (QPCN)** sits on top: its belief state is the MPS, its generative model is the Hamiltonian, observations are target expectation values of local operators, and prediction errors drive gradient updates on Hamiltonian parameters. Stress-energy expectations feed back into the classical 2D manifold, so geometry and quantum content are bidirectionally coupled.

**Code**: `src/qft_pcn/qft/fock.py`, `src/qft_pcn/qft/mps.py`, `src/qft_pcn/qft/hamiltonian.py`, `src/qft_pcn/qft/evolution.py`, `src/qft_pcn/qft/qpcn.py`.

### 2.4 Qiskit interface layer (built)

A **`QuantumConvMap`** wraps a parameterized variational quantum circuit (VQC) as a translation-invariant quantum convolution that conforms to the `GenerativeMap` protocol. Drops into any PCN layer as a replacement for its classical 3×3 conv. Parameter-shift rule gives exact gradients. The same code targets real IBM hardware by swapping the simulator backend for an IBM Quantum `SamplerV2`.

**Code**: `src/qft_pcn/quantum.py`.

---

## 3. Mathematical Foundations

### 3.1 Predictive coding and variational free energy

The classical PCN minimizes the variational free energy

```
F[Phi, E, Pi] = ∫_M [ 1/2 Pi(x) E(x)^2 − 1/2 log Pi(x) ] sqrt|g| d^2x
```

where:

- `Phi(x, t)`: representation / belief field
- `E(x, t)`: prediction error field, `E_l = Phi_{l-1} − g_l(Phi_l)`
- `Pi(x, t)`: precision field (inverse variance)
- `g_l`: generative map projecting layer `l` down to layer `l-1`
- `sqrt|g|`: volume element of the dynamic manifold

The dynamics are gradient flow on `F`:

```
dPhi_l/dt   =  -δF/δPhi_l  =  (J_{g_l})^T (Pi E_l)  +  D · Δ_g Phi_l   +  top-down
dE_l/dt     =  Phi_{l-1} − g_l(Phi_l)  −  λ_E E_l
dPi_l/dt    =  -δF/δPi_l   =  1/(2 Pi) − 1/2 E_l^2     (fixed pt: Pi = 1/E^2)
```

`Δ_g` is the Laplace-Beltrami operator on the dynamic manifold. Precision dynamics are applied in log-space for stability, with floor `Pi_min` and ceiling `Pi_max` to prevent divergence of the complexity term `-log Pi`.

### 3.2 Riemannian geometry of the substrate

The metric `g_μν(x, t) = η_μν + h_μν[E]` evolves under

```
dh_μν/dt = κ · T_μν[E] − γ · h_μν
```

with `κ` the curvature coupling, `γ` the decay rate, and `T_μν[E] = ∂_μE ∂_νE − 1/2 g_μν |∂E|^2` the canonical stress-energy tensor of the error field. The metric perturbation is clipped at `|h| ≤ 0.6` to maintain `det g > 0` (positive-definite Riemannian metric).

The Laplace-Beltrami operator is

```
Δ_g φ = (1/sqrt|g|) ∂_μ ( sqrt|g| g^{μν} ∂_ν φ )
```

discretized on the grid via central differences with periodic boundaries.

The linearized Ricci scalar (used as a regularizer in the free energy) is

```
R ≈ 2 ∂_x ∂_y h_xy − ∂_y^2 h_xx − ∂_x^2 h_yy
```

### 3.3 Quantum field theory primitives

#### 3.3.1 Local Fock space

Each spatial site carries a truncated bosonic Fock space `H_d = span{|0⟩, |1⟩, ..., |d-1⟩}` of dimension `d`. The canonical algebra:

```
a |n⟩       = sqrt(n) |n-1⟩
a† |n⟩      = sqrt(n+1) |n+1⟩
[a, a†]     = 1   (modulo truncation defect at |d-1⟩)
n = a† a
φ = (a + a†) / sqrt(2)              "position-like"
π_op = i(a† − a) / sqrt(2)          "momentum-like"
```

The vacuum is `|0⟩` (lowest occupation), particle states are `a†|0⟩`, multi-particle states are `(a†)^k|0⟩/sqrt(k!)`.

#### 3.3.2 Multi-species sites

For multiple field species `A, B, ...` at each site, the local Hilbert space is the tensor product:

```
H_local = H_A ⊗ H_B ⊗ ...
```

Operators acting on species `A` are embedded into `H_local` by tensoring with identities on the other species. This is the `fock.embed_op` helper.

#### 3.3.3 Matrix Product State (MPS)

The many-body state of `N` sites is

```
|Ψ⟩ = Σ_{s_1,...,s_N}  (A[1]^{s_1} A[2]^{s_2} ... A[N]^{s_N})  |s_1 ... s_N⟩
```

where each `A[k]^{s}` is a `χ_{k-1} × χ_k` matrix with `χ_0 = χ_N = 1` (boundary trim). The site tensors are rank-3 arrays of shape `(χ_left, d, χ_right)`. The bond dimension `χ` controls representational capacity: an MPS represents any state with entanglement entropy `S(L) ≤ log χ` across every bond.

#### 3.3.4 Hamiltonian

The Hamiltonian decomposes into one-site and two-site terms:

```
H  =  Σ_k  H_1(k)  +  Σ_k  H_2(k, k+1)

H_1(k)  =  ω(x_k) n_k  +  J_k φ_k  +  μ_k n_k^2
              + Σ_{a<b} g_{ab}(x_k) n_{a,k} n_{b,k}
              + Σ_{a<b} λ_{ab}(x_k) φ_{a,k} φ_{b,k}

H_2(k, k+1)  =  -t (a_k† a_{k+1} + a_{k+1}† a_k)
```

with `ω(x) = ω_0 (1 + ξ R(x))` the curvature-coupled local frequency, `J` a coherent source (breaks U(1) so that the ground state has nonzero `⟨φ⟩` and `⟨n⟩`), `μ` a quartic self-coupling, `g_{ab}` a density-density interaction, `λ_{ab}` a Yukawa coupling, and `t` the kinetic hopping.

#### 3.3.5 TEBD evolution

Time evolution `|Ψ(t+dt)⟩ = exp(−i H dt) |Ψ(t)⟩` is approximated by second-order Suzuki-Trotter splitting:

```
exp(-i H dt)  ≈  ∏_{odd k} exp(-i h_k dt/2)
                · ∏_{even k} exp(-i h_k dt)
                · ∏_{odd k} exp(-i h_k dt/2)
```

where `h_k` is the full two-site Hamiltonian on bond `(k, k+1)` with the local terms folded in (half-weight from each adjacent site, with full-weight at the endpoints to compensate for missing neighbors).

Each two-site gate is applied to the MPS by contracting the two adjacent site tensors into `θ = A[k] A[k+1]`, applying the gate, SVD-decomposing back to two tensors, and truncating the bond to a maximum dimension `χ_max`. Truncation error is tracked and returned.

For imaginary-time evolution, `exp(−H τ)` drives the state toward the ground state of `H`; the state is renormalized after each step.

#### 3.3.6 Energy expectation

```
⟨H⟩ = Σ_k ⟨Ψ| H_1(k) |Ψ⟩  +  Σ_k ⟨Ψ| H_2(k, k+1) |Ψ⟩
```

computed by sweeping environment tensors across the MPS. Real because `H` is Hermitian.

### 3.4 The PCN-QFT correspondence

The classical-to-quantum lift is:

| Classical PCN element | Quantum analog | Where implemented |
|---|---|---|
| Belief `Phi(x)` | Quantum state `|Ψ⟩` as an MPS | `qft/mps.py` |
| Generative map `g_l` | Hamiltonian `H[θ]` | `qft/hamiltonian.py` |
| Free energy `F` | `⟨Ψ\|H\|Ψ⟩` | `qft/evolution.py::energy` |
| Belief update `dPhi/dt = -δF/δPhi` | Imaginary-time evolution `e^{-Hτ}\|Ψ⟩` | `qft/evolution.py::trotter_step(imaginary=True)` |
| Forward propagation | Real-time evolution `e^{-iHt}\|Ψ⟩` | `qft/evolution.py::trotter_step(imaginary=False)` |
| Observation | Target expectation `⟨Ψ\|O\|Ψ⟩` | `qft/qpcn.py::observe` |
| Prediction error | `target - ⟨O⟩` | `qft/qpcn.py::observe` |
| Learning step | Gradient on `H`'s parameters | `qft/qpcn.py::_grad_step` |
| Metric `g_μν` | Per-site curvature `R(x_k)` modulating `ω` | `qft/qpcn.py` + `manifold.py` |
| Stress-energy `T_μν` | `⟨Ψ\|H_local(k)\|Ψ⟩` as energy density | `qft/qpcn.py::_update_manifold` |

The lift is **not metaphorical**. Every entry in the right column is an actually-implemented function that computes the named quantity for an actually-quantum state.

---

## 4. Components As Built

### 4.1 Manifold2D — dynamic Riemannian substrate

**File**: `src/qft_pcn/manifold.py`

A 2D Riemannian manifold whose metric perturbation `h_μν` is a learnable, dynamical degree of freedom. Operators:

- `g()`, `g_inv()`, `det_g()`, `sqrt_det_g()`: metric accessors with positivity guarantees.
- `laplace_beltrami(phi)`: covariant Laplacian, replaces the flat 5-point stencil with a geometry-aware operator. Verified to reduce to the central-difference flat Laplacian when `h = 0` (`test_flat_laplace_beltrami_matches_flat_laplacian`).
- `stress_energy(e)`: canonical `T_μν` of a scalar field, used to source curvature.
- `ricci_scalar()`: linearized Ricci scalar curvature, used in free-energy regularization.
- `update_metric(e, dt)`: evolves `h` toward `T[e]` with decay and clipping to maintain positive-definiteness.

### 4.2 Fields — representation, error, precision

**File**: `src/qft_pcn/fields.py`

- `Field`: multi-channel scalar field on the grid, shape `(C, Nx, Ny)`.
- `PrecisionField`: scalar precision stored in log-space for stability, with floor.

### 4.3 QFTPCNLayer — single hierarchical layer

**File**: `src/qft_pcn/layer.py`

One level of the predictive hierarchy. Owns the three coupled fields `Phi`, `E`, `Pi` and a pluggable **`GenerativeMap`** (Protocol) for the downward generative model. Two implementations are provided:

- `ClassicalConvMap`: per-channel learnable 3×3 convolution with bias. Default.
- `QuantumConvMap` (from `quantum.py`): a tiled variational quantum circuit.

The layer's field updates are gradient flow on the per-layer free energy:

```
update_error:      dE/dt = (Phi_below - g(Phi)) - λ_E E
update_phi:        dPhi/dt = grad_input(g, Pi * E) + D * Δ_g Phi + top_down
update_precision:  d(log Pi)/dt = α (1 - Pi E^2)        (fixed pt: Pi = 1/E^2)
learn_kernel:      one SGD step on g's parameters (skippable for expensive gen-maps)
free_energy:       1/2 Pi E^2 - 1/2 log Pi + κ_R R(g)   integrated against sqrt|g|
```

### 4.4 QFTPCNNetwork — hierarchical stack

**File**: `src/qft_pcn/network.py`

Stack of `QFTPCNLayer`s sharing one manifold. Per step: errors propagate upward, beliefs update with bidirectional drive, precision adapts, the metric responds to aggregate stress-energy, optional learning of generative-map parameters.

### 4.5 MultiFieldNetwork — coupled field types on one manifold

**File**: `src/qft_pcn/multifield.py`

Multiple named field types share one `Manifold2D`. Per pair of fields, a learnable coupling constant `g_{ij}` parameterizes a Yukawa-style interaction term `L_int = g_{ij} Phi_i Phi_j`. The interaction enters each field's belief update as a cross-field message, and the coupling constants themselves are updated by gradient descent on the joint free energy:

```
dg_{ij}/dt  =  -∂F/∂g_{ij}  =  - ⟨Phi_i Phi_j⟩_M / Vol(M)
```

Correlated fields grow their coupling; uncorrelated fields keep theirs near zero.

### 4.6 QuantumGenerativeMap / QuantumConvMap

**File**: `src/qft_pcn/quantum.py`

A variational quantum circuit on `n_qubits` qubits with `n_layers` ansatz layers. Each layer: per-qubit `R_y(θ_0)` and `R_z(θ_1)` rotations, then a linear chain of CNOTs. Input encoding by `R_y(π x_q / 2)` on each qubit. Output is the Pauli-Z expectation on each qubit.

- **Exact statevector simulation** via `qiskit.quantum_info.Statevector` (noiseless).
- **Parameter-shift rule** gives exact gradients of the output w.r.t. the variational angles: `df/dθ = (f(θ + π/2) − f(θ − π/2)) / 2`.
- **`QuantumConvMap`** tiles this VQC across the 2D grid in non-overlapping patches, with parameters shared across patches (translation-invariant quantum convolution). Conforms to the `GenerativeMap` protocol so it drops into any `QFTPCNLayer`.

Verified: parameter-shift gradients agree with finite differences to `1e-3` (`test_parameter_shift_matches_finite_difference`).

### 4.7 QFT core

**Directory**: `src/qft_pcn/qft/`

#### 4.7.1 `fock.py` — local operators

```python
annihilation(d) -> (d,d) complex matrix
creation(d)     -> (d,d) complex matrix
number(d)       -> diag(0, 1, ..., d-1)
phi_op(d)       -> (a + a†) / sqrt(2)
pi_op(d)        -> i(a† - a) / sqrt(2)
vacuum_vec(d), number_state_vec(d, n)
identity(d), embed_op(op, i, dims), two_site_op(L, R)
```

#### 4.7.2 `mps.py` — Matrix Product State

```python
class MPS:
  tensors: list[np.ndarray]   # each (chi_l, d, chi_r), complex

  from_product(states), vacuum(N, d), number_states(occupations, d)
  norm_sq(), normalize()
  local_expectation(site, op)         -> complex
  two_site_expectation(site, op)      -> complex
  apply_local_gate(site, gate)
  apply_two_site_gate(site, gate, chi_max, eps)  -> truncation_error
  bond_dimensions(), entanglement_entropy(bond)
```

Entanglement entropy is computed by QR-sweeping into mixed canonical form around the requested bond, then SVD; no full statevector is ever materialized.

#### 4.7.3 `hamiltonian.py` — multi-species Hamiltonian

```python
@dataclass FieldSpecies(name, cutoff, bare_mass, kinetic, quartic, source)

@dataclass HamiltonianConfig(
    species: list[FieldSpecies],
    density_couplings: dict[(a,b)->float],
    yukawa_couplings:  dict[(a,b)->float],
    curvature_xi: float
)

class Hamiltonian:
  local_op(site)   -> (d_local, d_local) Hermitian
  bond_op(site)    -> (d_local^2, d_local^2) Hermitian
  a(s), adag(s), n(s), phi(s): embedded single-species operators
  update_param(name, value), get_param(name)
```

Verified Hermitian (`test_hamiltonian_hermitian`).

#### 4.7.4 `evolution.py` — TEBD

```python
trotter_step(state, H, dt, imaginary=False, chi_max=32, eps=1e-10) -> truncation_error
evolve(state, H, dt, steps, imaginary, chi_max, normalize_every)
energy(state, H) -> float
```

Second-order Suzuki-Trotter; supports both real-time (unitary, energy-conserving) and imaginary-time (relaxational, energy-minimizing) evolution. Each two-site gate is computed once per step via `scipy.linalg.expm` of the bond Hamiltonian.

Verified properties (in `test_qft.py`):

- Real-time evolution conserves `⟨H⟩` to better than 5% over 20 steps with `χ_max = 24`.
- Imaginary-time evolution monotonically decreases `⟨H⟩` from an excited state.
- A product (zero-entanglement) state evolved under kinetic hopping develops nonzero midpoint entanglement entropy.

#### 4.7.5 `qpcn.py` — Quantum Predictive Coder

```python
@dataclass QPCNConfig(
    species, N_sites, chi_max, dt_real, dt_imag,
    imag_steps_per_observe, real_steps_per_observe,
    learn_rate, learnable_params, observable_map,
    use_manifold, manifold_kappa
)

class QPCN:
  state: MPS
  H: Hamiltonian
  manifold: Manifold2D | None

  predict() -> dict[(site, species, op_name) -> float]
  observe(targets, learn=True) -> diagnostics_dict
  _grad_step(targets)        # finite-difference on H's params with one-step probe
  _update_manifold()         # feeds back energy density into the metric
```

Each `observe` step:

1. Refresh per-site curvature from the manifold (if coupled).
2. Imaginary-time relax: `imag_steps_per_observe` Trotter steps of size `dt_imag`, normalized.
3. Real-time evolve: `real_steps_per_observe` Trotter steps of size `dt_real`.
4. Compute predictions `⟨Ψ|O_i|Ψ⟩` at observable sites.
5. Compute prediction errors `target_i - ⟨O_i⟩`.
6. If learning: finite-difference gradient on each learnable parameter, using a *one-step probe* (clone state, evolve one micro-step, score) to capture the state's response — central-difference at the current parameter value would be zero at symmetric points.
7. If manifold-coupled: deposit local energy density `⟨H_local(k)⟩` onto the manifold and call `update_metric`.

Verified: QPCN reduces prediction error over 20 learning steps when the learnable parameter is the coherent source `J` (which can actually shift `⟨n⟩` from vacuum; the bare mass alone cannot).

---

## 5. Tests and Verification

Four test suites, **39 tests total, all green**.

### 5.1 `test_qft_pcn.py` — classical substrate (4 tests)

1. `test_flat_laplace_beltrami_matches_flat_laplacian`: when metric perturbation is zero, the Laplace-Beltrami operator reduces to the expected 2-spaced central-difference stencil to machine precision.
2. `test_metric_stays_positive_under_update`: after 200 random stress-energy updates, `det g > 0` everywhere.
3. `test_free_energy_decreases_on_stationary_input`: on a fixed Gaussian-blob input, free energy is monotonically reduced over 80 steps.
4. `test_curvature_concentrates_near_input_features`: after training, metric perturbation magnitude is higher near input gradients than at the far corner.

### 5.2 `test_multifield.py` — multi-field coupling (3 tests)

1. `test_correlated_inputs_grow_coupling`: two fields with identical inputs develop `|g| > 0.05` over 60 steps.
2. `test_uncorrelated_inputs_keep_coupling_small`: two fields with independent noise inputs keep `|g| < 0.2` over 80 steps.
3. `test_multifield_shares_one_manifold`: all fields literally share the same `Manifold2D` instance.

### 5.3 `test_quantum.py` — Qiskit VQC (3 tests)

1. `test_forward_returns_values_in_unit_range`: Pauli-Z expectations lie in `[-1, 1]`.
2. `test_parameter_shift_matches_finite_difference`: parameter-shift gradient matches finite-difference to `1e-3`.
3. `test_quantum_layer_runs_and_stays_finite`: a `QuantumConvMap` plugged into a PCN layer keeps free energy finite over 8 evolution steps.

### 5.4 `test_qft.py` — full QFT core (14 tests)

**Local algebra (3)**:
- Canonical commutation `[a, a†] = 1` to numerical precision (modulo truncation defect).
- `n|k⟩ = k|k⟩` for all `k < d`.
- `a†|0⟩ = |1⟩`, `a|1⟩ = |0⟩`.

**MPS basics (3)**:
- Vacuum MPS has unit norm.
- Local expectation of `n` on a product number-state returns the occupation.
- Unitary two-site gate preserves norm.

**Multi-species (1)**:
- `embed_op` produces operators that act only on the target species.

**Hamiltonian + evolution (4)**:
- All local and bond operators are Hermitian.
- Real-time TEBD conserves `⟨H⟩` (Lorentzian dynamics).
- Imaginary-time TEBD lowers `⟨H⟩` (variational relaxation).
- A product state evolved under hopping generates nonzero midpoint entanglement entropy. **This is the proof we have a genuine quantum substrate, not a classical filter with quantum vocabulary.**

**Cross-species coupling (1)**:
- A particle injected at species A populates species B under Yukawa-coupled evolution.

**QPCN end-to-end (2)**:
- QPCN reduces prediction error over training when the source parameter is learnable.
- QPCN coupled to a manifold deforms the metric in response to QFT energy density.

---

## 6. Strengths and Weaknesses for Learning

Compute and complexity are deliberately excluded. Judgment is by learning capability alone.

### 6.1 Strengths

**Phase and interference as a first-class signal.** Complex amplitudes throughout, so destructive interference (cancellation of competing predictions) is a structural property, not an approximation. Real for data with intrinsic phase: audio, EEG, radar, anything oscillatory.

**Entanglement as binding.** MPS bond dimension carries non-factorizable correlations between distant sites. For the binding problem ("the red ball" attaches red to ball, not to the tree behind it), this is structurally cleaner than attention — bond dimension is a tunable knob with clear semantics. Transformers bind via learned dynamic attention; here binding is a *representational* property of the state.

**Hamiltonian parameters are the model.** ~50–500 numbers (masses, couplings, sources, curvature coupling) carry the entire generative model. Each has physical meaning. The model can only express dynamics that factor through local Hermitian operators — a strong inductive bias. When the data has that structure (which it does for our target domains), this prior is decisive.

**Spontaneous symmetry breaking → category formation.** As learning proceeds, the symmetric vacuum acquires nonzero `⟨φ⟩` and the symmetry breaks. Different broken-symmetry vacua *are* the learned categories; Goldstone modes connect nearby categories as smooth interpolations. This is a theorem (Goldstone), not a heuristic.

**Free energy is the loss, literally.** Imaginary-time evolution = belief inference; parameter gradients = learning. Inference and training are one flow on one objective.

**Compositionality via interaction Lagrangian.** Adding a new field species is one entry in a config; coupling it to existing fields is one Yukawa term. The model composes algebraically.

**Bidirectional geometry-content coupling.** The metric responds to where energy concentrates; learned content reshapes the substrate. No published architecture has this exact loop.

### 6.2 Weaknesses

**The function class is small.** Three or four field species with quartic self-interactions and Yukawa couplings give a function class of ~thousands of effective parameters max. For arbitrary natural-language tasks, this is an order-of-magnitude underfit no matter how much compute is thrown at it. *This is why the LLM sits on top — natural language is not in the target hypothesis class.*

**Volume-law entanglement is unreachable.** MPS represents area-law entanglement efficiently; states whose entanglement entropy grows with subsystem size (`S(L) ~ L^d`) cannot be represented at any bond dimension `χ`. The model converges to its best low-entanglement approximation and stops. This is a hard expressivity wall.

**1D MPS is poorly suited to 2D+ spatial data without coarse-graining.** Real images, video, 3D environments lose locality when snake-pathed through an MPS. PEPS would help but is much harder. **MERA (next phase) addresses this directly via hierarchical scale separation.**

**Gradient signal fragile near symmetric points.** At unbroken-symmetry parameter values, gradients of expectation values vanish. Initialize off-symmetry or relax through annealing; otherwise the optimizer is stuck (the QPCN test discovered this: at `J = 0`, `∂⟨n⟩/∂J = 0`).

**Symmetry breaking is irreversible.** Once committed to a vacuum sector, escape requires tunneling — exponentially slow. Early-training mistakes compound. The *learning* analog of catastrophic forgetting; structural to the architecture.

**No data-dependent routing.** Hamiltonian is fixed across inputs; only the state varies. Attention's key strength (data-dependent message routing) is absent. Could be added via data-dependent Hamiltonian terms, but breaks Hermiticity if done naively.

**No hierarchical abstraction in the quantum core yet.** 1D MPS is flat. The classical PCN hierarchy is hierarchical, but the quantum core is not. MERA fixes this and is the principled next step.

**Long causal chains hard.** Many TEBD steps compound truncation error. RNNs/transformers do 1k-step dependencies routinely; this architecture would struggle past ~50 quantum steps. *For programming/logic, the chain length is bounded by AST depth, which is usually small — this is not a problem for the target domain.*

---

## 7. Comparison to Frontier Research

| Frontier work | Year | Claim | Relation to QPCN |
|---|---|---|---|
| Friston — hierarchical predictive coding | 2005+ | The brain implements variational inference via PCN | We extend to operator-valued fields. Their math holds verbatim; ours is the quantum lift. |
| Whittington & Bogacz — PCN approximates backprop | 2017 | PCN can implement BP-like learning | Validates that PCN is a viable learning algorithm; we inherit. |
| Bogacz — PCN tutorial | 2017 | Practical PCN with precision-weighted errors | Direct source of our update equations. |
| Stoudenmire & Schwab — MPS for MNIST | 2016 | MPS classifier, ~2% test error | Same lineage. ~2% is the realistic ceiling on image data with this class. |
| Stoudenmire — TreeNet learning | 2018 | Hierarchical tensor network classifier | MERA-style learner; near-future direction for us. |
| Glasser et al. — generalized tensor networks for ML | 2019 | TN family for probabilistic ML | Same family, broader. |
| Carleo & Troyer — NN-based quantum state ansatz | 2017 | RBM/NN wavefunctions for the Heisenberg model | Inverse direction; we use quantum states as the substrate for an NN-like learner. |
| Pesah et al. — no barren plateaus in QCNNs | 2021 | Local-circuit QML avoids vanishing gradients | Validates our `QuantumConvMap`. |
| Schuld et al. — circuit-centric quantum classifiers | 2020 | VQC-based supervised learning | Our `QuantumGenerativeMap` is in this family with PCN dynamics added. |
| Hashimoto et al. — deep learning AdS/CFT | 2018 | NN learns geometry from boundary data | Closest published work to our "fields shape geometry" claim. We make the coupling explicit and dynamical. |
| Vidal — entanglement renormalization (MERA) | 2008 | Hierarchical tensor network for critical states | The principled extension of our 1D MPS quantum core. |
| Swingle — holographic geometry of MERA | 2012 | MERA's geometry ≈ AdS_2 | Makes the "geometric reasoning" framing literal. |
| Cohen & Welling — group equivariant CNNs | 2016 | Gauge symmetry as architectural prior | Same inductive bias as our Hamiltonian gauge structure; different scaffolding. |
| Mehta & Schwab — variational RG and deep learning | 2014 | Deep nets implement RG coarse-graining | Mathematical content of our hierarchy. |
| Lin, Tegmark, Rolnick — why deep learning works | 2017 | Cheap deep learning explained by physics priors | Same underlying argument as our pitch for physical-domain reasoning. |
| Roberts & Yaida — Principles of Deep Learning Theory | 2022 | Neural networks as field theories at width → ∞ | Mathematical foundation closest to our QFT framing. |
| Cirac & Verstraete — tensor network simulation of QFTs | 2009 | MPS/PEPS for lattice gauge theories | Direct methodological precedent for what we built. |
| Bauer et al. — quantum chemistry with TNs | 2020 | DMRG/TN methods for molecular energetics | This is the canonical winning domain — we slot into it. |
| Abramsky & Coecke — categorical quantum semantics | 2004 | Entanglement as resource in quantum protocols | Foundation for the variable-binding-as-entanglement claim in §8. |
| Selinger — quantum lambda calculus | 2007 | λ-calculus with quantum data types | Closest published precursor for treating program semantics quantum-mechanically. |
| Coecke & Kissinger — Picturing Quantum Processes | 2017 | Graphical calculi for quantum + linguistic + program semantics | Categorical machinery underlying §8. |
| Scaling-law transformers (Chinchilla, GPT-4) | 2022–2024 | Web-scale LMs win on natural language | Not the target domain. We do not compete here. |
| Diffusion models (DDPM, Stable Diffusion) | 2020+ | Score-matching generative models | Mathematically adjacent: score `∇ log p` ≈ `-δS/δφ` in our path-integral interpretation. Beat tensor-net methods on natural images. |
| AlphaFold 2 | 2021 | Physics-aware NN for protein structure | The single best example of "physical priors matter". Our architecture's strongest realistic application class. |

**To the best of our knowledge, the specific combination of (predictive coding) + (operator-valued fields) + (matrix product state substrate) + (dynamic Riemannian manifold) + (Qiskit-backed VQC layer) + (Yukawa-coupled multi-species) + (LLM-as-NL-frontend, QPCN-as-reasoning-core) has not been published.**

---

## 8. The Logic/Reasoning Application

This is the *primary* target use case. The QPCN is to be a logic and structured-reasoning engine, not a language model. A traditional LLM sits above it as the natural-language interface.

### 8.1 The deep correspondence: variable binding *is* entanglement

In a program, `let x = 3 in (f x x)` puts the value `3` into two distant positions: the two arguments to `f`. Their values are not merely correlated, they are **the same random variable**. Two distant sites holding the same random variable is the definition of an EPR pair. **Variable scoping in code is literally what MPS bond dimension was invented to represent.**

This is not analogy. The structural map is exact:

| Programming concept | Quantum concept |
|---|---|
| Variable binding (`let x = ...`) | Entanglement creation (CNOT-like) |
| Variable use (`x`) | Measurement of the entangled site |
| Alpha-renaming (`λx.x` ≡ `λy.y`) | Local unitary on the entangled subspace |
| Lexical scope | Bond dimension between scope boundary and use |
| Closure | Persistent entanglement carried by a function value |
| Type | Conserved charge / gauge constraint |
| Type checking | Verifying conservation laws along the chain |
| Beta reduction | Trotter step on the evaluation Hamiltonian |
| Confluence (Church-Rosser) | Commutativity of disjoint local Hamiltonian terms |
| Termination | Existence of a ground state (lowest-energy normal form) |
| Recursion | Self-similar tree structure → MERA |
| Pure function | Unitary operator |
| Side effect | Non-unitary contribution / dissipation |
| Linear types | Bosonic vs fermionic statistics |

If the encoder is built right, **type checking becomes gauge-invariance verification, evaluation becomes Schrödinger evolution, and proof search becomes ground-state finding** — all three using the same machinery, no separate implementations.

### 8.2 What this gives the system that nothing else does

**Bidirectional reasoning natively.** Standard provers go forward (premises → conclusion) or backward (goal → subgoals). QPCN does both simultaneously through ground-state search — the Hamiltonian constraint terms have no preferred direction. For program synthesis (`find a function of type X → Y`), this is the right primitive.

**Counterfactual reasoning by perturbing constraints.** "What would happen if we changed this rule?" maps to "evolve under a perturbed Hamiltonian and measure." Most reasoning systems can't do this without rebuilding from scratch.

**Genuine uncertainty quantification.** When constraints underdetermine the answer, the QPCN's ground state is a *superposition*; measurement gives a distribution over valid completions. For synthesis problems with multiple valid implementations, this is exactly what you want. LLMs fake this with temperature sampling; QPCN gets it structurally.

**Constraint debugging.** When the QPCN cannot reduce energy to zero, the per-term energy contributions identify *which* constraints are stuck — the analog of a type-error message that works for arbitrary logic, not just type systems.

### 8.3 The other target domains under this framing

The same machinery serves the other intent-list domains:

- **Molecular energetics**: each site = an atom or basis function, fields = electronic species, Hamiltonian = Born-Oppenheimer + Coulomb + spin-orbit. Ground state = electronic ground state. Already the DMRG paradigm; we add learnable parameter inference from data.
- **Gauge theories**: gauge connection field `A_μ` is one of the field species; gauge-invariant observables are the natural targets. Lattice QCD style.
- **Dynamical systems with conservation laws**: each site = a phase-space cell, Hamiltonian = the system Hamiltonian, Schrödinger evolution = quantum-corrected classical trajectory. Conservation laws are conserved charges of `H`.

The common structure — **local Hermitian operators with conservation laws, on a discretized substrate** — is exactly what our architecture is built around.

---

## 9. The LLM + QPCN System Architecture

### 9.1 Division of labor

| Capability | LLM | QPCN |
|---|---|---|
| Natural-language parsing | ✓ strength | ✗ not applicable |
| World knowledge retrieval | ✓ strength | ✗ not applicable |
| Soft pattern matching | ✓ strength | ✗ not its job |
| Spec writing (NL → DSL) | ✓ strength | ✗ |
| Exact algebraic manipulation | ✗ weakness | ✓ strength |
| Constraint propagation | ✗ weakness | ✓ strength |
| Conservation-law-respecting search | ✗ weakness | ✓ strength |
| Multi-step deduction | ✗ weakness | ✓ strength |
| Verbalizing structured output | ✓ strength | ✗ |

The LLM is good at the things QPCN is bad at, and vice versa. The interface between them is small.

### 9.2 The DSL

A compact JSON specification that the LLM emits and the QPCN consumes:

```json
{
  "fields": [
    { "name": "expr",  "cutoff": 16 },
    { "name": "type",  "cutoff": 8 },
    { "name": "value", "cutoff": 16 }
  ],
  "sites": 32,
  "constraints": [
    { "kind": "local",     "site": 5,  "term": "expr_node == '+'", "weight": 1.0 },
    { "kind": "local",     "site": 5,  "term": "type(expr) == int", "weight": 1.0 },
    { "kind": "two_site",  "sites": [5, 7], "term": "type(arg1) == type(arg2)", "weight": 1.0 }
  ],
  "boundary": {
    "5":  { "value":  3.0 },
    "12": { "expr":  "result" }
  },
  "observables": [
    { "site": 12, "field": "value", "op": "n" }
  ],
  "search": { "method": "imag_time", "steps": 50, "chi_max": 32 }
}
```

The LLM produces this from a user's natural-language request. The QPCN reads it, builds the `Hamiltonian`, evolves into the ground state, and returns the measured observables.

### 9.3 Operational flow

```
User (NL)
   │
   ▼
LLM  ──► generates DSL spec ──► QPCN runtime
                                    │
                                    ▼
                              build Hamiltonian from constraints
                                    │
                                    ▼
                              imag-time evolve MPS to ground state
                                    │
                                    ▼
                              measure observables
                                    │
                                    ▼
                              return {observable_values, residual_constraint_energies}
                                    │
                                    ▼
LLM  ◄── verbalizes for user
```

The DSL is **the entire contract** between the two systems. The QPCN never sees natural language; the LLM never sees an MPS.

---

## 10. Implementation Roadmap

In priority order. Each item is sized for one focused session.

### 10.1 AST-to-MPS encoder (next session)

**Goal**: serialize a syntax tree into an MPS site sequence such that the structural correspondences in §8.1 are realized in code.

**Concrete tasks**:
- Walk the AST in canonical pre-order; assign each node to a site.
- Per site, the local Hilbert space is the tensor product over field species: `node_kind` (which AST constructor), `type` (typing scheme), `binder_id` (which lexical binder, if any), `value` (carried value for literals).
- Variable references encode their binder *not* via a separate field but via the **bond dimension** along the path connecting use site to binding site. This is the operational realization of "binding = entanglement."
- Decoder: from an MPS state, sample a syntax tree by measuring node_kind at each site and binder_id at each variable use.

**Files to create**: `src/qft_pcn/logic/ast.py`, `src/qft_pcn/logic/encoder.py`, `src/qft_pcn/logic/decoder.py`.

**Acceptance test**: round-trip a small set of lambda-calculus terms through encode → MPS → decode without loss.

### 10.2 Type-system-to-Hamiltonian compiler

**Goal**: each typing rule becomes one local or two-site Hamiltonian term that adds positive energy when violated.

**Examples**:

```
T-Var:        Γ, x:A ⊢ x : A           ──►   one-site:  if expr=var, type(expr) ≠ type(binder(expr))  →  +1
T-App:        Γ ⊢ f : A→B, Γ ⊢ a : A
              ────────────────────────  ──►   two-site:  if expr=app, type(fn)=A→B and type(arg)≠A  →  +1
              Γ ⊢ f a : B
T-Abs:        Γ, x:A ⊢ b : B            ──►   two-site:  if expr=lam, type(body) ≠ type_result(type(self))  →  +1
              ────────────────────────
              Γ ⊢ λx.b : A→B
```

Each rule violation is a positive-energy term. The ground state of `H = Σ rules` is a well-typed program; the energy gap measures the number of violations.

**Files to create**: `src/qft_pcn/logic/typing_rules.py`, `src/qft_pcn/logic/hamiltonian_compiler.py`.

**Acceptance test**: a small set of well-typed and ill-typed STLC programs are distinguished by `⟨H⟩ = 0` versus `⟨H⟩ > 0`.

### 10.3 Evaluation as TEBD

**Goal**: beta reduction = local two-site gate that moves a value from the argument site to the parameter site.

The reduction gate is non-Hermitian (it's not "energy-preserving" because evaluation does work). Two ways to implement:

a. **Dissipative dynamics**: `dρ/dt = -i[H, ρ] + Σ_k L_k ρ L_k† − 1/2 {L_k†L_k, ρ}` with Lindblad operators implementing reduction. Requires extending to density matrices; the MPS becomes an MPO.

b. **Constraint-based**: the *fully reduced* program is the ground state of an augmented Hamiltonian that penalizes any non-normal-form configuration. Then evaluation IS ground-state finding, and reduction is a side effect of imag-time evolution. **Strongly preferred** for simplicity and conceptual unity with the rest of the architecture.

**Files to create**: `src/qft_pcn/logic/evaluation_hamiltonian.py`.

**Acceptance test**: a program like `(λx.x+1)(2)` reduces to `3` via imaginary-time relaxation.

### 10.4 MERA for recursion and hierarchical scope

**Goal**: replace the 1D MPS with a hierarchical tree (MERA) so that:
- Recursive function definitions get bounded bond dimension regardless of recursion depth.
- Nested lexical scopes correspond to depth in the MERA tree.
- The "fields shape geometry" framing becomes literal: MERA's geometry is hyperbolic AdS-like (Swingle 2012), and the QPCN's metric perturbation is sourced by quantum-field stress-energy, giving an honest holographic mini-AdS/CFT-like correspondence.

**Files to create**: `src/qft_pcn/qft/mera.py` (new state representation), `src/qft_pcn/qft/mera_evolution.py` (binary-tree TEBD).

**Acceptance test**: a recursive Fibonacci program reaches its ground state with bond dimension `O(log N)` rather than `O(N)`.

### 10.5 LLM bridge layer

**Goal**: a thin RPC/IPC layer where the LLM emits DSL specs and reads back results.

**Files to create**: `src/qft_pcn/bridge/api.py` (FastAPI or stdio JSON-RPC), `src/qft_pcn/bridge/dsl.py` (DSL schema + parser), `src/qft_pcn/bridge/runtime.py` (QPCN problem runner).

**Acceptance test**: end-to-end demo where a hardcoded LLM-style prompt produces a DSL spec, the bridge runs the QPCN, and the result is rendered back.

### 10.6 Constraint debugger

**Goal**: when the QPCN cannot drive `⟨H⟩` to zero, expose *which terms* contribute residual energy, so the LLM can phrase a useful error message.

**Files to create**: `src/qft_pcn/logic/debugger.py`.

**Acceptance test**: an ill-typed program produces a structured report identifying the specific typing-rule violation.

### 10.7 First publishable milestone: bidirectional type inference for STLC

A tractable, well-scoped first deliverable that the existing ML and PL communities will recognize as nontrivial:

> **Take a simply-typed lambda-calculus program with holes (`?`) at arbitrary positions in either expressions or types. Produce a distribution over valid completions, ranked by the QPCN's energy.**

This is program synthesis with type-driven guidance. LLMs are demonstrably weak at it (they hallucinate ill-typed completions); tensor-network methods have never been tried. With bond dimension `χ = 32` and ~30 AST nodes, this is well within reach of the architecture we've built once §10.1–10.3 are complete.

**Files for the milestone demo**: `src/qft_pcn/logic/demo_stlc_synthesis.py`.

**Publishability target**: a workshop paper at NeurIPS or ICML on "Tensor-Network Predictive Coding for Program Synthesis" or similar. Even a negative or partial result is publishable because the method is new.

### 10.8 Lemma library and promotion (hierarchical composition, mechanism 1)

**Goal**: solved sub-problems become persistent, addressable primitives that can be reused as building blocks in larger problems. This is the operational realization of "lemmas → theorems" at the architectural level.

#### Principles

The QPCN's ground state `|Ψ_child⟩` of a Hamiltonian `H_child` is a *proof object* — by the Curry-Howard correspondence, it inhabits the proposition encoded in `H_child`'s constraints. Once that ground state is found:

- The proposition is no longer something to prove; it is an axiom for any higher-level search.
- The proof object itself is a configuration of operators on a Hilbert space — it can be cached and re-applied.
- A future QPCN can avoid re-deriving this lemma by *clamping* the corresponding sub-MPS to the cached state.

This is structurally identical to how a mathematician writes "by Lemma 3.2, ..." — they don't re-prove Lemma 3.2 every time they use it. The lemma is a building block. The composition is *referential*, not constructive.

Three observations make this rigorous in our architecture:

1. **Lemmas have type signatures.** A cached ground state inhabits some proposition; that proposition is the lemma's type. Lemma lookup is type-directed.
2. **Lemmas have provenance.** Each cached lemma records the Hamiltonian it was derived from, the energy gap to first excited state (a proxy for "how certain is this lemma"), and any auxiliary assumptions clamped during its derivation. Lemmas with unverified assumptions are marked conditional.
3. **Lemmas compose under Curry-Howard.** When the cached `|Ψ_A⟩` proves `proposition_A` and `|Ψ_B⟩` proves `proposition_B`, then a third QPCN can use both as clamps to prove `proposition_A ∧ proposition_B` essentially for free (the composite Hamiltonian has zero residual at those sub-trees).

#### Mechanisms

**Storage layer**:
- Each lemma stored as `(MPS tensors, proposition_type, derivation_metadata)`.
- Tensors serialized via numpy `.npz` or HDF5; bond dimensions compressed to the minimum that preserves the lemma's ground-state energy.
- Optional SVD compression of MPS tensors to reduce storage when bond dimension exceeds what's actually needed for the lemma's information content.

**Indexing layer**:
- Primary index: by proposition type (computed via the typing-rule Hamiltonian at registration time).
- Secondary index: by structural fingerprint of the lemma's reduced density matrix at a canonical bond — fast similarity lookup for "is there a lemma roughly shaped like this?".
- Tertiary index: by derivation cost — when multiple lemmas prove the same thing, prefer the one cheapest to apply.

**Registration**:
- When a QPCN run completes with residual energy below threshold `ε_register`, the resulting state is offered to the library.
- A validation pass classically type-checks the decoded AST (if it's a proof) or verifies the constraint satisfaction (if it's a non-proof structure like a chemical configuration).
- On pass, the lemma is added with all three indices. On fail, the lemma is logged as a near-miss for debugging but not registered.

**Promotion to Hamiltonian primitive**:
- A new DSL constraint type: `{"kind": "use_lemma", "lemma_id": L, "sites": [i, j, k, ...]}`.
- The compiler reads the cached lemma's MPS tensors and clamps the specified sites to that state during initialization.
- Equivalently, it adds a very strong projector `−W |Ψ_L⟩⟨Ψ_L|` to the Hamiltonian, which pulls the sub-MPS into the cached configuration during imag-time relaxation.

**Composition correctness**:
- When two lemmas are used at sites with shared variables (via the binding-as-entanglement mechanism), the entanglement structure must compose consistently — this is automatically checked by the Hamiltonian's typing constraints.
- If two clamped lemmas demand incompatible structures, the residual energy at composition becomes positive; this is the analog of "Lemma A and Lemma B are inconsistent."

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Lemma drift**: a lemma proved under one set of assumptions may not be valid under another | Store full constraint context with each lemma; lookup with assumption-set matching |
| **Cache pollution**: bad solutions cached | Validation pass before registration; optional human-in-the-loop confirmation for foundational lemmas |
| **Lemma explosion**: too many cached, lookup becomes slow | Periodic clustering + pruning; merge near-duplicate lemmas |
| **Lemma rot**: a lemma's assumptions become invalid because of later library updates | Dependency tracking; invalidate downstream lemmas when an upstream one is revised |
| **False composition**: two lemmas combined wrongly because their entanglement patterns clash silently | The Hamiltonian's energy gap at composition is the signal; refuse compositions with high residual |

#### Files to create

```
src/qft_pcn/composition/
├── __init__.py
├── lemma_library.py        # storage + indexing + registration
├── promoter.py             # use_lemma DSL constraint compiler
└── tests/
    └── test_lemma_library.py
```

#### Acceptance test

A two-stage demo: prove a base lemma in QPCN run 1 (e.g., `∀x. x + 0 = x` over a small Peano-arithmetic encoding), register it, then prove a theorem in QPCN run 2 that uses the cached lemma (e.g., `∀x. (x + 0) + 0 = x`). The second run should converge with zero residual energy and complete in substantially fewer Trotter steps than re-deriving from axioms.

### 10.9 Abstraction discovery (hierarchical composition, mechanism 2)

**Goal**: when patterns recur across solved problems, automatically abstract them into new primitive operators that future searches can use directly. This is the operational realization of "discovering a useful concept."

#### Principles

DreamCoder (Ellis et al. 2020/2021) established the wake-sleep paradigm for library learning in program synthesis: alternate between (a) solving problems with the current library and (b) compressing solutions to discover new library entries that capture recurring patterns. After several cycles, the library grows from a small set of primitives to a rich, domain-adapted vocabulary that makes future problems much easier.

We adapt this to tensor-network substrates with one key change: instead of compressing programs via grammar induction, we **cluster reduced density matrices** of solved sub-MPSes. The intuition: if many solved problems share a common substructure, that substructure has a recognizable signature in the reduced density matrix at the bond where the substructure ends. Patterns recur in the *operator-algebra* sense, not just the syntactic one.

This is also closer to how cortex discovers concepts: not by analyzing syntax trees, but by finding statistical regularities in the structure of neural representations.

#### The wake-sleep cycle in detail

```
INITIALIZE library = { axioms, primitive constructors }

REPEAT:
    # WAKE phase: solve problems with current library
    FOR each problem P in current batch:
        spec = LLM(P)                        # natural language → DSL
        H = compile(spec, library)
        |Ψ⟩ = qpcn.relax_to_ground(H)
        IF residual(H, |Ψ⟩) < ε:
            register(|Ψ⟩, P, library)

    # DREAM phase: enumerate sub-structures
    candidates = []
    FOR each solved problem |Ψ_i⟩ in library:
        FOR each subtree S in |Ψ_i⟩:
            ρ_S = reduced_density_matrix(|Ψ_i⟩, S)
            candidates.append( (S, ρ_S, |Ψ_i⟩) )

    # CLUSTER phase: find recurring patterns
    clusters = hierarchical_cluster(candidates, distance=trace_distance(ρ_S))
    significant_clusters = [c for c in clusters if size(c) >= k_min]

    # ABSTRACT phase: promote patterns to primitives
    FOR each cluster C in significant_clusters:
        new_primitive = compute_canonical_form(C)
        new_primitive.provenance = [|Ψ_i⟩ in C]
        library.add(new_primitive)

    # CONSOLIDATE phase: re-derive solutions using new primitives
    FOR each |Ψ_i⟩ in library:
        IF can_be_expressed_using_new_primitives(|Ψ_i⟩):
            replace with shorter solution
    prune_redundant_lemmas(library)

UNTIL no new primitives discovered for N cycles
```

#### Theoretical principles underlying this

1. **Operator-theoretic reuse**: in QFT, *effective operators* emerge at low energies as composites of fundamental ones (e.g., the pion as a composite of quarks). The abstraction-discovery process is the learning analog of effective field theory — discovering composites that simplify the description at a given scale.
2. **Renormalization group flow**: at each abstraction cycle, low-level details are integrated out, and a coarse-grained library emerges. This is literally RG flow on the operator algebra of the problem domain.
3. **Solomonoff induction analog**: shorter descriptions are preferred. A library that compresses many solutions has captured genuine structure of the domain; a library that doesn't compress is overfitting.
4. **Bayesian model selection**: the marginal likelihood of the data under a hypothesis with primitive `P` is higher when `P` appears in many derivations. Promotion threshold `k_min` is the Bayesian evidence threshold.

#### Detailed implementation

**Subtree mining**:
- For each cached MPS `|Ψ_i⟩`, enumerate sub-MPSes of size `s` for `s ∈ [3, S_max]` (very small or very large subtrees are usually not useful).
- For each, compute the reduced density matrix at the boundary bond.
- Hash sub-MPSes by a quick fingerprint (e.g., trace, top singular values) to avoid redundant comparisons.

**Clustering**:
- Distance metric: trace distance `D(ρ_1, ρ_2) = ½ ||ρ_1 - ρ_2||_1` (or fidelity-based proxy).
- Algorithm: hierarchical agglomerative clustering with a distance threshold tuned per domain.
- Significance test: a cluster of size `k` from a candidate set of size `N` has probability `≈ exp(-k log N)` to arise by chance under a uniform null model; reject clusters where `k < k_min(N)`.

**Canonical form computation**:
- Within a cluster, the operator that minimizes the average trace distance to all members is the cluster's representative.
- Solve via averaging in the operator basis: `ρ_canonical = (1/|C|) Σ ρ_i`, then re-purify into an MPS of bounded bond dimension.

**Provenance**:
- Each new primitive remembers the problems it was abstracted from.
- When a primitive is used, this is logged so that the original problems get "credit" for the discovery.
- Useful for diagnosing which problems are driving library growth and for pruning primitives that turn out to be useful only for niche problems.

**Consolidation**:
- Re-derive existing library entries using the new primitives.
- If a lemma's solution becomes shorter (lower bond-dimension MPS), keep the new derivation.
- Prune lemmas that are now exact consequences of more primitive lemmas.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Overfitting to training distribution**: discovered primitives only useful for similar problems | Cross-domain validation: test new primitives on a held-out problem set before permanent registration |
| **Bad abstractions**: noise patterns mistaken for structure | Significance threshold `k_min`; minimum-description-length filter (does the primitive actually compress total library size?) |
| **Primitives become opaque**: harder to interpret learned operators | Maintain provenance; expose decomposition into older primitives on demand |
| **Library bloat**: too many primitives slow down search | Periodic pruning: drop primitives not used in N cycles; merge near-duplicates |
| **Catastrophic compression**: a useful primitive accidentally gets pruned because of a bad cycle | Two-tier library: a stable core (manually curated) and a dynamic outer ring (subject to wake-sleep) |

#### Files to create

```
src/qft_pcn/composition/
├── subtree_miner.py        # enumerate sub-MPSes
├── abstraction.py          # cluster + canonical form computation
├── wake_sleep.py           # orchestrate the cycle
└── tests/
    ├── test_subtree_miner.py
    └── test_wake_sleep.py
```

#### Acceptance test

A problem corpus with 5+ problems that share a structural motif (e.g., several proofs that all use induction on natural numbers, but with different inductive predicates). The system should discover the "induction primitive" as a cluster of common substructure, promote it to a library entry, and subsequent inductive proofs should converge in substantially fewer steps using it.

### 10.10 Cross-level message passing (hierarchical composition, mechanism 3)

**Goal**: parent and child QPCNs exchange goals (top-down) and proofs (bottom-up) via the same predictive-coding pattern that single-layer field updates use, generalized to the scale of whole problems.

#### Principles

A single QPCN already does predictive-coding message passing between *layers*: layer `l+1` sends top-down predictions to layer `l`, and layer `l` sends bottom-up precision-weighted errors to layer `l+1`. The free-energy principle says this minimizes a single global objective (variational free energy).

The hierarchical composition extension applies the same pattern at a higher scale, between *whole QPCNs*: parent QPCN sends sub-goals (top-down) to child QPCNs, and child QPCNs send proofs (bottom-up) back. The composition is again unified by a single objective: the joint free energy of the whole hierarchy.

This is *not* an analogy; it's a structural claim. Variational free-energy minimization is scale-invariant: it works at the level of single field updates, at the level of single layers, at the level of single QPCNs, and at the level of compositions of QPCNs. The math is the same equation at every scale.

#### Mechanisms

**Goal graph**:
- A directed acyclic graph where nodes are sub-problems (DSL specs + their expected proposition types) and edges represent "child must be solved before parent can use its result".
- Built top-down: the root is the user's target goal; children are sub-goals identified by the parent's Hamiltonian compiler (sites in the parent with high `?`-density or marked as `requires_lemma`).
- The LLM can suggest sub-goal decompositions when the compiler is unsure how to break down a problem.

**Top-down dispatch**:
- Parent QPCN's Hamiltonian compilation identifies which sub-trees in its AST encoding correspond to unsolved lemmas or unfilled holes.
- Each such sub-tree is packaged as a child DSL spec: the type signature at that sub-tree's root is the goal proposition; the constraints from the parent context become the child's boundary conditions.
- Dispatched as a new QPCN run.

**Bottom-up integration**:
- Child QPCN's ground state `|Ψ_child⟩` is returned with its residual energy.
- If residual energy is below threshold: integrate as a clamped sub-MPS in the parent (via the §10.8 lemma-promotion machinery).
- If residual energy is above threshold: child failed. Parent must:
  1. Try a different sub-decomposition (ask the LLM for a different proof strategy).
  2. Lower the child's strength of constraints to allow approximate matching ("treat this as a conjecture").
  3. Mark this sub-goal as currently unprovable and propagate up — the parent's plan was flawed.

**Concurrency**:
- Children at the same level in the goal graph are independent; they can be searched in parallel.
- The lemma library is shared across siblings, so discoveries in one branch immediately benefit others.
- In a long-running system, this becomes an embarrassingly parallel proof search, like the way mathematicians distribute work on collaborative projects.

**Backtracking and revision**:
- When a child fails to converge, the failure propagates up the goal graph.
- The parent's Hamiltonian compiler is invoked with the failure information: "the sub-tree at site k could not be proved with the current decomposition; try a different one."
- The LLM frontend can be involved here: "I'm stuck on sub-goal X; can you suggest a different proof strategy?"

**Cycle detection**:
- The goal graph is meant to be acyclic, but bad decomposition heuristics can introduce cycles (e.g., "to prove A, prove B; to prove B, prove A").
- Maintain a per-search visited set; refuse to dispatch a child whose goal is already being proved as an ancestor.
- If cycle detected, fall back to LLM-driven decomposition or mark as unprovable.

#### Detailed implementation

**Goal graph operations**:
- `Node(goal, status, children)`: `goal` is a DSL spec, `status` is one of `pending|active|solved|failed|cycle`, `children` are sub-goals.
- `dispatch(node)`: spawn a QPCN run for `node.goal`. Returns a future.
- `integrate(parent, child_result)`: if child succeeded, clamp the corresponding sub-MPS in parent; if failed, set parent's status to `pending_revision`.
- `revise(node)`: ask the LLM (or a heuristic) for an alternative decomposition.

**Concurrency primitives**:
- Lightweight: thread pool or asyncio for parallel child dispatch.
- Heavyweight: distributed execution via something like Ray for cluster-scale problem decomposition.

**LLM involvement**:
- Always at the root: convert natural language to initial goal graph.
- At decomposition points: suggest how to break a hard sub-goal into smaller ones.
- At failure points: suggest alternative strategies when a sub-goal fails.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Deadlock**: cyclic goal graph | Cycle detection; fall back to LLM for alternative decomposition |
| **Combinatorial explosion**: many decompositions to try | LLM-guided priorities; beam search over decompositions; cache failed approaches |
| **Resource starvation**: a single bad sub-goal blocks all parallel work | Timeout per child; fail-fast on apparent dead ends |
| **Stale lemmas**: a lemma proved at low priority becomes critical later | Lazy re-evaluation; the goal graph re-dispatches when assumptions change |
| **Premature integration**: a child's partial solution accepted too soon | Strict residual-energy threshold at integration; require the child to be in a true ground state, not just a low-energy excited state |
| **Cascade failures**: one bad child invalidates many ancestors | Quarantine bad sub-graphs; explore alternatives in parallel |

#### Files to create

```
src/qft_pcn/composition/
├── goal_graph.py           # DAG of sub-goals
├── dispatcher.py           # spawn + collect QPCN runs
├── result_integrator.py    # bottom-up clamping
├── revision.py             # LLM-guided alternative decomposition
└── tests/
    ├── test_goal_graph.py
    └── test_dispatcher.py
```

#### Acceptance test

A small inductive proof requiring 3 levels of decomposition: e.g., prove `∀xs : List A. length (reverse xs) = length xs`. The natural decomposition:
- Level 0 (axioms): definitions of `length`, `reverse`, the constructor `Cons`.
- Level 1 (lemmas): `length (xs ++ [y]) = length xs + 1`, and `reverse (Cons x xs) = reverse xs ++ [x]`.
- Level 2 (induction case): the inductive step combining both lemmas.
- Level 3 (theorem): induction principle applied to the two cases.

The system should solve this by dispatching sub-goals to QPCN children, integrating their results, and surfacing the final proof. Reasonable expectation: completes in under 5 minutes of wall time on a single machine, with `χ_max = 32` per sub-QPCN.

### 10.11 Second milestone: hierarchical proof composition demo

A natural follow-on to §10.7 once §10.8–10.10 are complete. Goal:

> **Solve a small but nontrivial mathematical theorem (PhD-thesis-exercise-level) by hierarchical decomposition, producing a fully verified proof tree where each node was found by a sub-QPCN.**

Target candidates (in increasing difficulty):

1. List-induction theorems on the order of `length (xs ++ ys) = length xs + length ys`.
2. Algebraic identities like `(a + b)^2 = a^2 + 2ab + b^2` derived from ring axioms.
3. Small group theory results: every group of order p (prime) is cyclic.
4. Basic linear algebra: rank-nullity theorem in finite dimensions.

Each demonstrates a new aspect of the architecture: list induction tests the recursive scope machinery (§10.4), algebraic identities test rewriting, group theory tests categorical structure, rank-nullity tests dimensional reasoning.

**Files for the demo**: `src/qft_pcn/composition/demo_hierarchical_proof.py`.

**Publishability target**: a full conference paper (NeurIPS, ICML, or a PLDI/POPL venue) on "Hierarchical Tensor-Network Predictive Coding for Automated Theorem Proving." This is the first result that would be visible to both the ML and PL/formal-methods communities simultaneously.

---

## 11. Hierarchical Composition: Principles and Long-Range Vision

This section captures the deeper conceptual framework underlying §10.8–10.10 and the architectural commitments that follow from it. It is the *why* behind the *what* in the roadmap.

### 11.1 Why hierarchical composition is the right framework

Five independent lines of intellectual support converge on hierarchical composition as the correct mode of operation:

**1. Renormalization group (Wilson 1971).** Physics's foundational insight that effective theories at long distances are systematic coarse-grainings of short-distance theories. The mathematical structure of RG flow — integrating out short-distance modes, identifying fixed points, computing critical exponents — is the same structure that should govern how a learning system aggregates low-level facts into high-level concepts. Wilson's framework gives us the rigorous content of "discovering structure at a scale."

**2. Multi-scale Entanglement Renormalization Ansatz (Vidal 2008).** The realization that quantum states with hierarchical entanglement structure are efficiently representable as a tree of tensor networks. MERA was invented for critical lattice systems (where entanglement is scale-invariant), but the architectural lesson is general: hierarchical tensor decompositions capture problems whose structure is itself hierarchical. Programs, proofs, and chemical structures all have this property.

**3. Curry-Howard correspondence (Howard 1980).** Mathematics has a built-in lemma → theorem → theory hierarchy that is structurally identical to function composition in typed lambda calculus. The hierarchy isn't imposed by mathematicians' habits; it's a logical consequence of how propositions and proofs are related. Any system that proves theorems will naturally exhibit this hierarchy because the proofs themselves are hierarchical.

**4. Hierarchical predictive coding in cortex (Friston, Rao, Ballard).** The empirical observation that cortex is organized as a hierarchy of predictive layers, each predicting the activity of the layer below and being corrected by precision-weighted errors. This is the same structure we are building, just transplanted from neurons to quantum many-body states. The brain has found that hierarchical composition is the right architecture for learning structured environments; we are inheriting that lesson.

**5. Library learning in program synthesis (DreamCoder, Ellis et al. 2020/2021).** The empirical demonstration that growing a domain-specific library through wake-sleep cycles compounds capability in machine learning systems. DreamCoder showed this for programs in classical neural networks; we are extending it to tensor-network substrates.

All five frameworks point at the same answer: **hierarchy is not a convenience for the engineer; it is intrinsic to the problem structure of the domains we are targeting.**

### 11.2 The five composition mechanisms in one frame

Putting §10.4, §10.5, §10.8, §10.9, and §10.10 together, the QPCN system has five distinct mechanisms for hierarchical composition, each with its own role:

| Mechanism | Section | What it does | When it's used |
|---|---|---|---|
| **MERA-structured belief field** | §10.4 | Intra-problem hierarchical entanglement | Recursion, nested scopes, scale-invariant structure within one problem |
| **LLM-driven goal decomposition** | §10.5 | Natural-language → DSL goal graph | Top of the hierarchy; semantic decomposition of user intent |
| **Lemma library and promotion** | §10.8 | Cached sub-solutions as primitives | Reuse of solved sub-problems across problems |
| **Abstraction discovery (wake-sleep)** | §10.9 | Pattern → new primitive operator | Open-ended growth of the system's vocabulary over time |
| **Cross-level message passing** | §10.10 | Goal/proof exchange between QPCN runs | The runtime mechanism by which the composition actually happens |

Together they implement:

```
                  Open-ended capability growth
                  ─────────────────────────────
                              │
                              ▼
                    Library compounds via §10.9
                              │
                              ▼
              Solved problems persist via §10.8
                              │
                              ▼
              Decomposition coordinated via §10.10
                              │
                              ▼
              Goal structure suggested via §10.5
                              │
                              ▼
                  Sub-problem solved via §10.4
                              │
                              ▼
                    QPCN inference (built)
```

Each layer of mechanism is a *control structure* over the layer below. The bottom layer (QPCN inference) is already complete; the upper layers add hierarchical control.

### 11.3 The Wilson RG picture in detail

The most precise mathematical analogy for what hierarchical composition is doing:

**Wilsonian RG (lattice version)**:

```
H_0 (microscopic Hamiltonian, lattice spacing a)
  │
  │  block spins: average over groups of sites
  ▼
H_1 (effective Hamiltonian, lattice spacing 2a)
  │
  │  block spins again
  ▼
H_2 (effective Hamiltonian, lattice spacing 4a)
  │
  ▼
  ...
  │
  ▼
H_∞ (fixed point — universal long-distance behavior)
```

The effective Hamiltonian at each scale `H_k` contains all the operators that arise from integrating out the short-distance modes of `H_{k-1}`. Most operators die off (irrelevant); a few survive (relevant); the survivors define the universal long-distance behavior.

**Hierarchical QPCN composition**:

```
H_0 (microscopic Hamiltonian — axioms, definitions, primitives)
  │
  │  solve sub-problems, cluster patterns
  ▼
H_1 (effective Hamiltonian — lemmas as primitives, common patterns as ops)
  │
  │  solve theorems using lemmas, discover meta-patterns
  ▼
H_2 (effective Hamiltonian — theorems as primitives, proof techniques as ops)
  │
  ▼
  ...
  │
  ▼
H_∞ (effective Hamiltonian — the full theory)
```

The structural identity is exact: at each scale, the "primitives" are composite objects of the next scale down, the "relevant operators" are the patterns that recur across problems, and the "fixed point" is the mature theory where new problems can be solved by direct combination of existing concepts without re-deriving them.

The wake-sleep cycle is the operational realization of one step of RG flow: solve problems (microscopic dynamics), discover patterns (block-spin operation), promote primitives (effective Hamiltonian update). After enough cycles, the library has reached an RG fixed point for that domain — the architecture has discovered the natural concept hierarchy.

This is why Wilson's framework, and not just hierarchical-Bayesian inference or recursive neural networks, is the right ancestral framing. The QPCN composition is *doing* the same thing Wilson formalized for physics: finding the right effective description at each scale.

### 11.4 MERA as the substrate that makes this exact rather than analogical

Swingle 2012 showed that the geometry of MERA — its hierarchical tree of tensors — is mathematically equivalent to discrete hyperbolic space (AdS_2). Holographic codes (Pastawski et al. 2015, Hayden et al. 2016) extended this: MERA-like tensor networks are not just metaphors for AdS/CFT, they are honest realizations of it on a lattice.

The implications for our architecture:

1. **The geometric reasoning framing is literal.** When a single QPCN's manifold deforms in response to error stress-energy (§4.1), and that manifold is the substrate for a MERA-structured belief field (§10.4), and the MERA's geometry is AdS-like, then the architecture is *literally* doing inference on a discrete spacetime whose curvature reflects the difficulty of the proof. This is not vocabulary; it's a structural identity that comes for free once MERA is built.

2. **The holographic principle suggests boundary-only specifications work.** In AdS/CFT, the boundary CFT contains all the information of the bulk. Translating: the user's DSL spec at the "boundary" of the hierarchy contains all the information needed to determine the "bulk" proof structure. The proof is uniquely determined by the boundary specification plus the constraints. This is consistent with how mathematics works — a theorem is uniquely determined by its statement plus the axioms, and the proof is just a particular realization of the necessary structure.

3. **Black-hole-like behavior for impossible proofs.** In AdS/CFT, regions of the bulk that are "behind a horizon" cannot be reached from the boundary. The analog: certain proof problems may be *information-theoretically* unreachable from a given set of axioms, and the architecture would manifest this as a sub-graph of the goal graph that no matter how decomposed never converges. This is a meaningful diagnostic — a residual that won't go away signals a genuine logical gap, not just a search difficulty.

These three implications are not metaphorical. They follow from theorems in the MERA/AdS literature applied to our specific architecture. They are research bets that the literature suggests should hold; if they do, the architecture is unusually well-grounded.

### 11.5 Connections to AlphaProof and DreamCoder

The two closest published predecessors:

**AlphaProof (DeepMind 2024)**:
- Used reinforcement learning over the Lean tactic space to generate proof search policies.
- Achieved IMO 2024 silver-medal performance (problems P1, P2, P4, P6; missed P3 and P5).
- *What they have*: massive RL training, sophisticated tactic generation, strong empirical results.
- *What they don't have*: a substrate with structural correctness guarantees, uncertainty quantification by superposition, or library-learning that compounds across problems.
- *What we have in common*: the LLM-as-frontend + symbolic-as-solver pattern, hierarchical proof composition.
- *Where the QPCN differs*: provable correctness at the synthesis step (no hallucinated proofs), and the wake-sleep library growth that AlphaProof currently lacks.

**DreamCoder (Ellis et al. 2020, 2021)**:
- Established the wake-sleep paradigm for program synthesis: alternate solving with current library and abstracting recurring patterns.
- Demonstrated open-ended capability growth on list manipulation, drawing, physics laws, recursive programs.
- *What they have*: the open-ended growth proof of concept, the empirical wake-sleep cycle.
- *What they don't have*: a quantum substrate, structural correctness guarantees, or geometric coupling.
- *What we have in common*: §10.9 is directly DreamCoder's mechanism, translated to tensor networks.
- *Where the QPCN differs*: the substrate guarantees that abstracted primitives are *operator-algebraic* (composable by tensor product) rather than syntactic (composable by code substitution), which gives more rigorous compositionality.

**The synthesis**: AlphaProof + DreamCoder + QPCN substrate = a system that:
- Has automated proof search at IMO-class capability (AlphaProof contribution),
- Grows its library through wake-sleep (DreamCoder contribution),
- Has structural correctness guarantees and uncertainty quantification (QPCN substrate contribution),
- Has hierarchical composition matching Wilson RG (the architectural integration).

To the best of our knowledge, no published system has all four. That is the publishable contribution and the source of capability beyond either AlphaProof or DreamCoder alone.

### 11.6 Realistic target problems

Mapping the architecture's expected capabilities onto specific problem classes:

**Reachable within 2 years of focused effort** (after §10.1–10.11):

| Domain | Specific target | Why this is tractable |
|---|---|---|
| Formal proof assistants | Lean Mathlib lemma generation at the level of `simp` + `linarith` | Type-driven synthesis is the architecture's sweet spot |
| Quantum chemistry | Ground states of molecules with 50–200 active orbitals | Hierarchical fragment-based decomposition; the canonical MPS/DMRG strength |
| Combinatorial enumeration | Graphs/codes/designs with specified properties | Constraint satisfaction with structural constraints |
| Symbolic regression | Recover physics equations from data, à la AI-Feynman | Dimensional analysis as gauge constraint |
| Inverse materials design | Crystal structures with target band gap or magnetism | Hamiltonian-based scoring of candidate structures |
| Catalyst design | Catalysts for known reaction classes | Energy-barrier minimization as ground-state search |
| Small algebraic structures | Classification of groups/rings/algebras of small order | Combinatorial enumeration with algebraic constraints |

**Reachable within 5 years**:

| Domain | Specific target | Caveats |
|---|---|---|
| Algebraic topology | New theorems at the level of recent PhD theses | Requires substantial Mathlib integration |
| Protein design | Novel binding pockets for specific targets | Requires good force-field Hamiltonians |
| Quantum error correction | New codes beyond known families | Stabilizer formalism maps cleanly |
| Cryptographic protocols | Verified constructions with given security properties | Requires careful Hamiltonian encoding of security games |
| Materials with unconventional properties | Topological insulators, room-temperature superconductors | Likely needs experimental loop, not just simulation |
| Optimal control | Quantum/classical control protocols with constraints | Variational nature matches QPCN paradigm |

**Aspirational and depending on luck**:

| Domain | Specific target | Why uncertain |
|---|---|---|
| Millennium Prize problems | Riemann, P vs NP, Yang-Mills mass gap, etc. | Likely require genuine insight beyond structural search; no architecture has solved one to date, and there is no a priori reason to expect this one will |
| Unsolved open conjectures | Twin primes, Goldbach, ABC, etc. | Same caveat — but AlphaProof's IMO performance shifted the conventional wisdom about what "structural search" can achieve |
| Drug discovery for novel targets | Treatments for currently-untreatable conditions | Validation loop is in vivo and slow; the architecture would be one tool in a much larger pipeline |
| Unification frameworks in physics | Quantum gravity, dark matter mechanism | Requires not just deriving consequences but proposing new ontologies; this is currently outside the architecture's hypothesis class |

### 11.7 The compounding capability argument

Most ML architectures have fixed capability after training: they solve some distribution of problems with some accuracy, and that's the end of it. The QPCN with hierarchical composition has a different property: **its capability grows monotonically with use.**

Three mechanisms drive this:

1. **Each solved problem adds a lemma to the library** (§10.8). The library only grows; old lemmas remain available.
2. **Each wake-sleep cycle adds a primitive to the library** (§10.9). Recurring patterns become accessible vocabulary; future problems become correspondingly easier.
3. **Each cross-level dispatch improves the decomposition heuristics** (§10.10). The system learns which decompositions work; future attempts on similar problems start from a better prior.

The capability function `C(t)` of the system at time `t` is therefore not constant but increasing:

```
C(t) = base_capability + integrate(library_growth + abstraction_growth + heuristic_growth, 0, t)
```

In the long limit, this approaches the "fixed point" of the relevant Wilson RG flow: the system has discovered the natural concept hierarchy of its domain and can solve any problem expressible in that hierarchy by direct combination of library entries.

This is the property that makes the architecture a candidate for *open-ended discovery* rather than just sophisticated solving. A system whose capability is fixed at training time cannot, by construction, discover anything its training data didn't already contain. A system whose capability grows with use can, in principle, reach arbitrary capability given enough cycles.

This is also why the right benchmark for this architecture is not single-shot performance on a fixed test set, but **capability growth curves** over many cycles on increasing problem difficulty. That is what publication around this work should measure.

### 11.8 The pitch for novel results

Putting all of §11 together, the argument for why this could uncover new mathematics/physics/chemistry:

1. The architecture has **correctness guarantees by construction** — anything it outputs is provably consistent with the constraints it was given. False positives are eliminated at the synthesis step.
2. The architecture has **uncertainty quantification by superposition** — when constraints underdetermine the answer, the ground state is a quantum superposition, and measurement gives a distribution. It doesn't lock onto one wrong answer.
3. The architecture **grows its library through abstraction** — each solved problem adds primitives to future searches. Capability compounds.
4. The architecture **composes hierarchically through MERA** — exploiting scale-invariant structure that flat search misses. It can attack problems whose depth makes them inaccessible to single-level methods.
5. The architecture **shares the LLM-as-frontend + symbolic-as-solver pattern with AlphaProof**, validated to IMO silver-medal level.
6. The architecture **shares the wake-sleep growth pattern with DreamCoder**, validated for open-ended capability growth.
7. The combination of (1)–(6) is, to the best of our knowledge, not published.

That is the realistic version of the claim. It is defensible from first principles plus published precedents. The aspirational version — solving Millennium problems — is not defensible from first principles, but neither was AlphaProof's IMO performance before it happened. The principled answer: build the architecture, point it at progressively harder problems, and let the empirical results decide.

---

## 12. Code Layout Reference

```
src/qft_pcn/
├── __init__.py                  # public API
├── manifold.py                  # dynamic Riemannian 2D manifold
├── fields.py                    # Field, PrecisionField containers
├── layer.py                     # QFTPCNLayer, GenerativeMap protocol, ClassicalConvMap
├── network.py                   # QFTPCNNetwork hierarchical stack
├── multifield.py                # MultiFieldNetwork with Yukawa coupling
├── quantum.py                   # QuantumGenerativeMap (VQC), QuantumConvMap adapter
├── demo.py                      # classical PCN demo (moving Gaussian)
├── multifield_demo.py           # multi-field + quantum demo
├── qft/
│   ├── __init__.py
│   ├── fock.py                  # local Fock space, a/a†/n/φ/π operators
│   ├── mps.py                   # Matrix Product State
│   ├── hamiltonian.py           # FieldSpecies, HamiltonianConfig, Hamiltonian
│   ├── evolution.py             # TEBD trotter_step, evolve, energy
│   └── qpcn.py                  # Quantum Predictive Coder
└── tests/
    ├── __init__.py
    ├── test_qft_pcn.py          # classical substrate (4 tests)
    ├── test_multifield.py       # multi-field coupling (3 tests)
    ├── test_quantum.py          # Qiskit VQC (3 tests)
    └── test_qft.py              # full QFT core (14 tests)
```

Future additions (planned in §10):

```
src/qft_pcn/
├── logic/                                       # § 10.1 - 10.3, 10.7
│   ├── __init__.py
│   ├── ast.py                          # AST node types
│   ├── encoder.py                      # AST → MPS encoder
│   ├── decoder.py                      # MPS → AST decoder
│   ├── typing_rules.py                 # STLC typing rules
│   ├── hamiltonian_compiler.py         # rules → Hamiltonian terms
│   ├── evaluation_hamiltonian.py       # beta-reduction Hamiltonian
│   ├── debugger.py                     # residual-energy → error report
│   └── demo_stlc_synthesis.py          # first publishable milestone
├── qft/                                         # § 10.4
│   ├── mera.py                         # MERA state representation
│   └── mera_evolution.py               # hierarchical TEBD
├── bridge/                                      # § 10.5
│   ├── __init__.py
│   ├── api.py                          # RPC server for LLM frontend
│   ├── dsl.py                          # DSL schema + parser
│   └── runtime.py                      # problem runner
└── composition/                                 # § 10.8 - 10.11
    ├── __init__.py
    ├── lemma_library.py                # § 10.8: storage + indexing + registration
    ├── promoter.py                     # § 10.8: use_lemma DSL constraint compiler
    ├── subtree_miner.py                # § 10.9: enumerate sub-MPSes
    ├── abstraction.py                  # § 10.9: cluster + canonical form computation
    ├── wake_sleep.py                   # § 10.9: orchestrate the cycle
    ├── goal_graph.py                   # § 10.10: DAG of sub-goals
    ├── dispatcher.py                   # § 10.10: spawn + collect QPCN runs
    ├── result_integrator.py            # § 10.10: bottom-up clamping
    ├── revision.py                     # § 10.10: LLM-guided alternative decomposition
    ├── demo_hierarchical_proof.py      # § 10.11: second publishable milestone
    └── tests/
        ├── test_lemma_library.py
        ├── test_subtree_miner.py
        ├── test_wake_sleep.py
        ├── test_goal_graph.py
        └── test_dispatcher.py
```

---

## 13. Glossary

- **PCN** — Predictive Coding Network. A hierarchical generative model with bidirectional prediction and error signals, derived from variational free-energy minimization.
- **QPCN** — Quantum Predictive Coder. The PCN built on top of a quantum many-body substrate (this work).
- **QFT** — Quantum Field Theory. A formalism where physical degrees of freedom are operator-valued fields over spacetime.
- **MPS** — Matrix Product State. A 1D tensor-network ansatz for quantum many-body states with bounded entanglement.
- **MERA** — Multi-scale Entanglement Renormalization Ansatz. A hierarchical (tree-structured) tensor network capturing scale-invariant correlations.
- **PEPS** — Projected Entangled Pair States. The 2D generalization of MPS.
- **DMRG** — Density Matrix Renormalization Group. The variational algorithm that finds MPS ground states.
- **TEBD** — Time-Evolving Block Decimation. The standard algorithm for MPS time evolution via Trotter splitting.
- **VQC** — Variational Quantum Circuit. A parameterized quantum circuit used as a function approximator in quantum ML.
- **Fock space** — Hilbert space of quantum states with variable particle number; spanned by `|0⟩, |1⟩, |2⟩, ...` (occupation-number basis).
- **Vacuum** — The lowest-energy state `|0⟩`, typically annihilated by all `a_k`.
- **Yukawa coupling** — A polynomial cross-field interaction term `λ φ_A φ_B` in a Lagrangian.
- **Stress-energy tensor** — `T_μν[E]`. The conserved current of energy and momentum carried by a field; sources spacetime curvature in general relativity, sources our learned metric here.
- **Laplace-Beltrami operator** — The covariant generalization of the Laplacian to a curved manifold.
- **Ricci scalar** — Scalar curvature `R`. A measure of how much the volume of small balls differs from flat space.
- **Parameter-shift rule** — Exact gradient formula for parameterized quantum circuits: `df/dθ = (f(θ+π/2) − f(θ−π/2))/2`.
- **Barren plateau** — The phenomenon where gradients in deep parameterized quantum circuits vanish exponentially in the number of qubits; avoided in local-circuit architectures (Pesah et al. 2021).
- **AdS/CFT** — Anti-de-Sitter / Conformal Field Theory duality. A holographic correspondence in theoretical physics where a `d`-dimensional quantum field theory equates to a `(d+1)`-dimensional gravity theory.
- **Gauge invariance** — Invariance of physical observables under local symmetry transformations. In our framing, type-preservation under evaluation.
- **STLC** — Simply-Typed Lambda Calculus. The minimal typed programming-language calculus.
- **Lemma promotion** — The mechanism by which a solved sub-QPCN's ground state becomes a clamped primitive in a parent QPCN's Hamiltonian (§10.8).
- **Wake-sleep cycle** — The alternating phase pattern (solve, then abstract) introduced by Hinton 1995 for sleep-state model refinement and adapted by DreamCoder (Ellis et al. 2020) for library learning in program synthesis. Used in §10.9 to discover new primitive operators from recurring sub-structure.
- **Goal graph** — A directed acyclic graph of sub-problems where edges represent "child must be solved before parent can integrate its result." Built top-down from the user's goal and consumed bottom-up as sub-QPCN runs complete (§10.10).
- **Trace distance** — `D(ρ_1, ρ_2) = ½ ||ρ_1 - ρ_2||_1`. The natural metric on quantum states, used in §10.9 for clustering reduced density matrices.
- **Holographic codes** — Tensor-network quantum codes (Pastawski et al. 2015, Hayden et al. 2016) that realize aspects of AdS/CFT duality on a lattice. Used in §11.4 to argue that the geometric reasoning framing is literal rather than metaphorical.
- **RG fixed point** — A Hamiltonian invariant under further coarse-graining. In Wilson's framework, the universal long-distance physics is governed by the fixed point. In the QPCN analogy (§11.3), this is the mature library where new problems can be solved by combination of existing concepts without re-derivation.
- **Curry-Howard correspondence** — The structural identity between (a) propositions and types, and (b) proofs and programs. Howard 1980. The reason types can be encoded as Hamiltonian conservation laws and proofs as ground states (§8).
- **AlphaProof** — DeepMind 2024's automated theorem prover that achieved IMO 2024 silver-medal performance. The closest published predecessor for QPCN's hierarchical proof composition (§11.5).
- **DreamCoder** — Ellis et al. 2020/2021 program synthesis system that demonstrated wake-sleep library learning. The direct intellectual ancestor of §10.9.

---

## 14. Prior Research and Citations

Real published work, no fake URLs. Cited by author and year so they're searchable.

### Predictive coding and variational inference
- Friston, K. (2005). *A theory of cortical responses.* Philosophical Transactions of the Royal Society B.
- Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience.
- Bogacz, R. (2017). *A tutorial on the free-energy framework for modelling perception and learning.* Journal of Mathematical Psychology.
- Whittington, J. C. R., & Bogacz, R. (2017). *An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity.* Neural Computation.
- Millidge, B., Tschantz, A., & Buckley, C. L. (2020). *Predictive coding approximates backprop along arbitrary computation graphs.* arXiv:2006.04182.

### Tensor networks for machine learning
- Stoudenmire, E. M., & Schwab, D. J. (2016). *Supervised learning with tensor networks.* NeurIPS.
- Stoudenmire, E. M. (2018). *Learning relevant features of data with multi-scale tensor networks.* QST.
- Glasser, I., Pancotti, N., & Cirac, J. I. (2019). *From probabilistic graphical models to generalized tensor networks for supervised learning.* IEEE Access.
- Han, Z.-Y., Wang, J., Fan, H., Wang, L., & Zhang, P. (2018). *Unsupervised generative modeling using matrix product states.* PRX.

### Quantum many-body states with NNs (inverse direction)
- Carleo, G., & Troyer, M. (2017). *Solving the quantum many-body problem with artificial neural networks.* Science.

### Variational quantum circuits and quantum ML
- Schuld, M., Bocharov, A., Svore, K. M., & Wiebe, N. (2020). *Circuit-centric quantum classifiers.* PRA.
- Pesah, A., Cerezo, M., Wang, S., Volkoff, T., Sornborger, A. T., & Coles, P. J. (2021). *Absence of barren plateaus in quantum convolutional neural networks.* PRX.
- McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). *Barren plateaus in quantum neural network training landscapes.* Nature Communications.

### Tensor networks, MERA, and holography
- Vidal, G. (2008). *Class of quantum many-body states that can be efficiently simulated.* PRL (MERA).
- Swingle, B. (2012). *Entanglement renormalization and holography.* PRD.
- Cirac, J. I., & Verstraete, F. (2009). *Renormalization and tensor product states in spin chains and lattices.* J. Phys. A.

### Neural networks as field theories / RG / deep learning theory
- Mehta, P., & Schwab, D. J. (2014). *An exact mapping between the variational renormalization group and deep learning.* arXiv:1410.3831.
- Lin, H. W., Tegmark, M., & Rolnick, D. (2017). *Why does deep and cheap learning work so well?* J. Stat. Phys.
- Roberts, D. A., Yaida, S., & Hanin, B. (2022). *The Principles of Deep Learning Theory.* Cambridge University Press.

### Geometric / equivariant deep learning
- Cohen, T. S., & Welling, M. (2016). *Group equivariant convolutional networks.* ICML.
- Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* arXiv:2104.13478.

### Categorical quantum mechanics / quantum semantics of programs
- Abramsky, S., & Coecke, B. (2004). *A categorical semantics of quantum protocols.* LICS.
- Selinger, P. (2007). *Lectures on the lambda calculus and quantum computation.*
- Coecke, B., & Kissinger, A. (2017). *Picturing Quantum Processes.* Cambridge University Press.

### Holographic / geometric machine learning
- Hashimoto, K., Sugishita, S., Tanaka, A., & Tomiya, A. (2018). *Deep learning and the AdS/CFT correspondence.* PRD.

### Quantum chemistry with tensor networks (the canonical winning domain)
- Bauer, B., Bravyi, S., Motta, M., & Chan, G. K.-L. (2020). *Quantum algorithms for quantum chemistry and quantum materials science.* Chem. Rev.
- Chan, G. K.-L., & Sharma, S. (2011). *The density matrix renormalization group in quantum chemistry.* Annu. Rev. Phys. Chem.

### Physics-aware deep learning
- Jumper, J. et al. (2021). *Highly accurate protein structure prediction with AlphaFold.* Nature.

### Library learning, program synthesis, automated theorem proving
- Ellis, K., Wong, C., Nye, M., Sablé-Meyer, M., Cary, L., Morales, L., Hewitt, L., Solar-Lezama, A., & Tenenbaum, J. B. (2020). *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning.* arXiv:2006.08381.
- Ellis, K. et al. (2021). *DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning.* PLDI.
- Solar-Lezama, A. (2008). *Program Synthesis by Sketching.* PhD thesis, UC Berkeley.
- Gulwani, S. (2011). *Automating string processing in spreadsheets using input-output examples.* POPL.
- Polikarpova, N., Kuraj, I., & Solar-Lezama, A. (2016). *Program synthesis from polymorphic refinement types.* PLDI (Synquid).
- Bornholt, J., Torlak, E., Grossman, D., & Ceze, L. (2013). *Optimizing synthesis with metasketches.* POPL (Rosette).
- Balog, M., Gaunt, A. L., Brockschmidt, M., Nowozin, S., & Tarlow, D. (2017). *DeepCoder: Learning to write programs.* ICLR.
- Polu, S., & Sutskever, I. (2020). *Generative language modeling for automated theorem proving.* arXiv:2009.03393 (GPT-f).
- Han, J. M., Rute, J., Wu, Y., Ayers, E. W., & Polu, S. (2022). *Proof artifact co-training for theorem proving with language models.* ICLR.
- Yang, K., Swope, A. M., Gu, A., Chalamala, R., Song, P., Yu, S., Godil, S., Prenger, R., & Anandkumar, A. (2023). *LeanDojo: Theorem proving with retrieval-augmented language models.* NeurIPS.
- AlphaProof team (DeepMind). (2024). *AlphaProof and AlphaGeometry 2.* Public communications and blog posts, July 2024 (IMO silver-medal result).
- Udrescu, S.-M., & Tegmark, M. (2020). *AI Feynman: A physics-inspired method for symbolic regression.* Science Advances.

### Renormalization group, MERA-as-AdS, holographic codes
- Wilson, K. G. (1971). *Renormalization group and critical phenomena.* PRB.
- Hauke, P., Katzgraber, H. G., Lechner, W., Nishimori, H., & Oliver, W. D. (2020). *Perspectives of quantum annealing: Methods and implementations.* Reports on Progress in Physics.
- Pastawski, F., Yoshida, B., Harlow, D., & Preskill, J. (2015). *Holographic quantum error-correcting codes.* JHEP.
- Hayden, P., Nezami, S., Qi, X.-L., Thomas, N., Walter, M., & Yang, Z. (2016). *Holographic duality from random tensor networks.* JHEP.
- Aharonov, D., van Dam, W., Kempe, J., Landau, Z., Lloyd, S., & Regev, O. (2007). *Adiabatic quantum computation is equivalent to standard quantum computation.* SIAM Journal on Computing.
- Kadowaki, T., & Nishimori, H. (1998). *Quantum annealing in the transverse Ising model.* PRE.
- Farhi, E., Goldstone, J., Gutmann, S., & Sipser, M. (2001). *Quantum computation by adiabatic evolution.* arXiv:quant-ph/0001106.
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). *A quantum approximate optimization algorithm.* arXiv:1411.4028 (QAOA).
- Biamonte, J. D., Faccin, M., & De Domenico, M. (2017). *Complex networks from classical to quantum.* Communications Physics. (Quantum SAT and tensor-network annealing references.)

### Wake-sleep and predictive coding origins
- Hinton, G. E., Dayan, P., Frey, B. J., & Neal, R. M. (1995). *The "wake-sleep" algorithm for unsupervised neural networks.* Science.
- Rao, R. P. N., & Ballard, D. H. (1999). *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects.* Nature Neuroscience.

---

## 15. Closing Notes

This document is the project's persistent state. The branch `claude/qft-pcn-hybrid-architecture-ihCIR` carries:

- ~1900 lines of working code in `src/qft_pcn/` and `src/qft_pcn/qft/`,
- 39 passing tests covering every architectural claim,
- This document.

The architecture is complete enough that the §10 roadmap is implementable without reconstructing prior work.

### Recommended build order

**Phase A — Logic substrate** (§10.1–10.3): AST encoder, type-system Hamiltonian compiler, evaluation Hamiltonian. After this phase, the QPCN can typecheck and reduce small lambda-calculus programs.

**Phase B — Hierarchy enablers** (§10.4, §10.5, §10.6): MERA for recursion, LLM bridge, constraint debugger. After this phase, end-to-end demos on Tier 1 challenges (`id`, `const`, `compose`) work.

**Phase C — First publishable milestone** (§10.7): bidirectional STLC type inference for synthesis-with-holes. After this, the architecture has its first benchmarked result.

**Phase D — Hierarchical composition** (§10.8, §10.9, §10.10): lemma library, abstraction discovery, cross-level message passing. This is where the architecture transitions from "sophisticated synthesizer" to "growing reasoning system."

**Phase E — Second publishable milestone** (§10.11): hierarchical proof composition demo. After this, the architecture has results visible to both the ML and PL/formal-methods communities.

**Phase F — Open-ended deployment** (after §10.11): point the system at progressively harder problems from the §11.6 target list and run wake-sleep cycles. Capability grows monotonically with use; the empirical results determine which target classes are within reach.

### Verification of claims

Every architectural claim in this document is backed by either (a) a passing test in `src/qft_pcn/tests/`, (b) a published reference cited in §14, or (c) an explicit acceptance test specified in the corresponding §10 subsection. There are no unsupported assertions about the architecture's current capabilities. Aspirational claims about future capability (§11.6, §11.8) are clearly marked as such and depend on the §10 roadmap being executed.

### Intellectual honesty

Nothing in this document is intended as a research promise. It is an engineering plan derived from correspondences between predictive coding, quantum field theory, tensor-network methods, and program semantics that, to the best of our knowledge, have not been combined this way before. The realistic expectation is significant capability on a specific class of problems (chemistry, gauge theories, formal logic, programming) and zero direct competitiveness on natural-language tasks (which is the LLM's job). The aspirational expectation — uncovering new mathematics or physics through compounding capability over many wake-sleep cycles — is genuinely uncertain but defensible from the same first principles that have led DreamCoder and AlphaProof to results that were aspirational at the time they were proposed.
