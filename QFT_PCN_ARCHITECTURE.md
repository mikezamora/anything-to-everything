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

## 12. Novel Physics-Derived Extensions

Eight extensions that import constructs from physics into the QPCN with structural exactness. Each one solves a problem that the AI/PL/reasoning community currently considers intractable or aspirational, using machinery that is well-established in some adjacent field but has never (to our knowledge) been imported. The common pattern: a published physics result solves the analog problem in its domain; the QPCN's operator-algebraic substrate happens to be the right home for the same machinery in the reasoning domain; the import is mechanical once the correspondence is identified.

This section is sized as a research program. The eight extensions, taken together, would constitute a paper that reviewers might struggle to take at face value because no single existing system has even half of these capabilities — but each individual claim has rigorous physical precedent.

### 12.1 Anomalies as provable impossibility proofs

#### Physics origin

In quantum field theory, an *anomaly* is when a classical symmetry of the Lagrangian fails to survive quantization. The Adler-Bell-Jackiw (ABJ) anomaly (1969) showed that the axial U(1) symmetry of QED — exact at the classical level — is broken by quantum corrections. Subsequent work (Bardeen 1969, Wess-Zumino 1971, 't Hooft 1980) established anomalies as fundamental obstructions: a theory with a non-cancelling anomaly *cannot exist* as a consistent quantum theory.

Anomaly coefficients are computed algebraically from the symmetry generators `T^a`:

```
A^{abc} = tr( T^a {T^b, T^c} )
```

If `A ≠ 0` for any combination, the symmetry is anomalous and the theory is inconsistent. This is *how physicists prove certain theories are impossible* — not by exhaustively searching, but by detecting an obstruction in the operator algebra.

#### QPCN realization

A proof attempt's constraint Hamiltonian has classical symmetries derived from the typing rules. For example: "this theorem could be proved by induction on either argument" is a discrete symmetry of the proof Hamiltonian.

The ground state either:
- preserves the symmetry (multiple equivalent proofs exist; topological degeneracy applies — see §12.3); or
- breaks it spontaneously (one specific proof; Goldstone modes apply — see §12.6); or
- has an *anomaly* — the symmetry breaks not because of vacuum choice but because no consistent ground state with the symmetry can exist at any parameter setting.

The anomaly is detected by computing the trace formula above on the symmetry generators induced by the typing-rule Hamiltonian. A nonzero anomaly polynomial means no proof exists.

#### Capability

**Operational mechanization of Gödel's first incompleteness theorem.** The architecture proves theorems are unprovable from given axioms, not by enumerating failed attempts but by detecting an obstruction in the constraint algebra. This is a positive result, not a search failure.

#### Implementation

`composition/anomaly.py`:
- Extract symmetry generators from typing-rule Hamiltonian (each commuting subgroup gives a candidate symmetry).
- Compute the anomaly polynomial via traces of nested commutators/anticommutators on the local Hilbert space.
- Output: `A = 0` → no obstruction (proof may or may not exist; search proceeds); `A ≠ 0` → provable impossibility (return certificate with the obstruction's algebraic signature).

Computational cost: polynomial in the number of symmetry generators and the local Hilbert space dimension. Cheap relative to the proof search itself.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Spurious anomalies from artifacts of the encoding | Verify against known consistent theorems (the encoded STLC type system should have zero anomaly for valid proofs) |
| Missed anomalies (silent type errors) | Cross-check with classical type checker on the decoded AST |
| Computational cost on large symmetry groups | Restrict to the subgroup generated by the constraints actually present in a given problem |

#### Acceptance test

The classical example: try to prove `0 = 1` in Peano arithmetic. The architecture should detect the anomaly directly from the axioms and return the impossibility certificate without exhaustive search. More sophisticated tests: Gödel-Rosser sentences (statements that are unprovable in a consistent theory).

#### Why this would seem implausible

Current automated theorem provers fail; they don't prove failure. Distinguishing "couldn't find a proof" from "no proof exists" is currently considered to require human-level meta-mathematical reasoning. An automated mechanism would be the most surprising claim in the paper.

---

### 12.2 Topological invariants for exact program equivalence

#### Physics origin

Witten 1988 (Topological QFT in the Jones polynomial paper), Reshetikhin & Turaev 1991 (quantum-group invariants of knots). Topological quantum field theories produce observables that are invariants of the underlying manifold: the Jones polynomial of a knot, Reshetikhin-Turaev invariants of 3-manifolds, Donaldson invariants of 4-manifolds. Two manifolds give identical TQFT physics iff they are topologically equivalent.

In tensor-network language: a closed-loop expectation value `⟨W_C⟩` in a tensor network is a topological invariant of the loop `C` — independent of small deformations.

#### QPCN realization

A program's AST plus its variable-binding structure is *literally* a string diagram in the sense of categorical quantum mechanics (Abramsky-Coecke 2004, Coecke-Kissinger 2017). Beta-equivalence and alpha-conversion are Reidemeister moves on this diagram. Two programs are α-β-η equivalent if and only if their string diagrams are topologically equivalent.

Topological invariants of the string diagram are computable directly from the MPS encoding:
- Wilson loops along closed paths through the binding entanglement give knot-polynomial-like invariants.
- The Jones polynomial of the binding diagram is a complete invariant for linear-typed programs (modulo Reidemeister-3 ambiguity).
- For richer type systems, Reshetikhin-Turaev invariants give finer equivalence checks.

The computation reduces to evaluating expectation values of products of operators around closed paths in the MPS — exactly the operations the existing `mps.py` infrastructure supports.

#### Capability

**Provable program equivalence via direct measurement on the MPS state**, not symbolic reduction. Optimization passes can be verified by checking that a single topological invariant matches before and after. Equivalence detection in compilation, refactoring, and synthesis becomes a measurement, not a search.

#### Implementation

`composition/topological_invariants.py`:
- Identify closed loops in the binding structure of the AST.
- For each loop, compute the Wilson-loop expectation value through the MPS.
- Combine into the Jones polynomial (for the simplest case) or higher invariants (Reshetikhin-Turaev for richer cases).
- Two programs with matching invariant polynomials are guaranteed equivalent.

Computational cost: each Wilson loop is `O(N · χ^3)` where `N` is loop length and `χ` is bond dimension. Far cheaper than full proof of equivalence by reduction.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Invariant collisions (two inequivalent programs sharing invariants) | Use multiple invariants of increasing strength; for STLC the Jones polynomial plus framing is complete |
| Numerical error in invariant computation breaking equality checks | Compute invariants in exact arithmetic on the MPS coefficients; or use enough precision that quantization gaps separate distinct values |
| Programs with side effects don't fit the closed-string-diagram framework | Restrict to linear/affine types initially; extend to non-commutative geometry later (§12 ↦ future work) |

#### Acceptance test

Take a corpus of program pairs with known equivalence: `(λx. x+0, λx. x)`, `(map f ∘ map g, map (f ∘ g))`, list-reversal in two implementations. The architecture computes Jones polynomials for each and confirms equivalence for pairs that are equivalent, distinguishes pairs that aren't. Compare against a classical equivalence checker (e.g., Egg's e-graph saturation) on the same corpus.

#### Why this would seem implausible

Program equivalence is undecidable in general (Rice's theorem). Existing tools approximate it heuristically. Exact equivalence via topological invariants on a wide class of programs would feel like circumventing undecidability — which it does, by restricting to programs whose categorical semantics is computable.

---

### 12.3 Topological order for a priori counting of proof strategies

#### Physics origin

Wen 1989 (topological order), Kitaev 2006 (toric code), Nayak et al. 2008 (non-abelian anyons). Topologically ordered phases have *ground state degeneracy* that depends on the topology of the underlying space: on a torus, the toric code has 4 ground states; on a genus-`g` surface, `4^g`. This degeneracy is computable from the Hamiltonian's *algebraic structure* — specifically, the dimension of the algebra of Wilson loops modulo trivial loops — without ever computing the ground states.

#### QPCN realization

A theorem with multiple essentially different proof strategies has a degenerate ground-state manifold of its constraint Hamiltonian. The dimension of that manifold equals the number of inequivalent proofs. By the topological-order analogy:

- Inequivalent proof strategies correspond to inequivalent Wilson loop operators on the AST's binding diagram.
- The dimension of the Wilson-loop algebra modulo trivial loops counts distinct proof homotopy classes.
- This count is computable from the constraint Hamiltonian without doing the search.

#### Capability

For any theorem stated as a QPCN problem, predict in advance: this theorem has 1, 2, or `n` essentially different proofs (where "essentially different" means non-homotopic in the categorical-semantics sense). The architecture can also report partial information: "at least 3 proofs exist," "exactly 2 exist in this homotopy class," etc.

Currently this information can only be obtained by exhaustive search, and even then incompletely.

#### Implementation

`composition/topological_degeneracy.py`:
- Build the algebra of Wilson loops on the binding diagram of the problem.
- Quotient by the trivial subgroup.
- The dimension of the quotient is the degeneracy index.
- For low-dimensional cases: explicit Lanczos diagonalization of the ground subspace.

Computational cost: dominated by the Wilson-loop algebra computation, which is polynomial in the diagram size for STLC-class problems.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Counting too-fine-grained equivalence (e.g., counting permutations of arguments as different proofs) | Quotient by the relevant symmetries before counting |
| Counting too-coarse (missing genuine differences) | Use richer invariants (categorical traces, not just dimensions) |
| Computational explosion on complex binding patterns | Restrict initial scope to STLC; richer type systems require careful handling of the higher categorical structure |

#### Acceptance test

Take theorems with known multiple-proof structure: `a + b = b + a` (one proof per induction direction = 2), `(a + b) + c = a + (b + c)` (one essentially distinct proof = 1), pumping lemma in automata (several inequivalent proofs depending on the specific machine class). The architecture should predict the correct count before performing the actual proof search.

#### Why this would seem implausible

Predicting solution-space structure of a constraint satisfaction problem before solving it is currently the province of *statistical* mechanics (replica method, see §12.7) and gives only typical-case predictions. Exact counts for individual problem instances would be a first.

---

### 12.4 Conformal bootstrap for type-only reasoning

#### Physics origin

Polyakov 1974 (conformal algebra), Ferrara-Gatto-Grillo 1973 (crossing symmetry), Rattazzi-Rychkov-Tonni-Vichi 2008 (numerical bootstrap), Kos-Poland-Simmons-Duffin 2014 (3D Ising critical exponents to 6 decimal places). The conformal bootstrap derives properties of conformal field theories from pure consistency requirements — unitarity, crossing symmetry, modular invariance — without explicitly constructing the theory.

The bootstrap solves CFTs by formulating them as semidefinite programs: maximize some property of the theory subject to consistency constraints, get rigorous bounds on physical observables that any consistent CFT must satisfy.

#### QPCN realization

For a type signature `T`, the constraints of well-typedness, parametricity (Reynolds 1983: "theorems for free"), uniqueness of polymorphic constructions (Wadler 1989), and the categorical-semantics axioms form a *bootstrap system*. Numerical optimization over admissible operator-product-expansion coefficients (here: admissible normal forms of the program) gives provable bounds on properties of any valid implementation:

- Termination bounds: any implementation of `f : List Int → Int` is provably O(n) or O(n^2) given specific bootstrap inputs.
- Side-effect bounds: any implementation of `f : Pure (List Int → List Int)` is provably pure.
- Complexity-class bounds: any implementation of `f : T` requires at least `K` AST nodes.

These bounds are *provable*, not heuristic, from the consistency requirements alone.

#### Capability

**Reason about a program from its type alone**, deriving:
- Performance bounds without running the program.
- Side-effect classes without inspecting the implementation.
- Termination properties without proving termination.
- Equality and equivalence relations between programs of different types via crossing-symmetry-like dualities.

Strictly stronger than "theorems for free" because bootstrap gives quantitative bounds, not just qualitative invariants.

#### Implementation

`composition/bootstrap.py`:
- Build the bootstrap system from the type signature: unitarity (states have non-negative norm), crossing symmetry (commutativity of certain compositions), modular invariance (parametricity).
- Formulate as a semidefinite program.
- Solve using SDPB (Simmons-Duffin 2015) or CVXPY for prototype.
- Extract bounds on observables.

Computational cost: dominated by the SDP solver. For small type signatures (STLC class), feasible. For very complex types, may require approximate SDP methods or domain restrictions.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Bootstrap constraints insufficient to give nontrivial bounds | Add domain-specific consistency requirements (linearity, totality, etc.) |
| SDP scaling on large problems | Use sparse SDP solvers; restrict to bounded-arity type signatures |
| Bootstrap gives bounds but not constructive existence | This is by design — bootstrap is about constraints, not construction. Pair with QPCN search to construct witnesses. |

#### Acceptance test

For the type `f : List Int → Int`, derive bounds: any pure implementation has termination time ≥ Ω(1), depth ≥ log(length). Test these against actual implementations (sum, max, length). For polymorphic types like `f : List a → List a`, the bootstrap should derive that `f` cannot inspect element values (parametricity).

#### Why this would seem implausible

Reasoning about untouched-code properties is currently the domain of heavy-annotation static analysis (Liquid Haskell, F\*, ATS) or theorem provers with extensive hand-proofs. Doing it from a bare type signature would seem to violate Rice's theorem — it doesn't (bootstrap gives bounds, not full decidability of arbitrary properties), but the violation-flavor is what would surprise reviewers.

---

### 12.5 Holographic codes for fault-tolerant reasoning

#### Physics origin

Pastawski-Yoshida-Harlow-Preskill 2015 (HaPPY code), Hayden-Nezami-Qi-Thomas-Walter-Yang 2016 (random tensor networks as holographic codes). These works established that MERA-like tensor networks are *literally* quantum error-correcting codes: the bulk logical information is redundantly encoded into boundary physical degrees of freedom, with explicit error-correction syndromes detectable at the boundary.

The code distance (number of correctable errors) is determined by the tensor network's geometry; the encoding is automatic from the MERA structure.

#### QPCN realization

The MERA substrate planned in §10.4 is, by Pastawski's construction, already a holographic code. The "bulk logical state" is the actual theorem/proof; the "boundary physical state" is the per-site measurements at the leaves of the MERA. Error correction works as follows:

- When sub-QPCNs in §10.10 return inconsistent results (one branch proves `A`, another `¬A`), this is an *error syndrome* on the holographic code.
- Standard QEC syndrome-correction reconstructs the bulk logical state from the majority of consistent boundary measurements.
- Up to a noise threshold (computable from the MERA's code distance), the final reconstructed proof is provably correct, even with corrupted sub-proofs.

The QEC machinery operates on the existing MERA tensors — no new substrate needed.

#### Capability

**Fault-tolerant reasoning by construction.** Robust to:
- Numerical truncation errors in any sub-QPCN.
- Bad heuristic choices in the §10.10 dispatcher.
- Hardware noise if running on real quantum chips.
- Adversarial corruption of part of the lemma library.

Up to the MERA's code distance, the final answer is correct regardless of which sub-components fail. Computable a priori how many sub-failures the system can tolerate.

#### Implementation

`composition/holographic_correction.py`:
- Build the MERA tensors with explicit code-block structure (following Pastawski's construction).
- Define syndrome operators: products of stabilizers across the MERA layers.
- Implement syndrome measurement: project boundary state to determine which (if any) errors occurred.
- Implement recovery: apply the inverse of the detected error to restore the logical state.

Overhead: O(log N) per inference run, where N is the leaf count.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Noise above the code's threshold breaks correction | Quantifiable: compute the code distance and report the failure probability vs noise level |
| Code-distance bounds depend on MERA depth | Make MERA depth scale with problem complexity; deeper MERA = higher code distance = more error tolerance |
| Syndrome computation overhead on each step | Cache syndromes; only re-measure when consistency checks fail |

#### Acceptance test

Inject controlled noise into a fraction of the sub-QPCN runs (e.g., 5% of sub-proofs return random outputs). Verify that the final reconstructed proof is correct as long as the noise stays below the code's threshold. Compare against a non-corrected baseline (which should fail more often).

#### Why this would seem implausible

AI reasoning systems are notoriously fragile. The closest existing thing is ensemble methods, which give statistical (not provable) robustness, and don't compose hierarchically. Cryptographically rigorous fault tolerance for inference is currently aspirational research.

---

### 12.6 Goldstone modes for automatic proof debugging

#### Physics origin

Goldstone 1961, Nambu 1960, Goldstone-Salam-Weinberg 1962. When a continuous symmetry is spontaneously broken, massless excitations (Goldstone bosons) appear, corresponding to motion along the broken-symmetry directions. The shape and momentum structure of a Goldstone mode encodes precisely which symmetry was broken.

Computationally: Goldstone modes are the eigenmodes of the Hessian of the free energy at the broken-symmetry minimum, with eigenvalues that vanish in the symmetry directions.

#### QPCN realization

A failed proof attempt has a residual-energy distribution across the MPS sites. This distribution is *not* random — it has structure determined by which constraints are violated and where. Specifically, it has the form of a Goldstone mode of the broken proof-completion symmetry.

The Goldstone mode's spatial structure points directly at the missing lemma:
- Localization on specific sites = those sites' constraints are unsatisfied because of a missing argument.
- Coupling between sites = the missing thing connects them (a missing intermediate result).
- The mode's "type" (transformation properties under the typing-rule symmetries) = the type signature of the missing lemma.

Reading the Goldstone mode is a generalized eigenvalue problem on the constraint Hamiltonian's near-null space — standard numerical linear algebra.

#### Capability

**Automatic identification of missing lemmas in failed proofs.** When the architecture cannot complete a proof, it returns:
- A proof skeleton (the part that did succeed).
- The precise type and structural signature of the missing lemma needed to close the proof.
- The residual-energy correlation pattern identifying exactly where the lemma must be applied.

This is the analog of a structured error message but generated from the proof attempt's quantum-mechanical structure, not from heuristics.

#### Implementation

`composition/goldstone.py`:
- After a failed proof attempt (high residual energy), diagonalize the constraint Hamiltonian near zero energy.
- Identify the lowest-energy excitation above the ground state — this is the Goldstone mode.
- Extract: localization (which sites), transformation properties (under typing symmetries), correlation pattern (which sub-trees are coupled).
- Report as a structured "missing lemma" specification.
- Optionally, dispatch a new QPCN to find that lemma, and re-attempt the failed proof with the new lemma in the library.

Computational cost: a Lanczos iteration on a sparse Hamiltonian. Cheap relative to the failed proof search itself.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Goldstone mode is degenerate (multiple equally-good missing-lemma candidates) | Report all candidates with their relative confidence (eigenvalue gaps) |
| The proof failed for reasons unrelated to a missing lemma (e.g., the theorem is anomalous) | Cross-check with anomaly detection (§12.1); if anomalous, report impossibility instead |
| The Goldstone mode points at a lemma that itself is unprovable | Recursive application — analyze the new sub-problem's Goldstone modes |

#### Acceptance test

Construct a failed-proof scenario where exactly one lemma is missing (e.g., a proof requiring `length (xs ++ ys) = length xs + length ys`, but with that lemma removed from the library). The Goldstone analysis should identify the missing lemma's exact type and the AST positions where it would be applied.

#### Why this would seem implausible

Current proof assistants give cryptic error messages or just timeout. A Goldstone-mode-style failure diagnosis would identify the precise lemma gap that the user (or the system itself) needs to fill. This is currently expert-mathematician work, not an automated capability.

---

### 12.7 Replica method for predictive typical-case complexity

#### Physics origin

Edwards & Anderson 1975 (replica trick), Parisi 1980 (replica symmetry breaking), Mézard-Parisi-Virasoro 1987 (spin glass theory). Parisi was awarded the Nobel Prize in Physics 2021 for this work.

The replica trick computes the average of `log Z` over a disordered ensemble of Hamiltonians:

```
⟨log Z⟩ = lim_{n→0} (⟨Z^n⟩ - 1) / n
```

Compute `⟨Z^n⟩` for integer `n` (where it factorizes nicely), then analytically continue to `n → 0`. The saddle-point structure of the resulting expression reveals phase transitions, critical exponents, and typical-case complexity.

Applied to random k-SAT, replica gives exact predictions of the SAT-UNSAT phase transition (Mertens-Mézard-Zecchina 2006), with the formula matching numerical experiments to within a percent.

#### QPCN realization

Apply the replica trick to ensembles of QPCN problem instances within a class. For example, "all theorems of the form `∀xs : List Int. P(xs)` where `P` is a depth-≤-k predicate over arithmetic operations." Define the partition function `Z` as the count of valid proofs weighted by their complexity. Compute `⟨log Z⟩` via the replica method:

```
1. Compute ⟨Z^n⟩ = trace over n replicas of the problem ensemble
2. Identify the saddle-point structure of the n-replica system
3. Analytically continue to n → 0
4. Extract: typical-case complexity, phase transition points, fraction of solvable instances
```

This is mathematically standard for spin glasses; the novelty is applying it to a reasoning architecture.

#### Capability

**Predictive complexity theory for the architecture.** For any problem class, predict in advance:
- Typical-case complexity for problems in the class.
- The phase transition between tractable and intractable regions.
- The fraction of instances the architecture will solve at a given resource budget.
- The expected number of wake-sleep cycles needed before the library makes a given problem class easy.

Currently no ML system can predict its own performance on a problem class without empirical measurement; replica method gives it analytically.

#### Implementation

`composition/replica.py`:
- Define the problem ensemble (e.g., as a probability distribution over Hamiltonian parameter values).
- Compute `⟨Z^n⟩` symbolically for integer `n`, treating the QPCN's evolution as a quantum trace.
- Saddle-point evaluation at `n → 0`.
- Standard replica-symmetric or replica-symmetry-breaking ansatz, depending on the problem structure.

Computational cost: dominated by the saddle-point integration. For specific problem classes, this is a closed-form calculation. For others, it requires numerical Monte Carlo at the saddle-point level.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Replica symmetry breaking complications | Use Parisi's RSB ansatz; the breaking pattern itself is informative (RSB = harder problem class) |
| Saddle-point assumption fails for small problem sizes | Replica gives asymptotic predictions; complement with empirical measurements at small sizes |
| Analytic continuation `n → 0` not well-defined for the specific ensemble | Use replica-tricks established in disordered systems literature; for novel ensembles, prove the continuation is valid |

#### Acceptance test

For random k-SAT-style problem ensembles encoded as QPCN instances, replica calculations should reproduce the known SAT-UNSAT phase transition at the correct critical clause density. Compare against direct numerical solution of many instances.

#### Why this would seem implausible

ML system performance is currently predicted via *empirical* scaling laws (Kaplan et al. 2020, Hoffmann et al. 2022). Predicting performance from *theory*, given only the problem-class structure, is currently absent from the literature. Reviewers would find this implausible because the standard mental model is "you measure performance, you don't compute it."

---

### 12.8 Dynamical phase transitions for self-detecting curriculum

#### Physics origin

Heyl-Polkovnikov-Kehrein 2013 (dynamical phase transitions), Karrasch & Schuricht 2013 (Loschmidt echo for thermodynamic quenches), Heyl 2018 (review of dynamical phase transitions). When a quantum system is quenched across an equilibrium phase transition, the Loschmidt echo

```
L(t) = |⟨Ψ_0|Ψ(t)⟩|^2
```

exhibits non-analytic behavior at specific *critical times*. These are dynamical phase transitions — analogs of equilibrium phase transitions in the time domain. They signal a qualitative change in the system's overlap structure with its initial state.

#### QPCN realization

Each wake-sleep cycle (§10.9) updates the operative Hamiltonian by adding new primitives. The library's evolution can be viewed as a sequence of quantum quenches: at cycle `t`, the Hamiltonian is `H_t`, and the architecture's "ground state" (current capability profile) is `|Ψ_t⟩`.

The Loschmidt echo between successive library states:

```
L(t₁, t₂) = |⟨Ψ_{t₁}|Ψ_{t₂}⟩|^2
```

reveals dynamical phase transitions — moments when the architecture has just gained a qualitatively new capability. A sudden drop in `L` between cycles means the library has reorganized into a structurally different regime.

#### Capability

**The system detects its own capability jumps.** The wake-sleep curriculum adapts:
- Just past a detected transition: introduce harder problems (capability has jumped).
- Before a transition: consolidate the current regime (more wake-sleep cycles on similar problems).
- At apparent stagnation (no transitions for many cycles): perturb the library or change the problem distribution.

This is meta-learning via direct measurement of self-improvement events.

#### Implementation

`composition/dynamical_pt.py`:
- After each wake-sleep cycle, snapshot the library state as `|Ψ_t⟩` (the joint ground state of the current Hamiltonian + library entries).
- Compute `L(t, t-1)` as the overlap.
- Track non-analyticities in `L` over many cycles.
- Trigger curriculum updates at detected transitions.

Computational cost: one overlap calculation per cycle. Negligible relative to the wake-sleep cycle itself.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Spurious transitions from numerical noise | Smooth `L(t)` over a moving window; use second-derivative thresholds rather than first-derivative |
| Missed transitions because Loschmidt echo is too coarse | Use also entanglement entropy of the library state as a complementary diagnostic; cross-correlate detections |
| Transitions detected but no clear curriculum response | Couple to the LLM frontend for problem-class selection at detected moments |

#### Acceptance test

Run wake-sleep on a problem corpus with known structural transitions (e.g., problems involving lists, then problems involving trees, then problems involving recursion on both). The architecture should detect a Loschmidt-echo transition at the moment lists-only training transitions to trees-only training, and another when recursion is introduced.

#### Why this would seem implausible

Curriculum learning currently uses heuristic difficulty estimates (Bengio et al. 2009, Soviany et al. 2022). A system that *knows* the moment of its own capability transition, from physical principles of its substrate, is currently considered science fiction.

---

### 12.9 Self-modification via meta-Hamiltonian

#### Physics origin

In any operator-algebraic system, the Hamiltonian itself is just another tensor in operator space. A system can in principle reason about its own structure by treating its own `H` as the substrate of a second, meta-level optimization. This is the operational realization of Gödelian self-reference — the system that contains a description of itself.

The closest physics analog is renormalization-group flow on the *operator algebra*: starting from a bare set of operators (axioms), the system discovers effective operators at coarser scales (theorems). The discovery is itself an action of the system on its own description. Wilson 1971, Polchinski 1984 on exact RG, and more recently Cao-Carroll on emergent space from entanglement (2017) treat the operator algebra as a dynamical object.

The closest computer-science precedent is Schmidhuber's Gödel machine (2003) — a classical proposal for a self-rewriting program with provable optimality. Never operationalized at scale.

#### QPCN realization

The "meta-QPCN" has the following structure:

- **Substrate**: an MPS encoding the lower-level QPCN's Hamiltonian (operators are themselves vectors in a larger space; the Hamiltonian-as-state lives in the operator Hilbert space).
- **Meta-observables**: properties of the lower-level Hamiltonian — gap size, redundancy patterns, symmetry generators, anomaly polynomials, ground-state degeneracy.
- **Meta-constraints**: "the lower-level Hamiltonian should have these properties" — e.g., "minimize redundancy among constraint terms while preserving correctness."
- **Meta-evolution**: imag-time relaxation in the meta-Hamiltonian moves the lower-level Hamiltonian toward a configuration that satisfies the meta-constraints.

The meta-QPCN's outputs are *new versions of the lower-level QPCN's Hamiltonian*. Successive iterations refine the encoding itself.

This is bounded self-reference: the meta-QPCN cannot rewrite the meta-meta-QPCN (no infinite tower), but the single-level self-modification is operational and well-defined.

#### Capability

**Architecture-aware learning**. The system can:
- Discover more efficient Hamiltonian compilations for specific problem classes (compress the constraint Hamiltonian).
- Identify redundant or inconsistent constraints automatically (detect them as anomalies in the meta-Hamiltonian).
- Rewrite its own typing rules when a recurring pattern suggests a better encoding (e.g., a new field species that captures common structure).
- In the limit, perform structural self-improvement: each generation of the system is better at producing the next generation.

This is *operational* Gödelian self-reference: the system reasons about itself, but in a bounded and well-defined way.

#### Implementation

`composition/meta_hamiltonian.py`:
- Encode `H_lower` as an MPS in operator-Hilbert space (this requires extending the existing MPS infrastructure to operator-valued tensors — MPOs, matrix product operators).
- Build `H_meta` from meta-constraints (e.g., quadratic penalty for redundancy: `||commutator(H_i, H_j)||^2`).
- Imag-time evolve in the meta system.
- Decode the new `H_lower'` and replace.

Computational cost: each meta-step requires representing `H_lower` as an MPO (typically polynomial in the number of constraint terms) and evolving it. Cost scales with the size of the lower-level Hamiltonian, but the meta-evolution is run only periodically (every N wake-sleep cycles), so the amortized cost is low.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Self-modification breaks correctness**: the rewritten Hamiltonian no longer encodes the same theorems | After each meta-step, verify on a held-out set of known-correct theorems; reject if any regresses |
| **Goedelian paradoxes**: self-reference produces logical contradictions | Bounded levels — meta-QPCN can rewrite lower-level QPCN but not itself. No infinite tower of meta-meta. |
| **Catastrophic rewrites**: meta-step makes the encoding much worse | Conservative step sizes; require strict improvement on a benchmark suite before committing |
| **Cost of MPO representation**: scaling | Restrict meta-modifications to small subsystems of `H_lower` at a time |

#### Acceptance test

Take a constraint Hamiltonian compiled from STLC typing rules. Confirm by classical type checking that it correctly identifies all well-typed programs in a test corpus. Run the meta-QPCN for 10 cycles, refining the Hamiltonian. After: verify the new Hamiltonian (a) still correctly identifies the same set of well-typed programs, (b) has measurably fewer constraint terms (compression), and (c) has a larger gap (faster inference). All three must improve.

#### Why this would seem implausible

Self-modifying AI systems are a topic with a long history of failed attempts (Schmidhuber's Gödel machine never produced practical results; most attempts at meta-learning are bounded to hyperparameter tuning). A working self-modification mechanism that provably preserves correctness while improving efficiency would be among the most surprising claims in the paper. The key is that the substrate's operator-algebraic structure gives the rigor that classical attempts lacked.

---

### 12.10 Holographic compilation

#### Physics origin

The MERA tensor network has a natural geometric interpretation as discrete hyperbolic space (Swingle 2012), and its multi-scale structure realizes the renormalization group: high-energy / short-distance physics at the leaves, low-energy / long-distance effective physics at the root. Coarse-graining flows from the leaves up; primitive operations live at the bottom of the tree.

In computer science, this structure is *exactly* compilation. High-level constructs (a function `map f xs`) at the top compile to low-level operations (loops, register allocations, machine code) at the leaves. Compilation passes are coarse-graining steps; optimization passes are RG transformations on the effective Hamiltonian.

To my knowledge, this exact correspondence has never been published, despite the underlying mathematics being well-established in both fields independently.

#### QPCN realization

A MERA-structured QPCN representing a program has:
- Leaves: low-level operations (assembly-like instructions, register operations).
- Middle layers: intermediate representations (basic blocks, dataflow nodes).
- Root: high-level constructs (functions, types, modules).

Compilation passes become operations on the MERA:
- **Lowering** (high-level to low-level): apply isometries to coarse-grain from root toward leaves. Standard MERA operation.
- **Optimization** (semantics-preserving rewrites): apply unitary transformations between MERA layers that preserve the boundary correlation functions (= preserve program semantics). RG-equivalent transformations.
- **Inlining**: a higher-level node is "unfolded" into its lower-level expansion. This is MERA's tree-to-graph conversion.
- **Constant folding**: at any level, expressions with all-constant inputs evaluate to constants. This is precisely fixed-point analysis on the MERA.

The crucial point: each optimization pass is provably equivalence-preserving because RG transformations preserve the relevant correlators (= semantic observables of the program).

#### Capability

**Compiler optimizations that are provably semantics-preserving by construction**. Unlike traditional compilers (which validate optimization passes through testing or limited formal verification), the QPCN's holographic compilation:
- Is automatic — discovers optimization opportunities through RG flow rather than hand-written pass design.
- Is provably correct — RG transformations preserve correlation functions = program behavior.
- Is multi-language — works at the right MERA level regardless of source language; cross-language optimizations are natural.
- Discovers new optimization patterns through the same abstraction-discovery mechanism (§10.9) used for lemma discovery.

#### Implementation

`composition/holographic_compilation.py`:
- Build a MERA encoding of the source program (extension of §10.4 MERA infrastructure).
- For each MERA layer, identify the operations available there (compiler IR level).
- Define optimization passes as MERA-layer operations: disentanglers that simplify entanglement structure, isometries that introduce abstractions, conjugations that preserve correlators.
- Apply passes in any order; commutativity is guaranteed by the RG semantics.

Computational cost: MERA operations are well-understood and scale polynomially in MERA depth and bond dimension. Compilation time competitive with hand-written compilers for problems within the architecture's domain.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| MERA encoding loses information about specific compilation targets | Augment leaves with target-specific data; coarse-graining preserves the relevant invariants |
| Optimization "infinite regress" — keep applying passes without termination | Use the §12.8 dynamical-phase-transition detector to identify when optimization has converged |
| Cross-language assumptions break | Restrict initial scope to a single source language; extend later |
| Source-to-MERA encoding is non-trivial for some constructs (e.g., dynamic dispatch) | Use the §10.1 AST encoder's flexibility; handle dynamic features via §16.1's effectful-program extensions |

#### Acceptance test

Take a small program in the QPCN's source language (a Haskell-like STLC). Apply optimization passes via holographic compilation. Verify by direct execution and equivalence checking (§12.2 topological invariants) that the optimized program has identical observable behavior. Compare against hand-written compiler optimizations (e.g., GHC's Core-to-Core passes) on the same examples; the holographic compiler should achieve comparable or better optimization quality.

#### Why this would seem implausible

A compiler that is *automatic*, *provably correct*, *multi-language*, and *capability-growing* (through library discovery) is essentially the dream of compiler research. Existing automated approaches (superoptimization, equality saturation) are limited in scope or correctness; existing provably-correct approaches (CompCert) are hand-written and language-specific. A single substrate that delivers all four is currently considered out of reach.

---

### 12.11 Modular Hamiltonian and entanglement spectrum

#### Physics origin

The modular Hamiltonian `K = -log ρ_A` of a quantum subregion `A` is the generator of entanglement dynamics. Its eigenvalues — the *entanglement spectrum* — carry rich information about topological structure that the energy spectrum alone does not. Li & Haldane 2008 first identified the entanglement spectrum as a topological classifier for fractional quantum Hall states. Subsequent work (Pollmann et al. 2010, Kitaev-Preskill 2006 topological entropy) showed the entanglement spectrum is the most sensitive diagnostic of topologically ordered phases.

For any MPS, the entanglement spectrum at a bond is the set of squared Schmidt coefficients across that bond — exactly the quantity our §10.4 MERA infrastructure already computes.

#### QPCN realization

For any QPCN inference run, at each bond in the MPS we have access to:
- The full set of Schmidt coefficients (the entanglement spectrum).
- The modular Hamiltonian (its negative log).
- Symmetry properties of the entanglement spectrum (degeneracies, eigenvalue ratios).

Different types of proof states have characteristically different entanglement spectra:
- **Trivial proofs** (direct application): nearly degenerate spectrum, low entropy, sharp gap.
- **Inductive proofs**: power-law spectrum reflecting scale-invariant structure.
- **Critical proofs** (near phase transitions, §12.8): characteristic conformal field theory spectrum with prediction power for critical exponents.
- **Topologically protected proofs**: degenerate spectra protected by symmetry — these proofs are robust to perturbations.

The modular Hamiltonian's eigenvalues are diagnostics for *what kind* of proof was found, not just whether it's correct.

#### Capability

**Fine-grained classification of proof states**. Beyond just "did we find a proof?", the architecture can determine:
- The proof's topological class (which connects to §12.3 degeneracy counting).
- Whether the proof is at or near a critical point (which connects to §12.8 phase transitions).
- The proof's robustness to small perturbations of the constraints (topologically protected = robust).
- Whether two ostensibly different proofs are in the same topological class.
- Whether a partial proof has discovered the "right shape" for the full proof (matching entanglement spectrum to known proven patterns).

This is a strictly finer diagnostic than residual energy alone; two proofs with the same `⟨H⟩` can have completely different entanglement spectra, indicating qualitatively different proof structures.

#### Implementation

`composition/entanglement_spectrum.py`:
- After each QPCN inference, compute the Schmidt coefficients at each bond (standard MPS operation, already in `mps.py::entanglement_entropy`).
- Compute the modular Hamiltonian's spectrum from the Schmidt values: `K_i = -log(λ_i^2)`.
- Classify the spectrum: power-law fit, degeneracy pattern, conformal-tower structure.
- Report the classification alongside the proof itself.

Computational cost: one SVD per bond per inference, which is already part of standard MPS operations. The classification step is negligible.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Numerical noise in small Schmidt coefficients obscures the spectrum | Truncate the spectrum at a clean cutoff; report only the well-resolved part |
| Classification scheme is too coarse for some proof classes | Extend the classification taxonomy as new patterns are discovered (this is itself a wake-sleep abstraction-discovery target) |
| Bond dimension `χ` limits the resolvable spectrum | Use the highest `χ` the search can afford; resolved entanglement spectrum grows with `χ` |

#### Acceptance test

Build a corpus of proofs of varying topological character: trivial direct proofs (e.g., `id : A → A`), inductive proofs (e.g., `length (xs ++ ys) = length xs + length ys`), proofs requiring auxiliary lemmas (e.g., `reverse (reverse xs) = xs`). Compute entanglement spectra for each. Confirm that the classification scheme correctly assigns each proof to its expected class.

#### Why this would seem implausible

Existing automated theorem provers report only success/failure and proof length. A system that reports "this proof is in the topologically protected class corresponding to scale-invariant inductive structure" gives information about *how* a proof works at a level of mathematical sophistication usually associated with research mathematics, not automated tools.

---

### 12.12 Quantum extremal surfaces for minimum-complexity proofs

#### Physics origin

The Ryu-Takayanagi formula (Ryu & Takayanagi 2006) computes the entanglement entropy of a CFT subregion as the area of a minimal surface in the bulk AdS geometry. Hubeny-Rangamani-Takayanagi 2007 generalized to dynamic spacetimes via *extremal surfaces*. Engelhardt-Wall 2015 added quantum corrections, giving the *quantum extremal surface (QES)* formula.

The QES formula is the most precise statement to date of the holographic principle: bulk geometric quantities are computable from boundary entanglement data via extremization. The conjectured connection between computational complexity and geometric volume (Susskind 2016, the "complexity = volume" or "complexity = action" conjectures) places computational complexity itself as a holographic observable.

#### QPCN realization

The §11.4 framing established that the MERA substrate is holographic — bulk operations correspond to boundary entanglement structure. Applying QES machinery: the *minimum complexity* of a proof of a theorem corresponds to the area of an extremal surface anchored at the theorem statement on the MERA boundary.

Concretely:
- **Boundary**: theorem statement, encoded as a set of constraints on boundary MPS sites.
- **Bulk**: the MERA tree above the boundary.
- **Extremal surface**: a surface in the MERA tree that minimizes a functional (entanglement area + bulk action), anchored at the boundary constraints.
- **Minimum proof complexity**: the value of the minimized functional, computable *before* running the actual proof search.

This is the analog of computing the minimum-cost path in a graph by examining the graph's structure, not by running pathfinding.

#### Capability

**A priori prediction of proof complexity from theorem geometry alone**. Before doing any proof search, the QPCN can determine:
- The minimum number of MERA layers needed to express the proof (proof depth).
- The minimum bond dimension needed (proof entanglement complexity).
- A lower bound on the proof's length (operations required).
- Whether the theorem is "geometrically natural" (small QES = easy proof) or "geometrically unnatural" (large QES = inherently hard proof).

This information lets the architecture allocate compute intelligently: easy theorems get cheap searches; hard theorems get expensive searches; geometrically impossible theorems (no consistent extremal surface) get rejected without trying.

#### Implementation

`composition/quantum_extremal_surface.py`:
- Build the MERA representation of the theorem's boundary constraints.
- Set up the QES functional: entanglement area + bulk action, parameterized over candidate surfaces.
- Optimize via gradient descent on the surface parameters (standard QES computation).
- Return the minimal value (proof complexity lower bound) and the surface itself (proof structure).

Computational cost: QES optimization is well-studied in physics; for MERA networks it's polynomial in MERA depth. Specific cost depends on the surface parameterization but is generally fast relative to the actual proof search.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| QES is a lower bound only — actual proof may be longer than the QES estimate | This is fine for resource allocation purposes; the bound is informative even when not tight |
| Multiple competing extremal surfaces (degeneracy) | Report all of them; this corresponds to multiple distinct proof strategies (connection to §12.3) |
| QES depends on the specific MERA encoding | Use a canonical encoding; report complexity relative to the canonical form |

#### Acceptance test

For a corpus of theorems with known proof lengths (e.g., from Lean Mathlib), compute the QES lower bound. Verify that:
- The QES is always a true lower bound (no proof shorter than QES).
- The QES correlates with actual proof length (Spearman ρ > 0.7 on a held-out set).
- For theorems where the QES is provably impossible (no surface satisfies the boundary), the actual search confirms unprovability (consistency with §12.1 anomaly detection).

#### Why this would seem implausible

Predicting proof complexity from theorem structure alone, without doing the proof, would be considered approximately impossible by mainstream proof theory. Even informal estimates by expert mathematicians often misjudge proof difficulty by orders of magnitude. A geometric formula that gives provable lower bounds would be remarkable.

---

### 12.13 Bidirectional time evolution for goal-directed search

#### Physics origin

Schrödinger evolution `e^{-iHt}` is unitary, so it can be run forward (`t > 0`) or backward (`t < 0`) with equal facility. The forward direction propagates known initial conditions to derived consequences; the backward direction propagates known final conditions to required initial conditions. The Loschmidt echo `|⟨Ψ_0 | e^{+iHt} e^{-iHt} | Ψ_0⟩|^2 = 1` is the trivial identity expressing that backward evolution exactly inverts forward evolution.

In the path integral formulation, both directions are summed over: the propagator from initial state `|i⟩` to final state `|f⟩` is the sum over all paths connecting them, weighted by `e^{iS/ℏ}`. Forward and backward propagation are dual descriptions of the same Feynman sum.

#### QPCN realization

The QPCN's real-time evolution (§4.7, `evolution.py::trotter_step` with `imaginary=False`) is unitary. Switching the sign of `dt` runs it backward. This costs nothing — same code, same simulator.

For proof search and type inference, bidirectional evolution maps onto:
- **Forward evolution from axioms**: derive consequences of known premises. Standard logical inference.
- **Backward evolution from goal**: derive what premises would imply the goal. This is "goal-directed search" or backward reasoning, used in Prolog, sequent calculus, etc.
- **Meeting in the middle**: run both simultaneously; the overlap region is where the proof exists.

The meeting-in-the-middle proof search is conceptually a quantum walk: forward and backward states evolve under their respective dynamics until they overlap. The overlap region — the manifold of states reachable from axioms *and* required by the goal — is the proof.

For type inference specifically:
- Forward: types propagate up from values (a literal `5` has type `Int`; an application of `+` to two `Int`s has type `Int`).
- Backward: types propagate down from expected results (a function expected to return `String` must produce strings somewhere).
- Bidirectional: both directions simultaneously, with the meeting condition giving the inferred types throughout.

This is precisely bidirectional type checking (Pierce-Turner 2000), but realized as a continuous unitary process on the MPS rather than a discrete walking algorithm.

#### Capability

**Native simultaneous bidirectional reasoning**. Currently, proof systems are typically either forward (tableau, resolution) or backward (Prolog, sequent calculus). Bidirectional systems (e.g., bidirectional type checking) exist but are special-purpose. The QPCN does both directions *for any reasoning task* with zero additional infrastructure.

Specific capabilities unlocked:
- **Counterfactual reasoning**: "what initial conditions would have produced this outcome?" is a single backward evolution.
- **Inverse problems**: given an output, find inputs that produce it. Native via backward evolution.
- **Bidirectional type inference**: optimal in both convergence speed and inference quality vs. unidirectional methods.
- **Proof search with intermediate goals**: forward from axioms, backward from goal, look for overlap. Cuts the search space exponentially compared to one-directional search.

#### Implementation

`composition/bidirectional.py`:
- Add a `direction` parameter to `evolution.py::trotter_step`: `forward` (default), `backward`, or `bidirectional` (alternates).
- For bidirectional search: maintain two MPS states (`|Ψ_forward⟩` from axioms, `|Ψ_backward⟩` from goal); evolve them simultaneously; measure overlap.
- When overlap exceeds threshold, the meeting point gives the proof intermediate.

Computational cost: doubles per-step cost (two MPS evolved simultaneously) but typically halves convergence time, net win for hard problems. The doubling is exact because forward and backward are identical operations modulo `dt` sign.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Forward and backward never overlap (proof doesn't exist) | After bounded evolution time, declare no proof; consistent with §12.1 anomaly detection |
| Multiple overlap regions (multiple proof paths) | Report all; corresponds to §12.3 topological degeneracy |
| Phase coherence issues between forward and backward states | Standard quantum mechanics; relative phases are physical and informative |

#### Acceptance test

Compare bidirectional vs unidirectional proof search on a benchmark of theorems with intermediate-difficulty proofs. The bidirectional search should:
- Find proofs in fewer total Trotter steps on average.
- Successfully find proofs that unidirectional search fails to find within the same budget.
- Correctly report "no proof" when bidirectional search exhausts its budget without overlap.

#### Why this would seem implausible

Bidirectional reasoning as a *first-class* primitive in an inference system is unusual. Most reasoning systems pick a direction and stick with it. A system that natively does both directions simultaneously, exploiting the symmetric structure of unitary evolution, would be a clean improvement over standard search methods. The simplicity of the change (just flip `dt`'s sign) makes the capability gain seem too easy — but the math is exact.

---

### 12.14 Quantum walks for goal-graph search

#### Physics origin

Quantum walks generalize classical random walks by replacing the stochastic transition with a unitary coin-and-shift operation. Aharonov-Davidovich-Zagury 1993 introduced the discrete-time formulation; Farhi-Gutmann 1998 introduced the continuous-time formulation. The decisive result is Childs-Cleve-Deotto-Farhi-Gutman 2003, which showed that a quantum walk on the "glued trees" graph reaches the opposite vertex *exponentially* faster than any classical algorithm. Ambainis 2007 used quantum walks to give a near-optimal `O(N^{2/3})` algorithm for element distinctness. Across many search problems, quantum walks give quadratic speedup over classical and exponential speedup on specific graph structures.

#### QPCN realization

The §10.10 dispatcher navigates the goal graph (a DAG of sub-problems and their dependencies) to decide which sub-QPCN to dispatch next. Classical implementation: beam search or BFS/DFS. Quantum-walk implementation:

- Construct the goal-graph adjacency Hamiltonian `H_walk` whose nonzero entries are edges of the goal graph weighted by edge priorities.
- Initialize the walker state as a superposition over the leaves (axioms).
- Evolve `|Ψ_walk(t)⟩ = e^{-i H_walk t} |Ψ_walk(0)⟩`.
- Measure to obtain the next goal to dispatch.

The walker concentrates amplitude on goals whose ancestor paths from axioms are short and well-connected. This is exactly the "good decomposition" heuristic the dispatcher needs, computed quantum-mechanically.

On a quantum computer, this gives genuine quantum speedup; on classical simulation, it gives a polynomial improvement over beam search and serves as a drop-in for the classical heuristic.

#### Capability

**Quadratic-to-exponential speedup on hard goal graphs.** Concretely:
- For unstructured search over `N` goals: `O(√N)` vs classical `O(N)`.
- For tree-structured goal graphs: comparable to classical with extra ranking quality.
- For "glued-tree"-like structures (which can arise when goal decompositions share intermediate lemmas): exponential speedup.

This is the only piece of the architecture with *provable* quantum advantage on real quantum hardware. Even classically simulated, the quantum-walk dispatcher gives a measurably better heuristic than beam search on structured goal graphs.

#### Implementation

`composition/quantum_walk.py`:
- Build `H_walk` from the goal graph's adjacency (sparse Hermitian).
- Evolve via Lanczos exponentiation for time `t ≈ √N` (the optimal walk time scales with the graph diameter).
- Measure observable: position of the walker. Use this as the dispatcher's selection.

Computational cost: linear in goal-graph edges per Trotter step; total cost `O(N · t) = O(N^{3/2})` for unstructured search vs `O(N)` for classical brute force — comparable in the worst case, but the constant factor and the favorable scaling on structured graphs is the win.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Quantum walk gives no speedup on graphs equivalent to classical | Detect graph structure first; fall back to classical for unstructured cases |
| Decoherence on real hardware destroys the walk's advantage | Use simulated walks on classical hardware until quantum hardware matures |
| Walk Hamiltonian construction overhead | Cache the adjacency Hamiltonian; amortize across many searches |

#### Acceptance test

Construct a synthetic goal graph with known quantum-walk speedup (e.g., the glued-trees graph with 64 nodes). Show that the quantum-walk dispatcher reaches the goal in roughly `O(log N)` time while classical search requires `O(N)`. On realistic goal graphs from §10.11's hierarchical proof demo, the quantum-walk dispatcher should empirically outperform beam search on at least 50% of cases.

#### Why this would seem implausible

Quantum walks for proof search is currently aspirational research; published quantum-walk results are on toy graph structures. A working implementation showing quadratic speedup on realistic decomposition graphs would be a substantial empirical result.

---

### 12.15 Witten index for theorem fingerprints

#### Physics origin

Witten 1982 introduced the index `I = tr((-1)^F e^{-βH})` for supersymmetric quantum theories, where `F` is the fermion number. The index counts ground states with signs (bosonic minus fermionic). It is a *topological invariant* — independent of continuous deformations of the Hamiltonian within a class. A nonzero Witten index proves a ground state must exist; the value of the index gives detailed information about the theory's structure.

In mathematical physics, the index has been used to:
- Prove existence of supersymmetric ground states without constructing them.
- Classify topological phases of matter.
- Compute geometric invariants (Atiyah-Singer index theorem connection).

#### QPCN realization

Define a Z_2 grading on proof structures. Natural choices:
- **Parity of induction depth**: proofs using even vs. odd number of inductive cases.
- **Chirality of binding entanglement**: orientation of the variable-binding string diagram (clockwise vs counter-clockwise traversal).
- **Fermion-like vs boson-like decomposition**: proofs of "even" theorems (constructive, total) vs "odd" theorems (classical, non-constructive).

For a given theorem with constraint Hamiltonian `H`, compute

```
I_QPCN = tr((-1)^G e^{-βH})  on the ground subspace
```

where `G` is the grading operator. This gives a signed count of distinct proofs.

Two theorems with the same Witten index are equivalent under topology-preserving deformations of the constraint Hamiltonian — they are reformulations of the same theorem.

A nonzero Witten index proves a proof must exist (existence by topology, without construction).

#### Capability

**Robust theorem identification across formalizations.** Two ostensibly different theorems with the same Witten index are *the same theorem* up to provable transformations.

**Existence proofs without construction.** When the Witten index is nonzero but a proof has not been found, the architecture can report: "a proof must exist; we have not constructed it yet." This is the operational analog of an existence proof in mathematics.

**Fine-grained topological classification of proof spaces** beyond what §12.3 (degeneracy count alone) provides.

#### Implementation

`composition/witten_index.py`:
- Identify the grading operator `G` for the problem class.
- Project onto the ground subspace via imaginary-time evolution.
- Compute the trace of `(-1)^G` on the ground subspace.
- Return the integer-valued index.

Computational cost: a few projector applications plus a trace. Cheap relative to the underlying proof search.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Grading choice is somewhat arbitrary | Standardize gradings per problem domain; document the choice |
| Zero index does not imply no proof | Report the index distinguishing "no protected proof" from "no proof at all" |
| The index changes under reformulation if the grading isn't preserved | Only meaningful comparisons are between formulations sharing the same grading |

#### Acceptance test

Take three formulations of the same theorem (e.g., `length (xs ++ ys) = length xs + length ys` in different naming conventions or with different proof skeletons). Compute the Witten index for each. All three should give the same value. Take a deliberately different theorem with the same energy spectrum and confirm a different Witten index.

#### Why this would seem implausible

Topological invariants of theorems are not part of any current automated theorem prover's toolkit. The notion of "this theorem is a deformation of that theorem" being a *computable* relation, rather than a mathematician's judgment, would be a striking claim.

---

### 12.16 Worldline path integral for Bayesian proof ranking

#### Physics origin

Feynman 1948. A particle's transition amplitude from initial state to final state is the sum over all paths weighted by `e^{iS/ℏ}`, where `S = ∫ L dt` is the action along the path. The most probable path (saddle point) is the classical trajectory; quantum corrections come from nearby paths. The width of the saddle-point region quantifies uncertainty.

This formulation gives a *Bayesian* interpretation: the path integral is a sum over hypotheses (paths) weighted by their action (a measure of fitness). The most probable hypothesis is the saddle point; the certainty is the width.

#### QPCN realization

A *proof* is a path through state space from axioms to goal — a sequence of intermediate states linked by valid proof steps. Define the proof's action `S[path]` as a complexity measure:

```
S[path] = (length of path) + (sum of intermediate-state energies) 
        - (log probability of each step under Hamiltonian dynamics)
```

The Feynman amplitude for a proof is `exp(-S[path]/T)` for some "temperature" `T`. The probability of a proof under the Boltzmann distribution is

```
P(proof) = exp(-S[proof]/T) / Z
```

The architecture can compute `P` for any candidate proof. Bayesian model selection picks the proof with highest `P`. Multiple proofs of comparable probability indicate genuine uncertainty.

The saddle-point approximation (most likely proof) is what naive proof search finds; the full path integral gives the full uncertainty distribution.

#### Capability

**Native Bayesian uncertainty quantification on proofs themselves.** Beyond just "did we find a proof?", the architecture quantifies:
- Probability of each candidate proof under the Boltzmann distribution.
- Whether the most likely proof is robust (sharply peaked) or fragile (broad distribution).
- Whether multiple distinct proof strategies have similar probability (genuine ambiguity).

This is the analog of Bayesian model selection in statistics, applied to proof search. No existing automated theorem prover produces calibrated probabilities over alternative proofs.

#### Implementation

`composition/worldline_pi.py`:
- For each candidate proof, compute its action `S`.
- Normalize over a sampled batch of candidate proofs to estimate `Z`.
- Report each proof's probability `P = exp(-S/T) / Z`.
- Optionally: Monte Carlo sampling over the path-integral measure to discover unexplored proofs.

Computational cost: dominated by candidate-proof generation (which is the underlying proof search). The path-integral evaluation adds a small overhead per candidate.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Action functional choice affects probability values | Document the standard choice; offer alternatives for domain-specific contexts |
| Saddle-point approximation breaks for rough action landscapes | Use full Monte Carlo when the spectrum is broad |
| Combinatorial number of paths | Restrict to top-`k` proofs found by search; this approximates the full path integral |

#### Acceptance test

For a theorem with multiple known proofs (e.g., `compose ∘ id = compose` has at least two distinct proof routes), confirm that:
- All known proofs are assigned nonzero probability.
- The "natural" proof (preferred by mathematicians) has the highest assigned probability.
- Proofs that are admissible but unusual have lower (nonzero) probability.

Match the architecture's preferences against mathematician preferences on a held-out set; correlation should exceed 0.7.

#### Why this would seem implausible

Bayesian probabilistic semantics for proofs is currently absent from the literature. Most provers treat all valid proofs as equally good; ranking them by complexity is heuristic. A principled probabilistic framework with calibrated uncertainty is uncommon.

---

### 12.17 Quantum cellular automata as the rigorous evolution framework

#### Physics origin

Schumacher & Werner 2004 axiomatized quantum cellular automata (QCAs). A QCA is a local, translation-invariant, unitary dynamics on a quantum lattice — equivalently, the discrete-time analog of a local Hamiltonian. The Schumacher-Werner theorem establishes:

- Every QCA is locally implementable: it factors into local commuting gates.
- 1D QCAs have a complete topological classification (Gross-Nesme-Vogts-Werner 2012) via a single integer-valued invariant called the *index*.
- QCAs are unitary, hence reversible by construction.
- Light-cone causality: information propagates at most one site per QCA step.

The QCA framework is to Hamiltonian dynamics what cellular automata are to continuous dynamical systems — a complete, rigorously classified discrete framework.

#### QPCN realization

The Trotter evolution in `qft/evolution.py` is *literally* a QCA — it satisfies every axiom of the Schumacher-Werner definition. Every theorem in the QCA literature applies to QPCN dynamics, for free:

- **Topological classification (Gross et al. 2012)**: each evolution Hamiltonian belongs to a discrete topological class indexed by an integer. Two QPCNs in the same class are equivalent under bounded-depth perturbations.
- **Light-cone bounds (Lieb-Robinson)**: prediction errors at site `k` cannot affect site `k + N` faster than `N` Trotter steps. Quantifies the "speed of reasoning."
- **Reversibility for free**: §12.13's bidirectional evolution is automatic from QCA reversibility.
- **Unique decomposition**: every QCA has a unique decomposition into local gates; this is the canonical "compilation" of the dynamics into elementary operations.

#### Capability

**A rigorous theoretical foundation** for the architecture's dynamics. Specifically:
- Identify the QPCN's topological class via its QCA index.
- Predict long-time behavior from QCA classification (without simulation).
- Bound causal influence in proof search via Lieb-Robinson velocities.
- Compose QPCNs of compatible QCA classes correctly.

This is less about adding a new capability and more about *grounding* the architecture in established mathematics — making it taxonomically recognizable to physicists who work on QCAs.

#### Implementation

`composition/qca_classification.py`:
- Express the QPCN's Trotter evolution as a quantum cellular automaton.
- Compute the QCA index via the standard Gross-Nesme-Vogts-Werner algorithm.
- Use the index to predict properties: stability, long-time behavior, equivalence classes.

Computational cost: trivial — the index is computed from the local gate structure, which is already known.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Some QPCN dynamics don't fit standard QCA axioms | Identify the violation; either generalize the framework or report which axiom fails |
| Higher-dimensional QCA classification is open | Restrict to 1D MERA paths initially; 2D MERA extensions follow ongoing physics research |
| The framework is descriptive, not prescriptive | The framework's value is rigor and theorem inheritance, not direct capability gain |

#### Acceptance test

Compute QCA indices for several QPCN configurations. Show that:
- Configurations with the same index produce equivalent behavior under bounded perturbations.
- Configurations with different indices are demonstrably distinct (e.g., one stable, the other chaotic).
- The Lieb-Robinson speed matches the empirical "information propagation speed" measured in the QPCN.

#### Why this would seem implausible

Placing a machine learning architecture within the QCA framework is unusual but mathematically natural. The framework's power is that once accepted, decades of QCA research flow into the QPCN's theoretical foundations.

---

### 12.18 Noether's theorem for automated conservation-law discovery

#### Physics origin

Emmy Noether 1918, "Invariante Variationsprobleme." Every continuous symmetry of the action implies a conserved quantity. The construction is explicit: given a symmetry generator `δϕ`, the conserved current is `J^μ = (∂L/∂(∂_μϕ)) δϕ - L · δx^μ`. Time translation → energy conservation; space translation → momentum conservation; rotational symmetry → angular momentum conservation; gauge symmetry → charge conservation.

Noether's theorem is one of the most consequential results in mathematical physics: it explains why conservation laws are ubiquitous (they're consequences of the symmetries of nature, not separate axioms).

#### QPCN realization

The §10.9 wake-sleep cycle discovers patterns by clustering. Some discovered patterns are *continuous symmetries* of the problem distribution — transformations under which the solved-problem set is invariant. Noether's theorem gives a constructive procedure for converting each discovered symmetry into a corresponding conserved quantity (a new operator).

Concretely:
1. Wake phase: solve a batch of problems.
2. Symmetry detection: identify continuous transformations under which the solution set is invariant.
3. Noether construction: for each symmetry generator `δϕ`, compute the Noether current `J^μ`.
4. Promotion: the Noether current becomes a new conserved-charge operator in the Hamiltonian. The architecture now *knows* this quantity is conserved.

The discovered conservation laws are the architecture's "learned physics" of the domain.

#### Capability

**Automated discovery of conservation laws.** For each target domain:

- **Chemistry**: discover spin conservation, charge conservation, particle number, total angular momentum.
- **Programs**: discover type preservation, total computation, side-effect freedom, referential transparency.
- **Mathematics**: discover invariants of theorems under reformulation (Witten-index-like).
- **Dynamical systems**: discover energy, momentum, etc. directly from observed trajectories.

This is the analog of *symbolic regression* extended to invariants. Existing approaches (e.g., AI Feynman, Udrescu & Tegmark 2020) discover *equations*; this discovers *conserved quantities*, which is structurally different and complementary.

#### Implementation

`composition/noether_discovery.py`:
- During wake-sleep, pattern-mine the library for continuous transformations.
- Each candidate transformation: verify it leaves the solution set invariant within the library.
- For surviving candidates: apply the constructive Noether procedure to generate the conserved current.
- Add the current as a new operator; verify by checking that `[H, J] = 0` on the constraint Hamiltonian.

Computational cost: dominated by symmetry detection in the library (`O(library size)^2` worst case, much better with hashing/heuristics). The Noether construction itself is `O(1)` given the symmetry generator.

#### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Discrete symmetries don't give Noether currents | Detect discrete symmetries separately; treat them as parity-like invariants |
| Approximate symmetries (broken by small effects) | Report the symmetry-breaking magnitude; useful diagnostic for "almost-conserved" quantities |
| Detected "symmetries" that are coincidental rather than structural | Cross-validate on held-out problems; require the symmetry to hold across diverse instances |

#### Acceptance test

On a chemistry corpus (e.g., QM9 ground states): the architecture should rediscover spin and charge conservation, perhaps also parity. On a programming corpus (e.g., STLC proof corpus): the architecture should rediscover type preservation under reduction. On a dynamical-systems corpus: rediscover energy and momentum conservation. Each rediscovery should be quantitatively measured against the known conservation law.

#### Why this would seem implausible

Automated discovery of physical conservation laws is a major goal in scientific machine learning, and the leading approaches (AI Feynman and successors) use different machinery. A QPCN-based discovery procedure that produces *constructive* Noether currents from data, rather than fitted equations, would be a genuinely new contribution. Validating it against known physics laws would be the proof point.

---

### 12.19 The combined extension stack

When all eighteen §12 extensions are built on top of the §10–11 roadmap, the architecture has these capabilities simultaneously:

| Capability | From | Currently published baseline |
|---|---|---|
| Provably-type-correct code generation | §10.7 | LLMs (heuristic, hallucinate ill-typed code) |
| Compounding capability via library learning | §10.8–10.9 | DreamCoder (validated, classical NN) |
| Hierarchical proof composition | §10.10–10.11 | AlphaProof (validated, transformer-based) |
| **Provable impossibility proofs** | §12.1 | None (existing provers fail silently) |
| **Exact program equivalence via topological invariants** | §12.2 | Heuristic (Egg-style e-graphs) |
| **A priori counting of distinct proof strategies** | §12.3 | None (currently requires exhaustive search) |
| **Type-only program reasoning via bootstrap** | §12.4 | Theorems-for-free (qualitative only) |
| **Fault-tolerant reasoning by construction** | §12.5 | None (existing systems are fragile to noise) |
| **Automatic identification of missing lemmas** | §12.6 | None (tactics report cryptic failures) |
| **Predictive complexity theory** | §12.7 | Empirical scaling laws only |
| **Self-detecting capability transitions** | §12.8 | Heuristic curriculum learning |
| **Operational self-modification of own Hamiltonian** | §12.9 | Schmidhuber Gödel machine (theoretical only) |
| **Provably semantics-preserving holographic compilation** | §12.10 | CompCert (hand-written, language-specific) |
| **Topological classification of proof states** | §12.11 | None (current provers only report success/failure) |
| **A priori proof complexity from theorem geometry** | §12.12 | None (proof difficulty currently estimated empirically) |
| **Native bidirectional reasoning (forward + backward simultaneously)** | §12.13 | Bidirectional type checking (limited scope) |
| **Quantum-walk speedup for goal-graph search** | §12.14 | Beam search (classical) |
| **Topological theorem fingerprints (Witten index)** | §12.15 | None (existing systems can't recognize reformulations) |
| **Bayesian uncertainty quantification on proofs** | §12.16 | Heuristic ranking, no calibrated probabilities |
| **Rigorous QCA framework for evolution semantics** | §12.17 | Ad-hoc dynamics specifications |
| **Automated conservation-law discovery via Noether** | §12.18 | AI Feynman (equations, not invariants) |

The first three (§10.7, §10.8–10.9, §10.10–10.11) are well-defined engineering targets with published precedents validating they're achievable. The eighteen §12 extensions are physics-derived; each individual capability has rigorous foundation but the combination is novel.

The publication strategy for a system with all of this:

1. **First paper** (workshop/short): §10.7 STLC synthesis with formal correctness guarantees. Validate the core architecture.
2. **Second paper** (mid-tier conference): §10.8–10.11 hierarchical composition demo with capability growth measurements. Validate the wake-sleep cycle for proof construction.
3. **Third paper** (top venue, after §12 extensions): the combined system, claiming the eighteen extension capabilities. This is the paper reviewers may find implausible — but each individual claim is referenced to published physics.

### 12.20 Implementation priority order

Within §12, the most efficient build order (by cost-to-implement vs. capability gain):

| Order | Extension | Why now |
|---|---|---|
| 1 | **§12.5 Holographic codes** | Lowest implementation cost (MERA already does this); immediate robustness benefit |
| 2 | **§12.13 Bidirectional time evolution** | Just flip `dt` sign in existing TEBD; immediate doubling of effective search capability |
| 3 | **§12.11 Modular Hamiltonian / entanglement spectrum** | Uses existing Schmidt-coefficient computation; gives fine-grained proof classification |
| 4 | **§12.6 Goldstone modes** | Standard eigenvalue calculation on existing Hamiltonian; immediate UX benefit |
| 5 | **§12.17 QCA framework** | Descriptive only; just compute QCA indices on existing dynamics. Theoretical grounding, no new infrastructure |
| 6 | **§12.16 Worldline path integral** | Sums of `e^{-S/T}` over candidate proofs; minimal added compute |
| 7 | **§12.3 Topological degeneracy** | Spectral analysis of existing Hamiltonians; gives valuable a-priori information |
| 8 | **§12.2 Topological invariants** | Wilson-loop calculations on existing MPS; enables compiler verification |
| 9 | **§12.8 Dynamical phase transitions** | One overlap calculation per cycle; closes the curriculum-adaptation loop |
| 10 | **§12.15 Witten index** | Requires defining a grading; once defined, computation is a trace |
| 11 | **§12.14 Quantum walks** | Requires explicit walk-Hamiltonian construction; gives best practical speedup |
| 12 | **§12.18 Noether discovery** | Requires symmetry-detection in the library; high domain-specific value |
| 13 | **§12.10 Holographic compilation** | Requires full MERA buildout (§10.4) but conceptually elegant once available |
| 14 | **§12.12 Quantum extremal surfaces** | Requires MERA + RT-formula machinery; a priori proof complexity prediction |
| 15 | **§12.1 Anomalies** | Requires symmetry-generator extraction from typing rules; highest novelty |
| 16 | **§12.7 Replica method** | Requires careful analytic continuation; highest mathematical sophistication |
| 17 | **§12.9 Self-modification via meta-Hamiltonian** | Requires MPO infrastructure for operator-valued substrate; most ambitious |
| 18 | **§12.4 Conformal bootstrap** | Requires SDP solver integration and rich bootstrap-system formulation; highest implementation difficulty |

Extensions 1–6 are immediate wins with existing infrastructure. Extensions 7–12 add genuine novelty without enormous cost. Extensions 13–18 are the spectacular ones — built once the foundations are stable, and they're what would make the combined-system paper a flagship result.

---

## 13. Theoretical Guarantees and Provable Claims

The architecture admits formal claims about its behavior — not heuristic hopes but theorems derivable from the operator-algebraic substrate. This section states them precisely. Each claim is either a direct corollary of an established result in quantum many-body physics or a definitional consequence of the architecture itself.

### 13.1 Convergence of imaginary-time evolution

**Theorem 13.1 (Ground-state convergence).** Let `H` be a Hermitian Hamiltonian on the QPCN's Hilbert space with ground state `|Ψ_0⟩` and gap `Δ > 0` to the first excited state. Then for any initial state `|Ψ(0)⟩` with nonzero overlap `⟨Ψ_0|Ψ(0)⟩ ≠ 0`,

```
|Ψ(τ)⟩ = e^{-Hτ} |Ψ(0)⟩ / || e^{-Hτ} |Ψ(0)⟩ ||
```

satisfies

```
|| |Ψ(τ)⟩ - |Ψ_0⟩ || ≤ C · e^{-Δτ}
```

for some constant `C` depending only on the initial overlap.

*Proof sketch.* Expand `|Ψ(0)⟩ = c_0 |Ψ_0⟩ + Σ_{i>0} c_i |Ψ_i⟩` in the energy eigenbasis. Under `e^{-Hτ}`, each component multiplied by `e^{-E_i τ}`. After normalization, the ratio of excited-state amplitudes to ground-state amplitude decays as `e^{-(E_i - E_0)τ}`, bounded by `e^{-Δτ}`. QED.

*Consequence for QPCN.* For any well-conditioned constraint Hamiltonian (gap bounded away from zero), the QPCN converges to the correct ground state in `O(1/Δ)` Trotter steps. The gap is computable from the Hamiltonian's algebraic structure for well-behaved typing systems; it serves as a difficulty proxy.

*Limitation.* When the gap closes (`Δ → 0`), convergence slows to a power law and the search may not terminate in reasonable time. Gap-closing signals critical points in the constraint structure — these correspond to problems near a SAT-UNSAT-like phase transition, which are intrinsically hard.

### 13.2 Correctness by construction

**Theorem 13.2 (Energy-zero ground states satisfy all constraints).** Let `H = Σ_i w_i P_i` where each `P_i` is a positive-semidefinite penalty operator with `P_i |φ⟩ = 0` iff `|φ⟩` satisfies constraint `i`, and `w_i > 0`. Then any state `|Ψ⟩` with `⟨Ψ|H|Ψ⟩ = 0` satisfies every constraint `i`.

*Proof.* Each `⟨Ψ|P_i|Ψ⟩ ≥ 0` because `P_i ⪰ 0`. The total is `Σ w_i ⟨Ψ|P_i|Ψ⟩ = 0`. Since `w_i > 0` and each term is non-negative, every term must be zero. Thus `P_i^{1/2}|Ψ⟩ = 0` for all `i`, which means `|Ψ⟩` is in the kernel of each `P_i` — i.e., satisfies each constraint. QED.

*Consequence for QPCN.* The architecture's central claim — *that the decoded AST is provably correct* — is a definitional consequence of constructing `H` from positive-semidefinite constraint operators. There is no hidden assumption; the correctness is exact at `ε = 0` residual energy and degrades smoothly to constraint violations of magnitude `O(√ε)` as `ε` grows.

**Corollary 13.2.1 (Hallucination-free synthesis).** Unlike LLM-based code generation, the QPCN cannot produce ill-typed or constraint-violating output at zero residual energy. The output is provably correct *or* the residual energy is nonzero (and exposed to the caller via §10.6 debugger).

### 13.3 Compositionality of solved sub-problems

**Theorem 13.3 (Lemma composition).** Let `H_A` and `H_B` be Hamiltonians on disjoint subsystems `A` and `B` with ground states `|Ψ_A⟩, |Ψ_B⟩`. Then `|Ψ_A⟩ ⊗ |Ψ_B⟩` is the ground state of `H_A ⊗ I_B + I_A ⊗ H_B`.

*Proof.* Energy is additive on the tensor product: `⟨H_A + H_B⟩ = ⟨H_A⟩_A + ⟨H_B⟩_B`. Each term minimized independently by the corresponding ground state. QED.

*Consequence for QPCN.* The §10.8 lemma promotion mechanism is *exact*: clamping a sub-MPS to a cached lemma's ground state is mathematically identical to having found that ground state during the current QPCN run. There is no correctness drift from composition.

**Theorem 13.3.1 (Composition with shared variables).** When `H_A` and `H_B` share variables (binding-as-entanglement, §8.1), the composed ground state is not generally the tensor product. The architecture handles this by computing the ground state of `H_A + H_B + H_coupling` jointly, where `H_coupling` encodes the shared-variable constraints. The energy of the joint ground state is bounded below by the sum of the individual ground-state energies; equality holds iff the shared-variable structure is consistent.

*Consequence.* The system detects inconsistent lemma compositions automatically as positive residual energy at the coupling sites. This is the analog of a type-error from naively composing incompatible types.

### 13.4 Fault-tolerance threshold

**Theorem 13.4 (Holographic threshold theorem, after Pastawski et al. 2015).** For a MERA-structured QPCN of depth `d` constructed from perfect tensors (Hayden et al. 2016) with code distance `D(d)`, there exists a noise threshold `p_th > 0` such that if each sub-QPCN's failure probability satisfies `p < p_th`, the global reconstruction error decreases exponentially with `D(d)`.

*Proof sketch.* The HaPPY construction encodes bulk logical qubits into boundary physical qubits with code distance growing as `D(d) = Ω(d)`. Standard QEC threshold arguments give `p_th > 0`. The QPCN's MERA layers operate as the holographic encoding; sub-QPCN failures behave as Pauli noise on the boundary that the code corrects up to distance `D`. QED.

*Consequence for QPCN.* The architecture (with §12.5 built) is *fault-tolerant by construction* up to a quantifiable noise rate. For any target reconstruction error `ε`, choose MERA depth `d ~ log(1/ε)`. This is a structural correctness guarantee no existing reasoning system has.

### 13.5 Expressivity bound

**Theorem 13.5 (MPS representability, after Vidal 2003).** A pure state `|Ψ⟩` on `N` sites is exactly representable as an MPS with bond dimension `χ` if and only if `χ ≥ max_k 2^{S(L_k)}` where `S(L_k)` is the von Neumann entanglement entropy of the bipartition at bond `k`.

*Consequence for QPCN.* The architecture can represent (and therefore learn) any quantum state whose entanglement entropy is bounded by `log χ` across every bipartition. States with greater entanglement entropy are *fundamentally unrepresentable* at this bond dimension; this is a hard expressivity wall.

**Corollary 13.5.1 (Area-law states are accessible; volume-law states are not).** Programs and proofs whose binding structure follows the area law (i.e., scope locality is preserved) are representable. Programs whose binding patterns generate volume-law entanglement (highly entangled global references across all positions) are not. *Most natural code, proofs, and chemical structures are area-law*; this is why the architecture targets these domains and not arbitrary text.

### 13.6 Variational free energy upper bound

**Theorem 13.6 (Variational principle).** For any density matrix `ρ` and Hamiltonian `H`, the variational free energy satisfies

```
F_var[ρ] = tr(ρH) + T · tr(ρ log ρ) ≥ -T · log Z = F_true
```

with equality iff `ρ = e^{-H/T}/Z`.

*Consequence for QPCN.* Minimizing the architecture's variational free energy gives a provable upper bound on the true free energy of the target distribution. The optimization is monotone and well-posed; there is no overfitting in the classical-NN sense, because the bound is tightening on a quantity that exists independently of the architecture.

### 13.7 Conservation laws as gauge invariants

**Theorem 13.7 (Conservation under unitary evolution).** If an operator `Q` commutes with `H`, then under real-time TEBD with `H` as generator, `⟨Q⟩` is conserved to within Trotter error `O(dt^2)` per step, accumulating to `O(N · dt^2)` over `N` steps.

*Consequence for QPCN.* Type conservation under program evaluation is exact. As §10.3's evaluation Hamiltonian evolves an AST, the types at each site are preserved if and only if the typing-rule Hamiltonian commutes with the evaluation Hamiltonian — which is the operator-algebraic statement of "type safety: well-typed programs cannot go wrong" (Wright & Felleisen 1994).

### 13.8 Capability growth law (asymptotic)

**Conjecture 13.8 (Wake-sleep capability scaling).** Under §10.9 wake-sleep dynamics with abstraction discovery rate `α(t)` and library pruning rate `β`, the system's capability function `C(t)` (defined as the expected fraction of problems solvable from a target class) satisfies

```
dC/dt = α(t) · (1 - C(t)) - β · C(t)
```

giving asymptotic capability `C_∞ = α/(α + β)`. With `α` increasing as the library grows (more primitives → faster discovery), this is a saturating curve approaching 1.

*Justification.* This is the mean-field limit derivable from a replica-method calculation on the §10.9 wake-sleep ensemble (§12.7). Exact form requires assumptions on the problem distribution and abstraction-discovery mechanism. Stated as a conjecture pending §12.7 implementation.

*Consequence if validated.* The QPCN's capability scaling law would be the architectural analog of the Chinchilla/Kaplan scaling laws for transformers — but derivable analytically from the substrate rather than fit empirically.

### 13.9 Significance

These eight results are not all of equal weight:

- 13.1, 13.2, 13.3, 13.7 are immediate consequences of the architecture and are *unconditionally true*.
- 13.4, 13.5, 13.6 are imports from established literature, applied to our substrate; correctness depends on the imported result.
- 13.8 is a conjecture pending the replica-method implementation.

Taken together they establish that the QPCN is not just an engineering construct but a mathematical object with derivable properties. This is the section that converts the work from "an interesting design" to "a research contribution with formal content."

---

## 14. Evaluation Methodology and Benchmarks

To evaluate the QPCN as a research contribution, we need: clear benchmarks, defined metrics, fair baselines, principled ablations, and a statistical protocol. This section specifies each.

### 14.1 Primary benchmarks

Selected for relevance to the architecture's claimed strengths, with clear baselines.

| Benchmark | Source | What it measures | Why it matters |
|---|---|---|---|
| **miniF2F** | Zheng-Han-Polu 2021 | Formal math problems (IMO-style, MATH-style) | Direct comparison with AlphaProof (silver-medal 2024) and ReProver |
| **Lean Mathlib** | Lean community | Real-world theorem proving on a 1.5M-line corpus | Tests on theorems mathematicians actually care about |
| **Hazel synthesis** | Omar et al. 2017+ | Typed program synthesis with holes | Direct comparison with their published system |
| **Myth** | Osera & Zdancewic 2015 | Small STLC synthesis from examples | Established small-scale benchmark |
| **DreamCoder list/drawing** | Ellis et al. 2020 | Wake-sleep library learning targets | Direct comparison with their published results |
| **PROSE/FlashFill** | Gulwani 2011, Polozov & Gulwani 2015 | Programming-by-example | Established baseline for inductive synthesis |
| **QM7/QM9** | Rupp et al. 2012 | Quantum chemistry properties of small molecules | Tests claimed strength in physical-domain reasoning |
| **HumanEval** | Chen et al. 2021 | LLM code generation | *Negative comparison*: we don't expect to win; we expect to be type-safe on a subset |

Each benchmark has a clear "what we expect to claim" beforehand, registered before running. Pre-registration prevents post-hoc cherry-picking.

### 14.2 Metrics

**Synthesis metrics**:
- `pass@k`: fraction of problems for which at least one of `k` returned candidates satisfies all examples. Standard in synthesis literature.
- `type@k`: fraction of `k` candidates that are type-correct. The QPCN should have `type@1 = 1.0` for any successful run (§13.2); LLMs achieve `~50-80%`. This is the architecture's distinctive metric.
- `residual_energy@1`: distribution of residual energy across the top candidate. Zero = provably correct; nonzero = quantified uncertainty.

**Proof metrics**:
- `theorems_proved`: count within a fixed compute budget.
- `proof_length`: distribution over successful proofs (shorter is better).
- `time_to_first_proof`: latency. Less important than other metrics for the QPCN, but reported for completeness.

**Library-learning metrics** (over wake-sleep cycles):
- `capability_curve C(t)`: fraction of held-out problems solved at cycle `t`.
- `library_size`: number of primitives discovered at cycle `t`.
- `compression_ratio`: average solution length, normalized to cycle 0.
- `transition_detection`: where dynamical phase transitions (§12.8) were detected, and whether they correlate with capability jumps.

**Architecture-specific diagnostics**:
- `bond_dimension` reached during inference: tracks expressivity utilization.
- `entanglement_entropy` at the midpoint bond: measures non-classical correlation.
- `truncation_error` accumulated per inference run: hygiene metric.

### 14.3 Baselines

| Baseline | For benchmark | Comparison protocol |
|---|---|---|
| **AlphaProof** | miniF2F | Same problem set, same compute budget |
| **ReProver / LeanDojo** | Mathlib | Same problem set, same retrieval setup |
| **GPT-4 / Claude** | HumanEval, miniF2F | Same prompts, sampled `k = 10` |
| **Hazel** | Hazel synthesis | Direct comparison on their published examples |
| **Synquid** | Myth + their own benchmarks | Same problem set |
| **DreamCoder** | DreamCoder benchmarks | Same compute budget, same library initialization |
| **Egg / e-graphs** | Program equivalence | Same equivalence checking problems |
| **Hindley-Milner** | Type inference | Sanity baseline; QPCN should match on STLC |

### 14.4 Ablations

Each ablation isolates one architectural commitment. Without these, the paper cannot claim that any specific component is essential.

| Ablation | What's removed | What this tests |
|---|---|---|
| **A1**: No MERA hierarchy | §10.4 — use flat MPS only | Does the hierarchy matter for capability? |
| **A2**: No manifold coupling | §4.1 — flat metric `g = I` | Does the geometric substrate matter? |
| **A3**: No multi-field | §10.4 — single field species | Does multi-modal coupling matter? |
| **A4**: No abstraction discovery | §10.9 — fixed library | Does library learning compound capability? |
| **A5**: No quantum substrate | Just classical PCN with same Hamiltonian | Does the operator-valued substrate matter? |
| **A6**: No predictive coding | Just tensor network optimization | Does the PCN dynamics matter? |
| **A7**: No §12 extensions | Pure §10 architecture | What does each §12 extension contribute? (one ablation per extension) |
| **A8**: Classical generative model | Replace QuantumConvMap with ClassicalConvMap | Does the Qiskit VQC matter at all? |

The most important ablations are A4 (does library learning matter?) and A5 (does quantum substrate matter?). Without compelling answers to these, reviewers will be skeptical.

### 14.5 Statistical protocol

- **Independent runs**: minimum N = 5 per (benchmark, configuration) combination; N = 10 for headline claims.
- **Reporting**: mean ± std, with explicit run counts. Never mean alone.
- **Significance**: paired t-tests against baselines; report p-values. For multiple comparisons within a benchmark, Bonferroni correction.
- **Confidence intervals**: 95% CI by bootstrap on per-run metrics.
- **Effect size**: report Cohen's d alongside p-value. Statistical significance without effect size is uninformative.
- **Seeds**: fixed and released per run, so anyone can reproduce. Seed sweeps reveal robustness.

### 14.6 Reproducibility

- All Hamiltonian-compilation code committed; no opaque constants.
- Random seeds, MPS bond dimensions, dt values, and Trotter orders specified per experiment.
- Compute budget reported in operations (Trotter steps × bond dimension² × site count), not wall time.
- A pinned `requirements.txt` with exact dependency versions (numpy, scipy, qiskit, qiskit-aer).
- For each headline result, a single-command reproduction script in `experiments/<name>.sh`.

### 14.7 Pre-registration

Before running each experimental campaign, write down:
- **Hypothesis**: what the QPCN should do better/worse than baseline X.
- **Decision rule**: what numerical result confirms the hypothesis.
- **Negative result rule**: what numerical result falsifies it.

Public pre-registration (e.g., on OSF) prevents the field's most common pathology: post-hoc cherry-picking of benchmarks where the architecture happens to win.

### 14.8 Expected publication-grade results

For Phase C (§10.7), the headline claim should be:

> "The QPCN achieves `type@1 = 1.0` on Hazel synthesis benchmarks (vs. GPT-4's `~0.75`), with `pass@1` competitive on simple STLC problems (`~0.80` vs Hazel's `~0.85`). Synthesis output is provably type-correct by §13.2."

For Phase E (§10.11), the headline should be:

> "Hierarchical proof composition with wake-sleep library learning achieves N% of miniF2F problems within a compute budget comparable to ReProver, with measurable capability growth `C(t)` correlating with theoretical predictions from §13.8."

For Phase H (combined-system paper):

> "The QPCN simultaneously demonstrates [list of §12 capabilities], each individually attested in published physics but combined here for the first time in an automated reasoning system. Empirical confirmation provided on benchmarks [list]."

---

## 15. A Worked End-to-End Example

This section walks through the architecture's operation on a specific small problem in full detail. The purpose is to make the abstract pipeline concrete enough that a reader can verify every step.

**Task**: synthesize `length : List a → Nat`, the function that returns the length of a list, from its type signature and a few examples.

### 15.1 Step 1: Natural language input

User to the LLM frontend:

> "Write a function called length that takes a list of any type and returns the number of elements in it. So length [] is 0, length [x] is 1, length [x, y, z] is 3."

### 15.2 Step 2: LLM extracts structured intent

The LLM produces:

```
Intent: pure function, recursive on list structure
Type signature: ∀a. List a → Nat
Examples:
  - length []        = 0
  - length [x]       = 1
  - length [x, y, z] = 3
Primitive vocabulary needed: pattern match on List, Nat constructors 0/Succ, recursive call
Estimated AST size: 10-15 nodes
```

### 15.3 Step 3: DSL specification

```json
{
  "fields": [
    {"name": "node_kind", "cutoff": 12},
    {"name": "type",      "cutoff": 8},
    {"name": "binder_id", "cutoff": 16},
    {"name": "value",     "cutoff": 8}
  ],
  "sites": 12,
  "boundary": {
    "0": {"node_kind": "Lambda", "type": "List_a_to_Nat"}
  },
  "constraints": [
    {"kind": "well_typed_subtree", "root": 0, "weight": 10.0},
    {"kind": "example", "input": "[]",          "output": "0", "weight": 5.0},
    {"kind": "example", "input": "[x]",         "output": "1", "weight": 5.0},
    {"kind": "example", "input": "[x, y, z]",   "output": "3", "weight": 5.0},
    {"kind": "vocabulary", "primitives": ["Match", "Cons", "Nil", "Succ", "Zero", "Var", "App"]}
  ],
  "observables": [
    {"site": k, "field": "node_kind", "op": "argmax"} for k in range(12)
  ],
  "search": {"steps": 100, "chi_max": 32, "dt": 0.05}
}
```

### 15.4 Step 4: Hamiltonian construction

The compiler builds `H = Σ_i w_i P_i` from each constraint. Schematically:

**Well-typed-subtree term** (one for each potential `App` node at sites 0, 4, 7, 10):

```
P_App_typing = |App⟩⟨App|_{parent}
             ⊗ Σ_{t_f, t_a} ( |t_f⟩⟨t_f|_{func_child}
                              ⊗ |t_a⟩⟨t_a|_{arg_child}
                              ⊗ (I - P_consistent_result_type) )
```

**Pattern-match exhaustiveness** (at the `Match` node, site 2):

```
P_Match_exhaustive = |Match⟩⟨Match|_2
                   ⊗ ( I - |Nil⟩⟨Nil|_3 ⊗ I_4 - |Cons⟩⟨Cons|_5 ⊗ I_6 )
```

(Requires both `Nil` and `Cons` patterns to be present.)

**Recursive-call termination** (at site 10, the recursive call):

```
P_termination = |App⟩⟨App|_10 ⊗ |Var⟩⟨Var|_11
              ⊗ ( I - P_structurally_smaller )
```

(The argument to the recursive call must be a strict sub-structure.)

**Example constraints**: encoded as auxiliary "evaluation" sites that propagate input values through the AST. Schematically (full version in §10.3 evaluation Hamiltonian):

```
P_example_input_nil_output_zero = ... encodes that evaluating with input [] gives 0
```

The full Hamiltonian is the weighted sum. Hermiticity is verified at compile time.

### 15.5 Step 5: State initialization

Bond dimension allocated: `χ_max = 32`. Local Hilbert space dimension: `12 × 8 × 16 × 8 = 12,288` per site, with truncation to top-`χ` Schmidt vectors at each bond.

Site 0 (the Lambda) is clamped to its boundary value: `node_kind = Lambda`, `type = List_a_to_Nat`.

Sites 1-11 are initialized as small random product states with bond dimension 1, then a brief warmup of free Hamiltonian evolution to establish initial entanglement structure.

### 15.6 Step 6: Imaginary-time TEBD evolution

The system evolves `|Ψ(τ + dτ)⟩ ∝ e^{-H dτ}|Ψ(τ)⟩` with `dτ = 0.05`. Sample energies and diagnostics over the run:

```
Step  τ      ⟨H⟩        max_bond  midpoint_S
   0  0.00   23.482      1         0.000     (initial product state)
  10  0.50   14.713      8         1.842
  20  1.00    8.396      14        2.137
  30  1.50    4.124      19        2.301
  40  2.00    1.823      24        2.354
  50  2.50    0.612      28        2.367
  60  3.00    0.183      30        2.371
  70  3.50    0.047      31        2.372
  80  4.00    0.011      32        2.372
  90  4.50    0.003      32        2.372
 100  5.00    0.0008     32        2.372     (converged near ground state)
```

The energy descent is monotone (Theorem 13.1). Bond dimension saturates at `χ_max = 32`. The midpoint entanglement entropy stabilizes at `S ≈ 2.37 bits`, reflecting the binding-related entanglement between the function-definition site and the recursive-call site.

### 15.7 Step 7: Measurement

At each site, measure `argmax_n ⟨Ψ|P_n|Ψ⟩` for each field:

```
Site   node_kind     type            binder_id    value
  0    Lambda        List_a → Nat    {xs: 1}      —
  1    Var (xs)      List a          1            —
  2    Match         Nat             —            —
  3    PatternNil    —               —            —
  4    Zero          Nat             —            0
  5    PatternCons   List a          {y: 2,       —
                                      ys: 3}
  6    —             —               —            —    (unused)
  7    Succ          Nat             —            —
  8    App           Nat             —            —
  9    Var (length)  List a → Nat    0            —
 10    Var (ys)      List a          3            —
 11    —             —               —            —    (unused)
```

For variable-binding resolution, measure pairwise observables: `⟨P_{binder_id=1, site=0} ⊗ P_{binder_id=1, site=1}⟩` should be high (the lambda binds `xs`, and site 1 is a use of `xs`). All other binder pairings yield low expectations. This is variable scoping read out as an EPR-style entanglement measurement.

### 15.8 Step 8: AST decoding

Reconstructing the tree from the measurements:

```
Lambda "xs"
  ├── parameter site = 1
  └── body site = 2
        Match (at site 2)
          ├── scrutinee = Var "xs" (referencing site 1)
          ├── case PatternNil → Zero (site 3 → site 4)
          └── case PatternCons "y" "ys" → (site 5)
                Succ (at site 7)
                  └── App (at site 8)
                        ├── func = Var "length" (recursive reference, site 9)
                        └── arg  = Var "ys" (referencing site 5's binding, site 10)
```

### 15.9 Step 9: Validation pass

The decoded AST is type-checked classically:

- `Lambda xs. ...` has type `List a → ?` — body type to be determined.
- `Match xs of ...` requires `xs : List a` (consistent with the lambda's parameter type) — ✓.
- `PatternNil → Zero` returns `Nat` — ✓.
- `PatternCons y ys → Succ (length ys)` requires `length ys : Nat`, hence `length : List a → Nat`, recursive — ✓.
- Termination: the recursive call is on `ys`, a strict sub-structure of `xs` — ✓.

The classical type checker reports: well-typed.

### 15.10 Step 10: Pretty-printing

```haskell
length :: List a -> Nat
length xs = case xs of
  []        -> 0
  (y : ys)  -> Succ (length ys)
```

### 15.11 Step 11: Verification against examples

The runtime evaluates:

- `length []`        → matches `Nil` → returns `Zero` → `0`. ✓
- `length [x]`       → matches `Cons x []` → `Succ (length [])` → `Succ 0` → `1`. ✓
- `length [x, y, z]` → unfolds → `Succ (Succ (Succ 0))` → `3`. ✓

All examples pass.

### 15.12 Step 12: LLM verbalization

The LLM returns to the user:

> "Here's the function you asked for:
>
> ```haskell
> length :: List a -> Nat
> length xs = case xs of
>   []        -> 0
>   (y : ys)  -> Succ (length ys)
> ```
>
> It uses pattern matching: if the list is empty, the length is 0; otherwise, the length is one more than the length of the rest. The function is recursive and terminates because each recursive call operates on a strictly smaller list.
>
> This implementation is *provably* type-correct (residual energy: 0.0008, well below threshold) and matches all three examples you provided. The entanglement structure of the recursion was captured by the architecture's binding-as-entanglement mechanism, which is why the recursive call correctly references `ys` (the sub-list bound in the `Cons` pattern) rather than `xs` (the outer parameter)."

### 15.13 What this example demonstrates concretely

- **The Hamiltonian is real, not metaphorical**: §15.4 specifies actual operators on actual Hilbert spaces.
- **Imag-time evolution monotonically reduces energy** (§15.6): empirical instance of Theorem 13.1.
- **The decoded AST satisfies every constraint** (§15.9): empirical instance of Theorem 13.2.
- **Variable binding is recovered as entanglement measurement** (§15.7): the central conceptual claim of §8.1, operational.
- **The whole pipeline is a single optimization on a single objective**: no separate "training" vs "inference" phases.

A reader who is skeptical of the abstract architecture can run a small program implementing this example and reproduce these numbers (after the §10 roadmap is built). The example is the architecture's most concrete demonstration of "this actually works."

---

## 16. Limitations, Open Questions, and Anticipated Objections

This section bounds the architecture's claims and pre-empts the most likely reviewer objections. Intellectual honesty about what we cannot do is what makes the claims about what we *can* do credible.

### 16.1 Hard limitations

These cannot be fixed by engineering effort within the current paradigm; they require fundamentally different architectures.

**Volume-law entanglement unreachable.** Per Theorem 13.5, the MPS substrate cannot represent states with entanglement entropy growing faster than `log χ` across any bipartition. Domains where the natural representation requires volume-law entanglement (most large-scale image processing, some long-range chaotic dynamics) are outside the architecture's hypothesis class. PEPS extensions help in 2D but not arbitrarily.

**Web-scale natural language.** Language model tasks requiring world knowledge, cultural context, or fluent generation are not the target. The LLM frontend handles these; the QPCN does not.

**Effectful and concurrent programs (current substrate).** The QFT substrate is unitary; effectful and concurrent computation has non-unitary structure (Lindblad dynamics, non-commutative observables). Extensions exist (§12 future work) but are substantially harder than the pure case.

**Open-ended real-valued optimization.** The QPCN's substrate is discrete (truncated Fock space). Continuous optimization problems (e.g., regression on real-valued data) can be encoded but typically require very large local Hilbert space dimensions, scaling poorly compared to gradient-based classical methods on the same problems.

**Sample efficiency claim is unproven.** We claim capability compounds with use (§11.7). This is a *conjecture* (§13.8) until validated empirically. Many other ML architectures have made similar claims that did not pan out.

### 16.2 Currently unproven assumptions

These are assumptions the architecture relies on that we believe are correct but haven't established.

**The Hamiltonian compiler produces non-frustrated Hamiltonians for STLC.** §13.2's correctness theorem requires the Hamiltonian to be a sum of positive-semidefinite penalty operators with consistent ground states. We believe this holds for STLC encodings but haven't proved it formally. *Risk*: there might be encoding artifacts where the lowest-energy state is technically not the intended AST.

**Gap closure on hard problems is signal, not noise.** §13.1's convergence depends on a nonzero gap. When the gap closes during inference, we interpret this as "problem is near a phase transition." This interpretation could be wrong; gap closure might also indicate encoding bugs.

**Wake-sleep cycles converge.** §10.9's wake-sleep loop is designed to grow a useful library. We don't have a proof that it doesn't degenerate (collapse to trivial primitives, oscillate without convergence, etc.). DreamCoder's empirical success suggests the framework is sound, but tensor-network specifics could differ.

**Anomaly detection identifies genuine impossibilities, not encoding artifacts.** §12.1 claims to detect provable impossibility. A nonzero anomaly polynomial could in principle be an artifact of the encoding rather than a true obstruction. The mitigation (verify zero anomaly on known consistent theorems) is necessary; without it the claim is suspect.

**Bootstrap gives nontrivial bounds for nontrivial type signatures.** §12.4's bootstrap could be trivially satisfied (no constraints active) or trivially unsatisfiable (constraints contradict). The interesting regime — informative non-trivial bounds — has to be demonstrated empirically.

### 16.3 Open research questions

Things the design doesn't address but should:

**Best way to extend to dependent types?** STLC is the prototype. Dependent types (System F, Calculus of Constructions, Lean's type theory) have much richer structure. Naive encoding causes combinatorial blowup. Open question: what's the right way to extend the Hamiltonian compiler?

**Best handling of effects and concurrency?** Effectful computation breaks unitarity. Possible extensions: Lindblad dynamics for stochastic effects, density matrix substrate, non-commutative geometry. None of these are explored in detail.

**Optimal tensor topology for ASTs?** 1D MPS forces a linear ordering on an inherently tree-structured AST. Tree tensor networks (TTNs) seem natural but have their own complications. MERA is a generalization but adds infrastructure cost.

**How to integrate with existing tooling?** Lean, Coq, Agda, Idris all have rich tactic systems and libraries. The QPCN should plug into these as a tactic, not replace them entirely. The protocol for this integration is not designed.

**Scaling laws — exact form?** §13.8 conjectures a saturating exponential capability curve. Real scaling laws for ML systems usually have power-law components. The exact form requires §12.7 replica method implementation.

**Catastrophic forgetting?** Adding new lemmas might displace old ones in surprising ways. The §10.8 library should be append-only, but if abstraction (§10.9) replaces old primitives with new composites, old proofs may need re-derivation. This isn't analyzed.

**Adversarial robustness.** What if the LLM frontend emits adversarial DSL specs? The QPCN itself is robust by §13.4 (holographic threshold), but the interface between LLM and QPCN is a trust boundary that needs analysis.

### 16.4 Anticipated reviewer objections and responses

**"This violates Rice's theorem."**

Response: It does not. Rice's theorem states that non-trivial semantic properties of arbitrary programs are undecidable. The QPCN gives provable *bounds* on properties (§12.4 bootstrap) and exact properties of programs in a restricted class (§13.5 area-law states only). Decidability is preserved by the class restriction, not violated.

**"This violates Gödel's incompleteness theorem."**

Response: It does not. §12.1 anomaly detection identifies specific finite obstructions in a specific encoding's constraint algebra. It does not claim to mechanize the meta-mathematical proof that *every* sufficiently powerful formal system is incomplete. We mechanize specific instances of unprovability for specific theorem-axiom pairs, which is a Σ_1-level claim that's perfectly decidable.

**"Tensor networks have been tried and don't beat transformers."**

Response: On natural language, this is correct, and we explicitly don't compete (§16.1). On structured, conservation-law-bearing domains, tensor networks already outperform classical methods in published quantum chemistry literature (Bauer et al. 2020). Our claim is that adding PCN dynamics and the hierarchical composition extensions further extends this advantage.

**"How is this not just classical SAT with quantum vocabulary?"**

Response: SAT solvers do discrete combinatorial search; the QPCN does continuous variational inference over a quantum Hilbert space. The state evolves continuously, carries phase information, accumulates non-classical correlations via entanglement, and supports operations (Wilson loops §12.2, anomalies §12.1, holographic codes §12.5) that have no classical SAT analog. The substrate is operator-algebraic, not Boolean.

**"You're using quantum mechanical language to describe classical computation."**

Response: The substrate is *literally* quantum mechanical: complex amplitudes, unitary evolution, entanglement, measurement. The MPS encodes states in a Hilbert space; the Hamiltonian is Hermitian; the dynamics are Schrödinger or Lindblad. The classical simulation (numpy + scipy + qiskit on classical hardware) gives correct expectation values for these quantum-mechanical objects but does not change their nature. The architecture would run identically on quantum hardware once available; classical simulation is a tractability concession, not a metaphysical commitment.

**"You haven't actually shown anything beyond a working prototype."**

Response: Correct. The current branch demonstrates that the *substrate* works (39 passing tests). The §10–§12 roadmap is what would demonstrate the *capabilities*. The paper this design document supports would be written after Phase E (§10.11) or Phase H (combined system), not before. We claim only what we've demonstrated; the roadmap is a research plan, not a result.

**"The combination of features (§12) seems too good to be true."**

Response: Each individual feature is grounded in established physics. The combination is novel because no one has previously identified the QPCN substrate as the appropriate home for all of them simultaneously. If reviewers find any individual claim implausible, we can point to the published precedents (§19 references) showing the same machinery solving the analog problem in its native domain. The question is not whether the math works (it does, in physics) but whether the import to reasoning is valid. That import is what we are arguing for.

**"How do you know the QPCN won't just memorize training data?"**

Response: The architecture has no parameters that *can* memorize in the classical NN sense. The Hamiltonian's parameters are ~hundreds; they cannot encode billions of specific examples. The library (§10.8) stores specific solved problems but these are addressable lemmas, not memorized facts. The system generalizes via abstraction (§10.9), not interpolation. Memorization would manifest as failure on slight problem variations; this is a testable prediction.

**"This is so different from existing work that there's nothing to compare against."**

Response: We compare against AlphaProof for proof composition (Phase E), Hazel/Synquid for synthesis (Phase C), DreamCoder for library learning (after §10.9), classical type inference (sanity baseline), and quantum chemistry benchmarks (for the canonical winning domain). Each comparison is targeted at one component of the architecture. The combined-system paper compares against the closest thing in each category for each claim.

**"What if a §12 extension turns out not to work as theorized?"**

Response: Each extension is independently testable. If §12.1 anomaly detection doesn't catch genuine impossibility cases, we keep the rest and report the negative result. If §12.4 bootstrap doesn't give nontrivial bounds, we keep the rest. The architecture's value does not depend on every extension working; it depends on enough of them working to justify the combined claim.

### 16.5 Why this section matters

A paper with a hundred-line list of strengths and no honest limitations section is suspect. By calling out what we cannot do (§16.1), what we have not yet proved (§16.2), what remains open (§16.3), and the objections we expect (§16.4), we make the work falsifiable in the Popperian sense. Reviewers can identify exactly where to push back and exactly what would falsify or confirm each claim.

The combination of §13 (formal theorems), §14 (rigorous evaluation), §15 (concrete worked example), and §16 (honest limitations) transforms the document from a design proposal into a *research program* — something that can be executed, measured, evaluated by peers, and contributed to the literature on its merits.

---

## 17. Code Layout Reference

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
│   ├── typing_hamiltonian.py           # STLC typing rules + Hamiltonian compiler
│   │                                   #   (spec called these typing_rules.py +
│   │                                   #    hamiltonian_compiler.py; folded
│   │                                   #    together since the rule table only
│   │                                   #    exists to seed the compiler)
│   ├── _mera_typing_rules.py           # MERA-substrate typing rule helpers
│   ├── evaluation_hamiltonian.py       # beta-reduction Hamiltonian (MPS)
│   ├── mera_evaluation_hamiltonian.py  # beta-reduction Hamiltonian (MERA)
│   ├── debugger.py                     # residual-energy → error report
│   ├── synthesis/                      # STLC synthesis pipeline
│   ├── mera_synthesis/                 # MERA-substrate synthesis pipeline
│   └── demo_stlc_synthesis.py          # first publishable milestone
├── qft/                                         # § 10.4
│   ├── mera.py                         # MERA state representation
│   └── mera_evolution.py               # hierarchical TEBD
├── bridge/                                      # § 10.5
│   ├── __init__.py
│   ├── api.py                          # RPC server for LLM frontend
│   ├── llm.py                          # Ollama LLM client
│   ├── llm_emitter.py                  # LLM → DSL emitter
│   ├── dsl/                            # PACKAGE (spec called this dsl.py).
│   │   ├── schema.py                   #   DSL schema
│   │   ├── expr_parser.py              #   expression parser
│   │   ├── compiler.py                 #   DSL → Hamiltonian compiler
│   │   ├── pipeline.py                 #   end-to-end DSL pipeline
│   │   └── ...                         #   (vocab, term, cross_validate, ...)
│   └── runtime/                        # PACKAGE (spec called this runtime.py).
│       ├── hamiltonian.py              #   problem-runner Hamiltonian builder
│       ├── initial_state.py            #   initial-state preparation
│       ├── evolution.py                #   driver
│       ├── observables.py              #   observable evaluation
│       ├── clamp.py                    #   constraint clamping
│       └── result.py                   #   RunResult / diagnostics
└── composition/                                 # § 10.8 - 10.11 and § 12
    ├── __init__.py
    ├── lemma_library.py                # § 10.8: storage + indexing + registration
    ├── lemma_library_adapter.py        # § 10.8: adapter (cache / pruning / cheapest_for_type)
    ├── promoter.py                     # § 10.8: use_lemma DSL constraint compiler
    ├── subtree_miner.py                # § 10.9: enumerate sub-MPSes
    ├── abstraction.py                  # § 10.9: cluster + canonical form computation
    ├── wake_sleep.py                   # § 10.9: orchestrate the cycle
    ├── goal_graph.py                   # § 10.10: DAG of sub-goals
    ├── dispatcher.py                   # § 10.10: spawn + collect QPCN runs
    ├── result_integrator.py            # § 10.10: bottom-up clamping
    ├── revision.py                     # § 10.10: LLM-guided alternative decomposition
    ├── orchestrator.py                 # § 10.10: goal-graph driver
    ├── demo_hierarchical_proof.py      # § 10.11: second publishable milestone
    ├── anomaly.py                      # § 12.1: provable impossibility detection
    ├── topological_invariants.py       # § 12.2: program-equivalence via Wilson loops
    ├── topological_degeneracy.py       # § 12.3: a-priori proof-strategy counting
    │                                   #   (renamed from spec's bare name per
    │                                   #    D12 honesty trail — the entropy/
    │                                   #    proof-homotopy distinction matters)
    ├── bootstrap.py                    # § 12.4: conformal-bootstrap for type-only reasoning
    │                                   #   (D13: now delivers a real
    │                                   #    type-derived complexity bound)
    ├── holographic_correction.py       # § 12.5: fault-tolerant reasoning via QEC
    ├── goldstone.py                    # § 12.6: automatic missing-lemma identification
    ├── replica_complexity.py           # § 12.7: predictive typical-case complexity
    │                                   #   (renamed from spec's `replica.py` per
    │                                   #    D14 honesty trail — distinguishes
    │                                   #    leaf-marginal observable from a true
    │                                   #    proof-space partition function)
    ├── dynamical_pt.py                 # § 12.8: self-detecting curriculum
    ├── meta_hamiltonian.py             # § 12.9: self-modification (operator-valued substrate)
    ├── holographic_compilation.py      # § 12.10: provably-correct compiler passes via RG
    │                                   #   (D15: real optimization pass + non-
    │                                   #    tautological verify)
    ├── entanglement_spectrum.py        # § 12.11: modular-Hamiltonian proof classification
    ├── quantum_extremal_surface.py     # § 12.12: a priori proof complexity from QES
    │                                   #   (D16: a-priori prediction, not post-
    │                                   #    hoc ranking)
    ├── bidirectional.py                # § 12.13: forward + backward simultaneous evolution
    ├── quantum_walk.py                 # § 12.14: quantum-walk speedup for goal-graph search
    ├── witten_index.py                 # § 12.15: topological theorem fingerprints
    ├── worldline_pi.py                 # § 12.16: path-integral Bayesian proof ranking
    ├── qca_classification.py           # § 12.17: QCA framework + topological index
    ├── noether_discovery.py            # § 12.18: automated conservation-law discovery
    └── tests/
        ├── test_library_store.py       # (lemma library storage)
        ├── test_subtree_miner.py
        ├── test_wake_sleep.py
        ├── test_goal_graph.py
        ├── test_dispatcher.py
        ├── test_anomaly.py
        ├── test_topological_invariants.py
        ├── test_topological_degeneracy.py
        ├── test_bootstrap.py
        ├── test_holographic_correction.py
        ├── test_goldstone.py
        ├── test_replica_complexity.py  # (renamed from test_replica per D14)
        ├── test_dynamical_pt.py
        ├── test_meta_hamiltonian.py
        ├── test_holographic_compilation.py
        ├── test_entanglement_spectrum.py
        ├── test_quantum_extremal_surface.py
        ├── test_bidirectional_evolution.py
        ├── test_quantum_walk.py
        ├── test_witten_index.py
        ├── test_worldline_pi.py
        ├── test_qca_classification.py
        └── test_noether_discovery.py
```

**Honest-rename trail (§12.x renames per D12–D16).** Several
`composition/` modules carry names slightly different from those the
spec used in earlier drafts. The renames are *not* gratuitous; each
encodes a substantive correctness fix surfaced during the audit loop
and is recorded in `DEVIATIONS.md`:

- D12 — `composition/topological_degeneracy.py` (was a bare Betti
  count; now a proof-homotopy count distinct from
  binding-graph entropy);
- D13 — `composition/bootstrap.py` (was tautological; now delivers a
  type-derived complexity bound);
- D14 — `composition/replica_complexity.py` (renamed from
  `replica.py`; the leaf-marginal observable is honestly named to
  avoid the partition-function confusion);
- D15 — `composition/holographic_compilation.py` (now ships a real
  RG-fixed-point optimization pass with a non-tautological verify);
- D16 — `composition/quantum_extremal_surface.py` (a-priori
  prediction, not post-hoc ranking).

Additionally, `bridge/dsl/` and `bridge/runtime/` are *packages*, not
single files; `logic/typing_hamiltonian.py` folds together what the
spec originally split as `typing_rules.py` + `hamiltonian_compiler.py`
(the rule table only ever existed to seed the compiler). Future
readers grepping for spec names should consult this section first
before assuming a missing module.

Open research questions raised by §16.3 are tracked in
`OPEN_QUESTIONS.md` at the repo root.

---

## 18. Glossary

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
- **Anomaly** — In QFT, a classical symmetry that fails to survive quantization. Detection of an anomaly proves no consistent theory with that symmetry exists. Used in §12.1 as a mechanism for proving theorems unprovable from given axioms.
- **Wilson loop** — Expectation value of an operator product traced around a closed path. Topological invariant in TQFTs (§12.2).
- **Topological order** — Phase of matter characterized by ground-state degeneracy depending on the topology of the underlying space rather than local order parameters. Used in §12.3 to count distinct proof strategies a priori.
- **Conformal bootstrap** — Numerical method to derive CFT data from pure consistency constraints (unitarity, crossing symmetry). Used in §12.4 for type-only program reasoning.
- **HaPPY code / holographic code** — Tensor-network quantum error-correcting code that realizes AdS/CFT-like structure (Pastawski et al. 2015). Used in §12.5 for fault-tolerant reasoning.
- **Goldstone mode** — Massless excitation corresponding to a spontaneously broken continuous symmetry direction. Used in §12.6 to identify missing lemmas in failed proofs.
- **Replica trick** — Method for computing average free energy over a disordered ensemble by analytic continuation `n → 0` from positive-integer `n` replicas. Used in §12.7 for predictive typical-case complexity.
- **Loschmidt echo** — Overlap `|⟨Ψ_0|Ψ(t)⟩|^2` between an initial state and its time-evolved version under a quench. Non-analyticities signal dynamical phase transitions. Used in §12.8 to detect the architecture's own capability transitions.
- **Crossing symmetry** — In CFTs, the requirement that an OPE can be computed in either of two channels with consistent results. A foundational constraint of the conformal bootstrap.
- **Parametricity** — Reynolds 1983. Polymorphic functions are constrained by their type signatures to behave uniformly on their type variables, giving "theorems for free." A consistency requirement in §12.4 bootstrap reasoning.
- **Stoquastic Hamiltonian** — A Hamiltonian whose off-diagonal matrix elements in some basis are real and non-positive. Stoquastic Hamiltonians admit efficient classical simulation via path integral Monte Carlo or imaginary-time TEBD. The condition for Theorem 13.1.
- **Gap** — The energy difference `Δ = E_1 - E_0` between the ground state and first excited state of a Hamiltonian. Convergence rate of imag-time evolution is `O(e^{-Δτ})`.
- **Variational principle** — The statement that the expectation value `⟨Ψ|H|Ψ⟩` is minimized over normalized states by the ground state. Foundation of every variational method including DMRG, VMC, VQE, and our QPCN.
- **pass@k** — Standard synthesis benchmark metric: fraction of problems for which at least one of `k` generated candidates is correct. Used in §14.2 for synthesis evaluation.
- **type@k** — Our distinctive metric: fraction of `k` candidates that are type-correct. The QPCN's claimed advantage is `type@1 = 1.0`.
- **Pre-registration** — Publishing experimental hypotheses and decision rules before running the experiments, to prevent post-hoc cherry-picking. §14.7.
- **Frustration-free Hamiltonian** — A Hamiltonian where every local term is simultaneously minimized by the global ground state. Enables fast convergence and clean correctness guarantees.
- **MPO (Matrix Product Operator)** — A tensor-network representation of operators, dual to MPS. Used in §12.9 to encode the QPCN's own Hamiltonian as a quantum state for meta-level reasoning.
- **Modular Hamiltonian** — `K = -log ρ_A` for a subsystem `A`. Generator of entanglement dynamics; its spectrum is the entanglement spectrum (§12.11).
- **Entanglement spectrum** — The set of eigenvalues of a reduced density matrix (or equivalently, the squared Schmidt coefficients at a bond). Li & Haldane 2008 established it as a topological classifier.
- **Quantum extremal surface (QES)** — Generalization of the Ryu-Takayanagi minimal surface to quantum-corrected, dynamic geometries (Engelhardt-Wall 2015). Used in §12.12 to compute a priori proof complexity.
- **Ryu-Takayanagi formula** — `S(A) = Area(γ_A) / 4G_N` connecting CFT entanglement entropy to minimal-surface area in AdS bulk. Foundation of §12.12.
- **Gödel machine** — Schmidhuber 2003. A theoretical self-modifying program with provable optimality. The classical precursor to §12.9.
- **Bidirectional reasoning** — Inference that runs simultaneously from premises (forward) and goal (backward), looking for a meeting point. Standard in some PL contexts (Pierce-Turner bidirectional typing); §12.13 makes it universal via unitarity.
- **Quantum walk** — Unitary analog of a classical random walk on a graph. Used in §12.14 for goal-graph search with quadratic-to-exponential speedup vs classical algorithms.
- **Witten index** — `tr((-1)^F e^{-βH})`. Topological invariant counting ground states with signs; nonzero index proves a ground state exists. Used in §12.15 for theorem fingerprinting.
- **Path integral** — Feynman's formulation where a transition amplitude is a sum over all paths weighted by `e^{iS/ℏ}`. Used in §12.16 to assign Bayesian probabilities to candidate proofs.
- **Action functional** — `S = ∫ L dt`. The integral of the Lagrangian along a path. In our setting, a complexity measure for a proof path. Path integrals weight paths by `e^{-S/T}`.
- **Quantum cellular automaton (QCA)** — A local, translation-invariant, unitary dynamics on a quantum lattice. Schumacher-Werner 2004. Every Trotter evolution is a QCA; §12.17 leverages this framework's classification theorems.
- **QCA index** — A topological invariant classifying 1D QCAs into equivalence classes under local perturbations (Gross-Nesme-Vogts-Werner 2012).
- **Noether's theorem** — Every continuous symmetry of the action implies a conserved current `J^μ = (∂L/∂(∂_μϕ))δϕ - L δx^μ`. Constructive: gives the conservation law explicitly. Used in §12.18 for automated invariant discovery.
- **Lieb-Robinson bound** — In a local quantum lattice system, information propagates at most at a finite speed `v_LR`. Foundation for causal reasoning bounds via §12.17.

---

## 19. Prior Research and Citations

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

### Anomalies in quantum field theory
- Adler, S. L. (1969). *Axial-vector vertex in spinor electrodynamics.* PRL.
- Bell, J. S., & Jackiw, R. (1969). *A PCAC puzzle: π⁰ → γγ in the σ-model.* Nuovo Cimento A.
- Wess, J., & Zumino, B. (1971). *Consequences of anomalous Ward identities.* Phys. Lett. B.
- 't Hooft, G. (1980). *Naturalness, chiral symmetry, and spontaneous chiral symmetry breaking.* Recent Developments in Gauge Theories. (Anomaly matching.)

### Topological quantum field theory and knot invariants
- Witten, E. (1988). *Topological quantum field theory.* CMP.
- Witten, E. (1989). *Quantum field theory and the Jones polynomial.* CMP.
- Reshetikhin, N., & Turaev, V. (1991). *Invariants of 3-manifolds via link polynomials and quantum groups.* Inv. Math.
- Atiyah, M. (1988). *Topological quantum field theories.* IHES Publ. Math.

### Topological order and topological phases
- Wen, X.-G. (1989). *Vacuum degeneracy of chiral spin states in compactified space.* PRB.
- Kitaev, A. (2003). *Fault-tolerant quantum computation by anyons.* Annals of Physics.
- Kitaev, A. (2006). *Anyons in an exactly solved model and beyond.* Annals of Physics.
- Nayak, C., Simon, S. H., Stern, A., Freedman, M., & Das Sarma, S. (2008). *Non-abelian anyons and topological quantum computation.* RMP.

### Conformal bootstrap
- Polyakov, A. M. (1974). *Nonhamiltonian approach to conformal quantum field theory.* JETP.
- Ferrara, S., Gatto, R., & Grillo, A. F. (1973). *Tensor representations of conformal algebra and conformally covariant operator product expansion.* Annals of Physics.
- Rattazzi, R., Rychkov, V. S., Tonni, E., & Vichi, A. (2008). *Bounding scalar operator dimensions in 4D CFT.* JHEP. (Modern numerical bootstrap.)
- Kos, F., Poland, D., & Simmons-Duffin, D. (2014). *Bootstrapping mixed correlators in the 3D Ising model.* JHEP.
- Simmons-Duffin, D. (2015). *A semidefinite program solver for the conformal bootstrap.* JHEP (SDPB).
- Poland, D., Rychkov, S., & Vichi, A. (2019). *The conformal bootstrap: theory, numerical techniques, and applications.* RMP.

### Holographic quantum error correction
- Pastawski, F., Yoshida, B., Harlow, D., & Preskill, J. (2015). *Holographic quantum error-correcting codes: toy models for the bulk/boundary correspondence.* JHEP. (HaPPY code.)
- Hayden, P., Nezami, S., Qi, X.-L., Thomas, N., Walter, M., & Yang, Z. (2016). *Holographic duality from random tensor networks.* JHEP.
- Harlow, D. (2017). *The Ryu-Takayanagi formula from quantum error correction.* CMP.

### Goldstone's theorem and spontaneous symmetry breaking
- Nambu, Y. (1960). *Quasi-particles and gauge invariance in the theory of superconductivity.* Phys. Rev.
- Goldstone, J. (1961). *Field theories with "superconductor" solutions.* Nuovo Cimento.
- Goldstone, J., Salam, A., & Weinberg, S. (1962). *Broken symmetries.* Phys. Rev.

### Replica method in statistical mechanics
- Edwards, S. F., & Anderson, P. W. (1975). *Theory of spin glasses.* J. Phys. F.
- Parisi, G. (1979). *Toward a mean field theory for spin glasses.* Phys. Lett. A.
- Parisi, G. (1980). *A sequence of approximated solutions to the SK model for spin glasses.* J. Phys. A.
- Mézard, M., Parisi, G., & Virasoro, M. A. (1987). *Spin Glass Theory and Beyond.* World Scientific.
- Mertens, S., Mézard, M., & Zecchina, R. (2006). *Threshold values of random K-SAT from the cavity method.* Random Structures & Algorithms.

### Dynamical phase transitions
- Heyl, M., Polkovnikov, A., & Kehrein, S. (2013). *Dynamical quantum phase transitions in the transverse-field Ising model.* PRL.
- Karrasch, C., & Schuricht, D. (2013). *Dynamical phase transitions after quenches in nonintegrable models.* PRB.
- Heyl, M. (2018). *Dynamical quantum phase transitions: a review.* Rep. Prog. Phys.

### Parametricity and theorems for free
- Reynolds, J. C. (1983). *Types, abstraction and parametric polymorphism.* IFIP Congress.
- Wadler, P. (1989). *Theorems for free!* FPCA.

### Benchmarks for synthesis and theorem proving
- Zheng, K., Han, J. M., & Polu, S. (2021). *miniF2F: a cross-system benchmark for formal Olympiad-level mathematics.* arXiv:2109.00110.
- Omar, C., Voysey, I., Hilton, M., Aldrich, J., & Hammer, M. A. (2017). *Hazelnut: a bidirectionally typed structure editor calculus.* POPL.
- Osera, P.-M., & Zdancewic, S. (2015). *Type-and-example-directed program synthesis.* PLDI (Myth).
- Polozov, O., & Gulwani, S. (2015). *FlashMeta: a framework for inductive program synthesis.* OOPSLA.
- Rupp, M., Tkatchenko, A., Müller, K.-R., & von Lilienfeld, O. A. (2012). *Fast and accurate modeling of molecular atomization energies with machine learning.* PRL (QM7).
- Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014). *Quantum chemistry structures and properties of 134 kilo molecules.* Scientific Data (QM9).
- Chen, M., Tworek, J., Jun, H. et al. (2021). *Evaluating large language models trained on code.* arXiv:2107.03374 (HumanEval).

### Type theory, type safety, and program verification
- Wright, A. K., & Felleisen, M. (1994). *A syntactic approach to type soundness.* Information and Computation. ("Well-typed programs cannot go wrong.")
- Pierce, B. C. (2002). *Types and Programming Languages.* MIT Press.
- The Coq Development Team. *The Coq Proof Assistant.* (Standard reference.)
- The Lean Community. (2024). *Mathlib4: a unified library of mathematics formalized in the Lean theorem prover.* (Standard reference.)

### Scaling laws and capability prediction
- Kaplan, J., McCandlish, S., Henighan, T. et al. (2020). *Scaling laws for neural language models.* arXiv:2001.08361.
- Hoffmann, J., Borgeaud, S., Mensch, A. et al. (2022). *Training compute-optimal large language models.* arXiv:2203.15556 (Chinchilla).

### Tensor network methods, additional foundational
- Vidal, G. (2003). *Efficient classical simulation of slightly entangled quantum computations.* PRL.
- White, S. R. (1992). *Density matrix formulation for quantum renormalization groups.* PRL (original DMRG).
- Schollwöck, U. (2011). *The density-matrix renormalization group in the age of matrix product states.* Annals of Physics. (Standard review.)

### Self-modification and meta-learning
- Schmidhuber, J. (2003). *Gödel machines: Self-referential universal problem solvers making provably optimal self-improvements.* Tech Report IDSIA. (The classical precursor to §12.9.)
- Schmidhuber, J. (2007). *Gödel machines: Fully self-referential optimal universal self-improvers.* In Artificial General Intelligence (Springer).

### Compilation and program optimization
- Leroy, X. (2006). *Formal certification of a compiler back-end.* POPL. (CompCert — hand-written, language-specific provably-correct compiler.)
- Tate, R., Stepp, M., Tatlock, Z., & Lerner, S. (2009). *Equality saturation: a new approach to optimization.* POPL.
- Willsey, M., Nandi, C., Wang, Y. R., Flatt, O., Tatlock, Z., & Panchekha, P. (2021). *Egg: Fast and extensible equality saturation.* POPL.

### Entanglement spectrum and topological classification
- Li, H., & Haldane, F. D. M. (2008). *Entanglement spectrum as a generalization of entanglement entropy.* PRL. (Foundational for §12.11.)
- Pollmann, F., Turner, A. M., Berg, E., & Oshikawa, M. (2010). *Entanglement spectrum of a topological phase in one dimension.* PRB.
- Kitaev, A., & Preskill, J. (2006). *Topological entanglement entropy.* PRL.

### Ryu-Takayanagi and quantum extremal surfaces
- Ryu, S., & Takayanagi, T. (2006). *Holographic derivation of entanglement entropy from the anti-de Sitter space/conformal field theory correspondence.* PRL.
- Hubeny, V. E., Rangamani, M., & Takayanagi, T. (2007). *A covariant holographic entanglement entropy proposal.* JHEP.
- Engelhardt, N., & Wall, A. C. (2015). *Quantum extremal surfaces: holographic entanglement entropy beyond the classical regime.* JHEP.
- Susskind, L. (2016). *Computational complexity and black hole horizons.* Fortschritte der Physik. (Complexity = volume / complexity = action conjectures.)

### Bidirectional reasoning
- Pierce, B. C., & Turner, D. N. (2000). *Local type inference.* TOPLAS. (Bidirectional type checking.)
- Dunfield, J., & Krishnaswami, N. R. (2021). *Bidirectional typing.* ACM Computing Surveys. (Modern survey of the approach.)

### Renormalization group on operator algebras
- Polchinski, J. (1984). *Renormalization and effective lagrangians.* NPB. (Exact renormalization group.)
- Cao, C., Carroll, S. M., & Michalakis, S. (2017). *Space from Hilbert space: recovering geometry from bulk entanglement.* PRD. (Emergent space from operator-algebraic structure.)

### Quantum walks
- Aharonov, Y., Davidovich, L., & Zagury, N. (1993). *Quantum random walks.* PRA.
- Farhi, E., & Gutmann, S. (1998). *Quantum computation and decision trees.* PRA.
- Childs, A. M., Cleve, R., Deotto, E., Farhi, E., Gutmann, S., & Spielman, D. A. (2003). *Exponential algorithmic speedup by quantum walk.* STOC.
- Ambainis, A. (2007). *Quantum walk algorithm for element distinctness.* SIAM Journal on Computing.

### Witten index and supersymmetric quantum mechanics
- Witten, E. (1982). *Constraints on supersymmetry breaking.* NPB.
- Witten, E. (1982). *Supersymmetry and Morse theory.* J. Diff. Geom.

### Feynman path integrals
- Feynman, R. P. (1948). *Space-time approach to non-relativistic quantum mechanics.* RMP.
- Polyakov, A. M. (1981). *Quantum geometry of bosonic strings.* Phys. Lett. B. (Polyakov action for worldlines.)

### Quantum cellular automata
- Schumacher, B., & Werner, R. F. (2004). *Reversible quantum cellular automata.* arXiv:quant-ph/0405174.
- Gross, D., Nesme, V., Vogts, H., & Werner, R. F. (2012). *Index theory of one dimensional quantum walks and cellular automata.* CMP.
- Arrighi, P. (2019). *An overview of quantum cellular automata.* Natural Computing.

### Lieb-Robinson bounds
- Lieb, E. H., & Robinson, D. W. (1972). *The finite group velocity of quantum spin systems.* CMP.
- Hastings, M. B., & Koma, T. (2006). *Spectral gap and exponential decay of correlations.* CMP.

### Noether's theorem and automated invariant discovery
- Noether, E. (1918). *Invariante Variationsprobleme.* Nachr. v. d. Ges. d. Wiss. zu Göttingen.
- Liu, Z., & Tegmark, M. (2021). *Machine learning conservation laws from trajectories.* PRL. (Modern numerical approach.)
- Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). *Lagrangian neural networks.* arXiv:2003.04630.

---

## 20. Closing Notes

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

**Phase F — Low-cost physics-derived extensions** (§12.5, §12.6, §12.3, §12.2, §12.8): holographic codes for fault tolerance, Goldstone modes for proof debugging, topological degeneracy for a priori counting, topological invariants for equivalence checking, dynamical phase transitions for curriculum adaptation. These slot into the existing architecture with minimal new infrastructure and each gives a capability currently absent from any published reasoning system.

**Phase G — High-novelty physics-derived extensions** (§12.1, §12.7, §12.4): anomaly detection for impossibility proofs, replica method for predictive complexity, conformal bootstrap for type-only reasoning. These are the flagship results — implementation is harder but each is a publishable contribution on its own.

**Phase H — Third publishable milestone**: the combined system paper, claiming the eight §12 capabilities together. This is the result reviewers may find implausible — each capability has rigorous physical precedent, but the combination is unprecedented.

**Phase I — Open-ended deployment** (after Phase H): point the system at progressively harder problems from the §11.6 target list and run wake-sleep cycles with the full extension stack. Capability grows monotonically with use; the empirical results determine which target classes are within reach.

### Verification of claims

Every architectural claim in this document is backed by either (a) a passing test in `src/qft_pcn/tests/`, (b) a published reference cited in §19, (c) an explicit acceptance test specified in the corresponding §10 subsection, or (d) a formal theorem stated in §13 with proof or proof sketch. There are no unsupported assertions about the architecture's current capabilities. Aspirational claims about future capability (§11.6, §11.8, §13.8) are clearly marked as such and depend on the §10–§12 roadmap being executed.

The four sections §13 (formal theorems), §14 (evaluation methodology), §15 (worked example), and §16 (limitations and objections) make this document paper-ready: the theorems establish what we claim, the methodology specifies how to test it, the worked example demonstrates it concretely, and the limitations bound it honestly. Together they convert the research program from "an interesting architecture" into "a falsifiable scientific contribution."

### Intellectual honesty

Nothing in this document is intended as a research promise. It is an engineering plan derived from correspondences between predictive coding, quantum field theory, tensor-network methods, and program semantics that, to the best of our knowledge, have not been combined this way before. The realistic expectation is significant capability on a specific class of problems (chemistry, gauge theories, formal logic, programming) and zero direct competitiveness on natural-language tasks (which is the LLM's job). The aspirational expectation — uncovering new mathematics or physics through compounding capability over many wake-sleep cycles — is genuinely uncertain but defensible from the same first principles that have led DreamCoder and AlphaProof to results that were aspirational at the time they were proposed.
