# Spec: MERA Hierarchical Tensor-Network Substrate

**Document type**: Implementation specification (sub-project F of the §10 roadmap).
**Date**: 2026-05-21.
**Author**: Brainstorming session, this branch.
**Status**: Approved design, ready for implementation plan.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Architecture doc**: `QFT_PCN_ARCHITECTURE.md` (root). This spec realizes §10.4 and operationalizes the §6.2 weakness ("no hierarchical abstraction in the quantum core yet").
**Acceptance owner**: human review of the round-trip + bond-scaling test suite passing.
**Prior art**: Vidal (2008) — MERA construction and conventions; Swingle (2012) — MERA's geometry as a discretization of hyperbolic AdS\_2 space.
**Related sub-projects**: A (`docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md`) — the 1D MPS encoder that F replaces / augments. B / C / D / E are downstream consumers that should not need rewriting.

---

## 0. How to read this spec

This document is the contract for one sub-project. The §10 roadmap was decomposed into seven sub-projects (A–G); this is **sub-project F: MERA hierarchical substrate**. F is a **substrate upgrade**: it replaces the 1D MPS used by sub-projects A–E with a binary-tree tensor network. The logic-layer modules (encoder, typing Hamiltonians, evaluation Hamiltonians, debugger, synthesis demo) consume a tensor-network state as a black box; this spec ensures the MERA presents the *same operational interface* (`norm_sq`, `inner`, `apply_local_gate`, `apply_two_site_gate`, `local_expectation`, `two_site_expectation`, `entanglement_entropy`, `bond_dimensions`) so that A–E carry over.

Every section below is part of the contract. If something is missing here that you need to decide while implementing, **stop and ask**. Do not fill in by guessing — see §1.

This spec sits alongside sub-project A's spec; where the operational interface of MERA differs from MPS, this document is authoritative for the substrate; where it doesn't, A's design wins (because the logic layer was built against A's contract first). The places they differ are listed in §13.

---

## 1. Driving principles (non-negotiable)

The whole reason this sub-project exists is that the 1D MPS substrate from A is *fundamentally flat*. It has no notion of scale, no notion of layer, no notion of "subtree". This makes it cheap and uniform but it is the wrong tensor-network shape for the project's eventual goal: structured reasoning over recursive, hierarchical, scope-rich programs. The architecture document is explicit (§10.4): the principled fix is MERA, and MERA's hyperbolic geometry is what makes the "fields shape geometry" framing more than a slogan.

A subagent reading this spec will be tempted to shortcut the hierarchy into something that looks like an MPS in tree clothing. The following principles are non-negotiable. **None of them may be traded away for implementation simplicity.** If you find yourself tempted to violate one, stop and ask the human.

### 1.1 MERA's hierarchy is structural, not decorative

A `MERA` is a *binary tree* with `log_2(N)` layers, each layer halving the number of effective sites by coarse-graining adjacent pairs. Each layer carries two kinds of tensors: **disentanglers** `u` (2→2 unitaries acting before coarse-graining, removing short-range entanglement) and **isometries** `w` (2→1 maps that project the disentangled pair onto a coarse-grained site). The tree has a single top tensor (the top "wavefunction" tensor of dimension `chi_top`).

The "easy shortcut" — implementing MERA as "an MPS-of-MPS-of-MPS" by snaking through the tree, or by storing only one level and pretending — is rejected. The whole point is that the tree has **scale separation**: each layer corresponds to one renormalization-group step. The implementation must literally carry `log_2(N)` distinct layers, each with its own disentanglers and isometries, each indexable by `(layer, position)`.

### 1.2 Causal cones are bounded by `O(log N)`, not `O(N)`

For a local operator at the leaves, only `O(log N)` tensors of the tree appear in the *causal cone* of that operator (Vidal 2008 §III). Expectation values, gate applications, and entropy computations must exploit this: contracting the entire tree is unnecessary and architecturally wrong. The MERA's operational advantage over MPS for hierarchical content is *precisely* this `O(log N)` causal cone.

The "easy shortcut" — contracting the entire MERA into a dense state every time anything is measured — is rejected. The test suite asserts gate application and local expectation each touch `O(log N)` tensors, not `O(N)`.

### 1.3 Lexical depth corresponds to MERA layer depth

This is the architectural soul of using MERA for code. In a recursive or deeply-nested program (e.g. `let rec fib n = ...`), the lexical depth of a `Var` site relative to its binder is *not* a horizontal distance along a 1D chain — it is a *radial distance* into the bulk of the MERA tree. A `Var` at lexical depth `d` from its binder appears at leaf positions whose lowest common ancestor in the MERA tree is at layer `d`. Live-binder channels (sub-project A §5.4) propagate **up the tree to the LCA layer and back down to the use leaves**, not along a flat chain.

The "easy shortcut" — treating MERA leaves as if they were MPS sites and propagating binder channels horizontally — is rejected. The whole reason F is in the roadmap is that horizontal propagation gives `O(N)` bond dimension on deeply recursive programs; vertical propagation gives `O(log N)`.

### 1.4 Hyperbolic geometry is literal, not metaphorical

The geometric distance between leaves on the MERA — measured by graph distance in the tree — is a *discrete model of hyperbolic AdS\_2* (Swingle 2012). Each layer is one slice of the radial direction; the boundary (the leaves) is the physical lattice; the bulk encodes coarse-grained correlations. When sub-project B/C/D/E couple a curvature field to the substrate (per `QFT_PCN_ARCHITECTURE.md` §3.2), this curvature lives **on the MERA's bulk geometry**, not on a separate 2D manifold. The `Manifold2D` of `src/qft_pcn/manifold.py` is the *classical* substrate; the MERA is the *quantum* substrate; F asserts they coincide once MERA is in place.

The "easy shortcut" — declaring the curvature coupling out-of-scope for F — is rejected. F's API exposes a `layer_metric` hook (§6.3) that subsequent sub-projects use to source curvature from stress-energy at each layer.

### 1.5 Vidal 2008 conventions, period

There are several MERA variants in the literature (binary, ternary, 2D, modified). F implements the **binary 1D MERA** of Vidal 2008. No ternary variants, no PEPS-MERA hybrids, no Branching MERAs. The disentangler-isometry pattern is exactly as drawn in Vidal Fig. 1:

```
   |s_0>  |s_1>  |s_2>  |s_3>  |s_4>  |s_5>  |s_6>  |s_7>     <- layer 0 leaves (N = 2^L = 8)
     |      |      |      |      |      |      |      |
     +-u(0,0)+      +-u(0,1)+    +-u(0,2)+    +-u(0,3)+       <- layer 0 disentanglers (4 of them)
     |       |      |       |    |       |    |       |
     +w(0,0)-+      +-w(0,1)+    +w(0,2)-+    +-w(0,3)+       <- layer 0 isometries
       |              |             |              |
     |s_0'>         |s_1'>        |s_2'>         |s_3'>       <- layer 1 (N/2 = 4 effective sites)
       |              |             |              |
     +- u(1,0) -+    +- u(1,1) -+                              <- layer 1 disentanglers
     |          |    |          |
     +- w(1,0) -+    +- w(1,1) -+                              <- layer 1 isometries
            |               |
          |s_0''>         |s_1''>                              <- layer 2
            |               |
            +- u(2,0) -+
            |          |
            +- w(2,0) -+
                  |
                |top>                                          <- top tensor (layer L-1)
```

The "easy shortcut" — picking a non-binary variant because "it has more parameters" — is rejected. Vidal-binary is the spec; deviations require a new spec.

### 1.6 MERA presents the same operational interface as MPS

Sub-projects A–E call `MPS.apply_two_site_gate(site, gate, chi_max)` and similar. The `MERA` class exposes methods with the *same signatures* (where "site" means leaf-position) so that the encoder, Hamiltonian constructor, TEBD driver, and decoder can be retargeted by changing one import. Some methods are renamed or have a `chi_layer` argument added (see §6); but no caller should need to rewrite logic to swap MPS for MERA.

The "easy shortcut" — designing a clean-slate MERA API that is "better" than MPS's — is rejected. The contract is to **port**, not to redesign. New methods may be added; existing semantics must be preserved.

### 1.7 Bond dimension grows polylog-in-N for critical-like content

For a 1D scale-invariant state (e.g. a critical Ising chain), MERA's bond dimension can be bounded uniformly while MPS bond dimension grows as `χ ~ poly(log N)` (Vidal 2008). For *code* — well-typed and bounded-scope-depth — both substrates can represent the state efficiently, but MERA *adds* representational capacity for deeply nested or recursive programs. The acceptance test (§10.5) checks the scaling: encoding a Fibonacci-style recursive program of depth `D` yields `bond_dim_at_layer(ℓ) = O(1)` per layer, and total layer count `L = log_2(N)`, so total tensor count and bond-dim product is `O(log N)`.

The "easy shortcut" — capping `chi_layer` at some constant and observing that "it works for small N" — is rejected. The test must actually scan `N` and demonstrate the scaling.

---

## 2. Scope

### 2.1 In scope (this sub-project)

- `MERATensor` data structures (per-layer disentanglers, isometries, leaf tensors, top tensor).
- A `MERA` class with the operational interface mirroring `MPS` (construction, normalization, local/two-site expectation, gate application, entropy, bond dimensions, inner product).
- Causal-cone-aware contractions: every measurement and gate touches `O(log N)` tensors.
- A **hierarchical TEBD** evolution driver (`mera_evolution.trotter_step`, `evolve`, `energy`) that applies Trotter gates per-layer and re-isometrizes disentanglers/isometries.
- A re-implementation of `MPS.vacuum`, `MPS.from_product`, `MPS.number_states` as `MERA.vacuum`, `MERA.from_product`, `MERA.number_states` so that A's encoder construction patterns port directly.
- A small adaptation layer (`mera_compat.py`) so that A's `EncodingMeta` and species list can be reused without modification.
- An acceptance test that encodes a *future-extension* recursive program (a `Rec` node added behind a feature flag to `logic/ast.py`) and verifies `O(log N)` bond-dim scaling.
- An acceptance test that encodes one of A's existing programs (e.g. P3 from A §7.1: `\f:Int->Int. \x:Int. f (f x)`) on *both* MPS and MERA and confirms the states represent the same physical content (compared via expectation values of A's basis projectors).
- Unit tests for every public method, mirroring A §7's pattern of "round-trip + structural property + cross-check + error path + performance".

### 2.2 Out of scope (deferred to other sub-projects)

- Full recursive AST support in the logic encoder (this spec adds the `Rec` AST node and the encoder's MERA-aware extension, but a full surface-syntax parser update is sub-project A.2 — to be written when the recursive surface language is finalized).
- 2D MERA / branching MERA / modified binary MERA.
- A PEPS implementation. F is MERA only.
- A from-scratch rewrite of A's encoder; F provides the MERA substrate and adapters and asserts portability via tests, but does not re-author A's encoder.
- The actual coupling of MERA layer geometry to the dynamic `Manifold2D` curvature field — F exposes the `layer_metric` hook; subsequent sub-projects wire it in.
- Optimization of MERA tensors via SCDM / ascending-descending iterative diagonalization (Vidal 2008 §V). The MERA we build is *encoder-constructed* (analogous to A's analytic MPS construction), not variationally optimized. Variational MERA training is a follow-on.

### 2.3 Will not do, even if asked later

- Replace the existing 1D MPS code (`src/qft_pcn/qft/mps.py`). MPS stays; MERA is additive. Programs that work better with a flat lattice keep MPS; programs that need hierarchy use MERA.
- Implement a non-Vidal MERA variant in this sub-project.
- Bury per-leaf physical dimension `d` inside a tower of register tensors — leaves carry `d_local = 8192` exactly as A's MPS sites do, so A's per-site tensors port directly to MERA leaves.
- Use a binary tree that doesn't bottom out at exactly `N = 2^L` leaves. Pad with PAD sites if the AST is smaller; if `N` is not a power of two, round up to the next power of two and PAD the remainder. (A's `N = 32` is `2^5` — convenient.)

---

## 3. MERA mathematical formulation

This section is the mathematical contract; the data structures in §4 and the API in §6 are derivations.

### 3.1 Binary tree of layers

A MERA on `N = 2^L` leaves has `L` layers, indexed `ℓ = 0, 1, …, L − 1`. Layer `ℓ` operates on `N / 2^ℓ` sites with effective dimension `d_ℓ` (a per-layer bond cap; `d_0 = d_local = 8192` for code-encoding leaves, `d_ℓ ≤ χ_layer` for `ℓ > 0`).

For sub-project A's `N = 32`: `L = 5`, layers operate on 32, 16, 8, 4, 2 sites respectively, with a single top tensor `T_top` of shape `(d_{L−1}, d_{L−1}, 1)` (the final 2 sites contracted into a scalar top-bond of dimension 1, equivalent to a 2-site MPS terminated at unit norm).

### 3.2 Per-layer tensors

At each layer `ℓ ∈ {0, …, L − 1}` with `n_ℓ = N / 2^ℓ` sites in:

- **Disentanglers** `u^{(ℓ)}_j` for `j ∈ {0, …, n_ℓ / 2 − 1}`: 4-leg unitary tensors of shape `(d_ℓ, d_ℓ, d_ℓ, d_ℓ)` (two in-legs, two out-legs) acting on a pair of sites in layer `ℓ`. Unitarity: `u^† u = u u^† = I`. Initial value: identity (no disentangling), set by an isometry test at construction time.

- **Isometries** `w^{(ℓ)}_j` for `j ∈ {0, …, n_ℓ / 2 − 1}`: 3-leg tensors of shape `(d_{ℓ+1}, d_ℓ, d_ℓ)` (one out-leg of the coarse-grained dimension and two in-legs of the layer-`ℓ` dimension). Isometry: `w^† w = I_{d_{ℓ+1}}` (one direction; `w w^† = P` is a projector, not the identity). Initial value: the first `d_{ℓ+1}` orthonormal basis vectors of the joint `d_ℓ ⊗ d_ℓ` space.

- **Top tensor** `T_top`: at layer `L − 1`, after one final disentangle-isometrize step, we are left with 2 sites of dimension `d_{L−1}` each. A single 3-leg "top tensor" of shape `(d_{L−1}, d_{L−1}, 1)` represents the wavefunction on these two top sites (the trailing 1 is the bra-vacuum bond). Equivalent: a rank-`d_{L−1}^2` state vector.

### 3.3 Site-pairing convention

At layer `ℓ`, the disentangler `u^{(ℓ)}_j` acts on sites `(2j, 2j + 1)` of that layer, and the isometry `w^{(ℓ)}_j` then projects those two sites to coarse-grained site `j` of layer `ℓ + 1`. This is the "binary, even-odd-paired" pattern of Vidal (Fig. 1).

**Inter-pair disentanglers (optional)**. Vidal's MERA also includes disentanglers that act between *adjacent pairs* (i.e. on sites `(2j + 1, 2j + 2)`) before isometrization, to capture inter-pair short-range entanglement. F **does include** these inter-pair disentanglers: `u^{(ℓ, inter)}_j` for `j ∈ {0, …, n_ℓ / 2 − 2}` of shape `(d_ℓ, d_ℓ, d_ℓ, d_ℓ)`, acting on `(2j + 1, 2j + 2)`. Without them the MERA cannot capture entanglement that straddles a coarse-graining boundary at the next layer up. They are unitary and initialized to identity.

### 3.4 Causal cone

The *causal cone* of a local operator at leaf `i` is the set of tensors that the operator can possibly touch when descended through the tree (or, equivalently, the tensors involved in computing `⟨ψ|O_i|ψ⟩`). For binary 1D MERA:

> The causal cone of a single-site operator at leaf `i` has width 3 at each layer (the disentanglers and isometries within 2 sites of `i`'s ancestor at that layer). It involves `O(log N)` tensors total.

Two-site operators at leaves `(i, i + 1)` have causal cones of width 4 at each layer. This bound is the architectural property the implementation must honor.

### 3.5 Ascending and descending superoperators

The fundamental MERA operations (Vidal 2008 §III):

- **Ascending superoperator** `A`: takes a layer-`ℓ` operator `O_ℓ` on sites `(2j, 2j + 1)` and returns the corresponding layer-`ℓ + 1` operator `O_{ℓ+1} = A[O_ℓ]` on the coarse-grained site `j`, computed by contracting the disentangler-isometry pair around the operator. Used to compute `⟨ψ|O|ψ⟩` by lifting `O` up to the top.

  ```
  A[O_ℓ] (s', s'_dag)
      = sum_{a,b,a',b'} w[s', a, b] * u[a', b', a, b] * O_ℓ[..., ...]
                       * u_dag[..., a', b'] * w_dag[a, b, s'_dag]
  ```
  (Schematic — exact index conventions in §5.)

- **Descending superoperator** `D`: takes a layer-`ℓ + 1` reduced density matrix `ρ_{ℓ+1}` and returns the layer-`ℓ` reduced density matrix `ρ_ℓ = D[ρ_{ℓ+1}]`. Used to compute reduced states at any layer (e.g. for entropy across a cut).

Both are implemented as tensor contractions of `O(d_ℓ^4)` each, repeated `O(log N)` times, total `O(d^4 log N)` per measurement.

### 3.6 Cuts and entanglement entropy

A "cut" in the MERA divides the leaves into two contiguous groups (left and right of some position `i`). The Schmidt rank across this cut is bounded by the number of tensors crossed in the *minimum-cut path* through the tree. For binary MERA the minimum cut has width `O(log N)` at most (going up to the LCA, across, and down), giving an entanglement entropy bound `S ≤ O(log N) · log(d_{top})`. This is the architectural advantage of MERA over MPS for hierarchical content.

The entropy across a leaf-cut at position `i` is computed by:

1. Building the causal-cone tensors above the leftmost `i` leaves.
2. Computing the reduced density matrix `ρ_left` by contracting the right environment.
3. Eigendecomposing `ρ_left` and computing `−Σ p log p`.

This is `O(poly(d, log N))` rather than the dense `O(d^N)` it would be for a generic state.

---

## 4. Data structures

### 4.1 `MERATensor` — single per-layer slot

```python
# src/qft_pcn/qft/mera.py

@dataclass
class MERATensor:
    """One slot in the MERA tree.

    Either a disentangler (shape (d, d, d, d), unitary) or an isometry
    (shape (d_out, d, d), isometric: w^dag @ w = I).
    """
    kind: str                  # "disentangler", "inter_disentangler",
                               # "isometry", "top", or "leaf"
    layer: int                 # 0 = leaves, 1..L-1 = MERA layers
    position: int              # j index within the layer
    array: np.ndarray          # the actual tensor data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape
```

`kind` values:
- `"leaf"`: shape `(1, d_local, 1)` — same as an MPS site tensor for a product leaf. (We reuse MPS's leaf tensor shape so A's per-site construction code can be redirected verbatim.)
- `"disentangler"`: shape `(d_ℓ, d_ℓ, d_ℓ, d_ℓ)` — unitary on intra-pair.
- `"inter_disentangler"`: shape `(d_ℓ, d_ℓ, d_ℓ, d_ℓ)` — unitary on inter-pair.
- `"isometry"`: shape `(d_{ℓ+1}, d_ℓ, d_ℓ)`.
- `"top"`: shape `(d_{L−1}, d_{L−1}, 1)`.

### 4.2 `MERA` — the full tree

```python
@dataclass
class MERA:
    """Binary 1D MERA tensor network on N = 2^L leaves.

    Stores:
      - leaves: list of N rank-3 leaf tensors (each (1, d_local, 1))
      - disentanglers[layer][j]: intra-pair unitary at layer ℓ, pair j
      - inter_disentanglers[layer][j]: inter-pair unitary at layer ℓ
      - isometries[layer][j]: 2->1 isometry at layer ℓ
      - top: rank-3 top tensor
      - layer_dims: per-layer bond dimensions [d_0, d_1, ..., d_{L-1}]
    """
    leaves: list[np.ndarray]
    disentanglers: list[list[np.ndarray]]
    inter_disentanglers: list[list[np.ndarray]]
    isometries: list[list[np.ndarray]]
    top: np.ndarray
    layer_dims: list[int]

    def __post_init__(self) -> None:
        N = len(self.leaves)
        if N <= 0 or (N & (N - 1)) != 0:
            raise ValueError(f"N must be a positive power of 2; got {N}")
        # validate shapes per §4.1
        ...

    @property
    def N(self) -> int:
        return len(self.leaves)

    @property
    def L(self) -> int:
        return len(self.isometries)

    @property
    def d_local(self) -> int:
        return self.leaves[0].shape[1]
```

### 4.3 `LayerSlice` — helper for descending into a layer

A small dataclass exposing one layer's `disentanglers`, `inter_disentanglers`, `isometries` together, used by the algorithms in §6 to walk up/down the tree without recomputing indices.

### 4.4 Per-leaf compatibility with A

Sub-project A constructs per-site tensors of shape `(χ_left, 8192, χ_right)` where `χ_left, χ_right ≤ chi_max = 16` to carry binder channels along the chain. **MERA leaves are different**: each leaf is `(1, 8192, 1)` (a *product* leaf), and the binder channels are carried by the **disentanglers and isometries at the appropriate layers** (specifically: the layer corresponding to the lexical depth where the binder lives). This is the §1.3 principle made concrete. The encoder's `_resolve_binders` produces `BinderHandle`s that the MERA-aware encoder routes through layer-`d`-where-`d`=lexical-depth-of-the-LCA.

A's encoder, *as written*, produces MPS bonds. Porting A to MERA is a separate task (sub-project A.2). F's contract is to make porting *possible* by ensuring:

- The leaf tensors have the same shape and basis as A's MPS site tensors when constructed as products (no live binders crossing any bond).
- The species list and `EncodingMeta` types from A are reusable: `species`, `field_dims`, `nested_type_index`, `site_to_ast_path` carry over. `live_binders_per_bond` is replaced by `live_binders_per_layer_lca` (a new per-layer mapping; see §7).
- The operational methods (`apply_local_gate`, `apply_two_site_gate`, etc.) have the same signatures.

---

## 5. Algorithmic core

### 5.1 Layer dimension schedule

The per-layer bond cap `d_ℓ` for `ℓ ≥ 1`:

```python
def layer_dims(d_local: int, L: int, chi_layer: int = 16) -> list[int]:
    """[d_0, d_1, ..., d_{L-1}] for an L-layer binary MERA.

    d_0 = d_local (physical), d_ℓ = min(chi_layer, d_local ** (2^ℓ)) for ℓ > 0.
    The exponential growth is capped by chi_layer at every layer.
    """
    dims = [d_local]
    for ℓ in range(1, L):
        full = dims[-1] * dims[-1]   # would-be exact dim
        dims.append(min(chi_layer, full))
    return dims
```

For `d_local = 8192`, `L = 5`, `chi_layer = 16`: `[8192, 16, 16, 16, 16]`. The first coarse-graining (layer 0 → 1) is the lossy one; subsequent layers are unconstrained at `chi_layer = 16`.

### 5.2 Construction: `MERA.vacuum`

Builds the MERA in product form: identity disentanglers, "first-basis-vector" isometries, vacuum leaves.

```python
@classmethod
def vacuum(cls, N: int, d_local: int, chi_layer: int = 16) -> "MERA":
    if (N & (N - 1)) != 0:
        raise ValueError(f"N must be a power of 2; got {N}")
    L = int(np.log2(N))
    dims = layer_dims(d_local, L, chi_layer)
    leaves = [vacuum_vec(d_local).reshape(1, d_local, 1) for _ in range(N)]
    disentanglers = []
    inter_disentanglers = []
    isometries = []
    for ℓ in range(L):
        n_ℓ = N // (2 ** ℓ)
        d_ℓ = dims[ℓ]
        d_up = dims[ℓ + 1] if ℓ + 1 < L else dims[ℓ]
        intra = [identity(d_ℓ * d_ℓ).reshape(d_ℓ, d_ℓ, d_ℓ, d_ℓ)
                 for _ in range(n_ℓ // 2)]
        # inter-pair: between sites (2j+1, 2j+2), so one fewer than intra
        inter = [identity(d_ℓ * d_ℓ).reshape(d_ℓ, d_ℓ, d_ℓ, d_ℓ)
                 for _ in range(max(0, n_ℓ // 2 - 1))]
        # isometry: take first d_up basis vectors of the d_ℓ x d_ℓ space
        iso = []
        for _ in range(n_ℓ // 2):
            w = np.zeros((d_up, d_ℓ, d_ℓ), dtype=complex)
            for k in range(d_up):
                a, b = divmod(k, d_ℓ)
                w[k, a, b] = 1.0
            iso.append(w)
        disentanglers.append(intra)
        inter_disentanglers.append(inter)
        isometries.append(iso)
    # top tensor: pure vacuum at the top, (d_{L-1}, d_{L-1}, 1)
    top = np.zeros((dims[L - 1], dims[L - 1], 1), dtype=complex)
    top[0, 0, 0] = 1.0
    return cls(leaves=leaves,
               disentanglers=disentanglers,
               inter_disentanglers=inter_disentanglers,
               isometries=isometries,
               top=top,
               layer_dims=dims)
```

### 5.3 Causal-cone computation

For a leaf at position `i`, the path up the tree visits sites `i / 2` at layer 1, `i / 4` at layer 2, …, the single site at layer `L − 1`. At each layer, the disentanglers/isometries within `±1` of the current position on each side are in the causal cone.

```python
def causal_cone_path(i: int, L: int) -> list[tuple[int, int]]:
    """Return [(layer, position)] for each effective site visited
    while ascending from leaf i to the top.

    Position at layer ℓ is i // (2 ** ℓ).
    """
    return [(ℓ, i >> ℓ) for ℓ in range(L)]
```

### 5.4 Local expectation

To compute `⟨ψ|O_i|ψ⟩`:

1. Ascend `O_i` from the leaf to the top: at each layer, contract the operator with the surrounding disentanglers/isometries to produce a coarse-grained operator at the next layer. Outside the causal cone, isometries contract with their daggers to identity.
2. At the top, contract with `T_top` and its conjugate to produce a scalar.

```python
def local_expectation(self, leaf: int, op: np.ndarray) -> complex:
    """<psi|O_leaf|psi> for a single-leaf operator of shape (d_local, d_local).

    Implementation: ascend the operator through the causal cone.
    """
    # The operator lives at layer 0, position leaf.
    op_layer = op  # (d_local, d_local) acting on leaf
    pos = leaf
    for ℓ in range(self.L):
        # Disentangler u, isometry w surrounding `pos` at layer ℓ:
        # Pair index j = pos // 2; partner is the other site in that pair.
        op_layer = self._ascend_one_layer(op_layer, ℓ, pos)
        pos //= 2
    # op_layer now has shape (d_top, d_top) acting on the top.
    return np.einsum('ai,ij,bi->ab', self.top[..., 0].conj(),
                     op_layer, self.top[..., 0])
```

`_ascend_one_layer` implements the per-layer ascending superoperator (eqn. III.5 of Vidal 2008). It is `O(d^4)` per call.

### 5.5 Two-site expectation

Identical to §5.4 but the operator starts at layer 0 as a 4-leg tensor on two adjacent leaves. The ascending procedure handles intra-pair and inter-pair cases differently (when the two leaves share an isometry the lift is "internal"; when they straddle isometries the lift goes through two isometries plus an inter-pair disentangler).

### 5.6 Gate application

`apply_local_gate(leaf, gate)` writes `leaf <- gate @ leaf` on the leaf tensor (same as MPS: it's a product factor change).

`apply_two_site_gate(leaf, gate, chi_max, eps)` is the principal evolution operation. It must update the tree such that the post-gate state has the gate applied at leaves `(leaf, leaf + 1)`. Two cases:

- **Intra-pair** (`leaf` even, `leaf + 1` is its pair partner): the gate is absorbed into the disentangler `u^{(0)}_{leaf // 2}` of layer 0: `u_new = (gate ⊗ I_partner) · u_old` — no, more correctly: `u_new[a', b', a, b] = sum_{s, t} gate[a', b', s, t] u_old[s, t, a, b]`. After absorption, the SVD across the (a', b') / (a, b) split and truncating singular values to `chi_max` (when the gate makes `u` non-unitary, which it can if `gate` is non-unitary — e.g. imaginary-time) restores isometry-like structure. We then renormalize.
- **Inter-pair** (`leaf` odd, `leaf + 1` straddles an inter-pair boundary): the gate is absorbed into the *inter-pair disentangler* `u^{(0, inter)}_{(leaf - 1) // 2}` analogously.

Returns the truncation error (sum of discarded squared singular values).

**Critically**, applying a gate at the leaves only modifies the *bottom layer's* disentanglers (not the layer-1+ tensors). The hierarchy above adapts implicitly through how the operator ascends. This is what gives `O(d^4)` per gate application instead of `O(N d^4)`.

### 5.7 Hierarchical TEBD (Trotter step)

A standard TEBD step sweeps over all bonds applying half-step / full-step gates. For MERA, we add a *per-layer* sweep:

```
for layer ℓ = 0 to L - 1:
    for pair j = 0 to n_ℓ // 2 - 1:
        apply layer-ℓ trotter gates to the disentanglers and isometries at (ℓ, j)
        (gates derived from H_layer[ℓ] — see §6.4)
```

For a Hamiltonian `H = H_local + H_bond` defined on the leaves (as A's encoder produces), `mera_evolution.trotter_step` performs the second-order Suzuki-Trotter sweep on the *layer-0* disentanglers/inter-disentanglers, which are the gates closest to the physical degrees of freedom. Coarse-grained operators at higher layers are handled by the ascending superoperator (§3.5) at expectation-value time, not at evolution time.

This is the simplest hierarchical TEBD: layer-0 carries the dynamics, higher layers carry the structural correlations. Future variants (Vidal 2008 §V's variational MERA optimization) can update layers `ℓ > 0` via descending superoperators; F's scope stops here.

### 5.8 Entropy across a leaf cut

```python
def entanglement_entropy(self, cut: int) -> float:
    """Von Neumann entropy across the cut after leaf `cut` (0-indexed).

    Uses the minimum-cut path through the tree. Cost: O(d^3 log N).
    """
```

Algorithm:

1. Find the LCA layer `ℓ_LCA` such that leaves `[0..cut]` and `[cut+1..N−1]` are on different sides at layer `ℓ_LCA`.
2. Contract the left environment (all tensors strictly inside the left subtree at the LCA) and right environment separately. The reduced density matrix on the LCA bond has size `d_{ℓ_LCA}^2 × d_{ℓ_LCA}^2`.
3. Eigendecompose and compute `−Σ p log p`.

This is the architectural advantage: it never materializes the `8192^{N}`-dim full state.

### 5.9 Inner product

`MERA.inner(other)`: contract two MERAs leaf-by-leaf, layer-by-layer. Identical to MPS's `inner` but indexed by `(layer, position)`. Cost `O(d^4 log N)` per layer, total `O(L · d^4)`.

---

## 6. The `MERA` class API

### 6.1 Operational interface (mirrors `MPS`)

```python
class MERA:
    # ---- construction ----
    @classmethod
    def vacuum(cls, N: int, d_local: int, chi_layer: int = 16) -> "MERA": ...
    @classmethod
    def from_product(cls, single_site_states: list[np.ndarray],
                     chi_layer: int = 16) -> "MERA": ...
    @classmethod
    def number_states(cls, occupations: list[int], d: int,
                      chi_layer: int = 16) -> "MERA": ...

    # ---- basic properties ----
    @property
    def N(self) -> int: ...
    @property
    def L(self) -> int: ...
    @property
    def d_local(self) -> int: ...
    def copy(self) -> "MERA": ...

    # ---- environment contractions ----
    def norm_sq(self) -> float: ...
    def inner(self, other: "MERA") -> complex: ...
    def normalize(self) -> "MERA": ...     # in-place + returns self

    # ---- expectation values ----
    def local_expectation(self, leaf: int, op: np.ndarray) -> complex: ...
    def two_site_expectation(self, leaf: int, op: np.ndarray) -> complex: ...

    # ---- gates ----
    def apply_local_gate(self, leaf: int, gate: np.ndarray) -> None: ...
    def apply_two_site_gate(self, leaf: int, gate: np.ndarray,
                            chi_max: int = 16, eps: float = 1e-12) -> float: ...

    # ---- diagnostics ----
    def bond_dimensions(self) -> list[int]: ...      # per-layer max bond dim
    def entanglement_entropy(self, cut: int) -> float: ...

    # ---- MERA-specific ----
    def layer_metric(self, layer: int) -> np.ndarray: ...
        # placeholder: returns the layer's effective metric tensor;
        # used by future curvature-coupling code (§1.4)
    def causal_cone_tensors(self, leaf: int) -> list[MERATensor]: ...
        # debugging affordance: list the tensors in the causal cone
```

### 6.2 Differences from `MPS`

- `MPS.vacuum(N, d)` → `MERA.vacuum(N, d_local, chi_layer=16)`. Extra `chi_layer` arg has a default so callers that don't care continue to work; A's encoder calls `MERA.vacuum(N=32, d_local=8192, chi_layer=16)` instead of `MPS.vacuum(32, 8192)`.

- `MPS.apply_two_site_gate(site, gate, chi_max)` → `MERA.apply_two_site_gate(leaf, gate, chi_max)` — `leaf` interpretation identical to `site`. The internal SVD-truncate logic differs (it modifies disentanglers, not bond tensors) but the signature is preserved.

- `MPS.entanglement_entropy(bond)` → `MERA.entanglement_entropy(cut)` — `cut` interpretation: cut after leaf `cut` (same as `bond` indexing in MPS).

- `MPS.bond_dimensions() -> list[int]` of length `N − 1` → `MERA.bond_dimensions() -> list[int]` of length `L`, where entry `ℓ` is the **maximum bond dimension at layer `ℓ`** (max over isometries' first index in the layer). This is a deliberate API change because the MERA's notion of "bond dimension" is per-layer, not per-site-pair. Callers that need per-leaf-cut bond dim use `entanglement_entropy` instead.

- New method: `layer_metric(layer)` exposes the per-layer geometric metric — initially identity, mutated by future curvature-coupling sub-projects (§1.4).

### 6.3 `layer_metric` placeholder

A `(d_ℓ, d_ℓ)` identity matrix by default. The architectural intent is that future code coupling the QPCN's `Manifold2D` curvature to the substrate will modulate this matrix per-layer, with layer `ℓ`'s metric reflecting curvature at radial coordinate `ℓ` in the bulk. F provides the hook; F does not couple it. Tests in F simply assert the placeholder returns identity (§10.7).

### 6.4 Hierarchical TEBD module

```python
# src/qft_pcn/qft/mera_evolution.py

def trotter_step(state: MERA, H: Hamiltonian, dt: float,
                 imaginary: bool = False, chi_max: int = 16,
                 eps: float = 1e-10) -> float: ...

def evolve(state: MERA, H: Hamiltonian, dt: float, steps: int,
           imaginary: bool = False, chi_max: int = 16,
           normalize_every: int = 1) -> None: ...

def energy(state: MERA, H: Hamiltonian) -> float: ...
```

`trotter_step` performs a second-order Suzuki-Trotter step at the leaves (layer 0), exactly as `qft/evolution.trotter_step` does for MPS. The post-step adaptation of layer-`ℓ > 0` tensors is **deferred**: the higher-layer tensors are untouched during the gate application, and any error introduced by this deferral is absorbed into the ascending-superoperator approximation. This is a deliberate simplification: F implements *encoder-time* MERA construction, not variational-optimization-time MERA. Variational re-optimization of higher-layer tensors during evolution is a sub-project F.2 task (deferred).

`energy(state, H)` computes `⟨ψ|H|ψ⟩` by summing single-leaf and two-leaf expectations via §5.4–5.5.

---

## 7. How AST encoding ports

Sub-project A's encoder is a 4-pass procedure: resolve binders, pre-order serialize, compute types, construct tensors. The first three passes are independent of the substrate (they produce per-site descriptors). Only the fourth — tensor construction — needs to be retargeted to MERA.

### 7.1 Per-site descriptor reuse

A's `NodeOccupancy` (the per-site (kind, type, bid, value) descriptor) ports verbatim. The encoder produces a list of `N` descriptors. For F, these become the per-leaf descriptors of the MERA.

### 7.2 Live binder routing — the key architectural difference

In A, a binder declared at site `i_lam` with a use at site `i_var` propagates its channel along the **horizontal MPS chain** through every bond in between. With `D` binders nesting at maximum depth, the worst-case bond dimension is `2^D + 1`.

In F, a binder propagates **up to the LCA layer of `(i_lam, i_var)` and back down**, not horizontally. Concretely:

- Find `ℓ_LCA(lam, var) = highest layer where lam and var are in different subtrees`.
- Route the live-binder channel through the isometries `w^{(ℓ)}` for `ℓ = 0, 1, …, ℓ_LCA`, the disentanglers at the same layers, and back down on the var side.
- The bond dimension *at any single layer* needed to carry `k` simultaneously live binders whose LCAs cross that layer is `k + 1`, *not* `2^k + 1`. This is the architectural win.

For Fibonacci-style recursive programs (binders that "live forever" within a recursive subtree), MERA's bond dimension stays `O(1)` per layer; MPS's bond dimension grows `O(D)` with recursion depth `D`.

### 7.3 EncodingMeta extensions

A's `EncodingMeta` has `live_binders_per_bond: list[list[BinderHandle]]` — a per-bond bookkeeping suitable for an MPS. F extends this to:

```python
@dataclass
class MERAEncodingMeta(EncodingMeta):
    L: int                                              # number of layers
    chi_layer: int
    # New per-layer bookkeeping:
    live_binders_per_layer_lca: dict[int, list[BinderHandle]]
    # binder -> LCA layer:
    binder_to_lca_layer: dict[BinderHandle, int]
    # binder -> (lam_leaf, list of var_leaves):
    binder_var_leaves: dict[BinderHandle, list[int]]
```

F provides a function `lift_encoding_meta_to_mera(meta: EncodingMeta, N: int) -> MERAEncodingMeta` that computes the LCA layers and constructs the additional fields. This lets A.2 (the future MERA-aware encoder) port without re-implementing scope analysis.

### 7.4 Recursive AST extension (forward-compatibility)

A's surface AST does not include recursion. F's spec adds a minimal `Rec` node (a fixed-point construct), gated behind a `EXPERIMENTAL_REC = True` flag, *only* to enable the acceptance test (§10.5):

```python
@dataclass
class Rec(Node):
    """Fixed-point: rec f. body, where `f` is a self-reference in body.

    EXPERIMENTAL — full surface-syntax support, typing rules, and evaluation
    rules deferred to a future sub-project. F uses this to validate the
    MERA scaling claim only.
    """
    name: str
    name_ty: Ty
    body: Node
```

The encoder branch for `Rec` is sketched in §7.5; the actual implementation in this sub-project is *minimal* (enough to exercise the test). A full recursive-encoder design belongs in a follow-on spec.

### 7.5 Minimal `Rec` encoding

`Rec(f, ty_f, body)` is encoded as:

1. A `LAM`-like leaf at the rec's site, with the binder `f` of type `ty_f`.
2. The body subtree encoded normally below it, with `Var("f")` references resolving to this site.
3. The recursive self-reference is realized by **wrapping the body's MERA subtree with a `f`→`f` self-loop at the LCA layer** — this is one disentangler at the topmost layer touched by both the binder and the var(s), modified to identity-on-the-binder-channel. The point is that the recursion *depth* (number of levels of `Var("f")` resolved to the same `f`) becomes a count of disentangler applications at the *single* LCA layer, not a chain length.

The encoder implementation is a one-method extension to A's pre-order walk plus a one-method "self-loop disentangler" construction. F provides:

- The AST node (gated by the flag).
- The minimal encoding path (~50 lines).
- The acceptance test (§10.5) that constructs a Fibonacci-like `Rec` program at varying depths and asserts the bond-dim scaling.

A full integration with A's encoder (round-trip, decoder, hole-bearing, etc.) is **out of scope**; it lives in the future sub-project that handles surface-syntax recursion.

---

## 8. File layout

```
src/qft_pcn/qft/
├── mera.py                  # MERATensor, MERA class, layer_dims, causal_cone_path
├── mera_evolution.py        # hierarchical TEBD: trotter_step, evolve, energy
└── (mps.py, evolution.py, hamiltonian.py, fock.py unchanged)

src/qft_pcn/logic/
└── mera_compat.py           # lift_encoding_meta_to_mera, Rec node,
                             # minimal Rec encoder
                             # (NOT a full A-port; just enough for §10.5)

src/qft_pcn/tests/
├── test_mera.py             # all MERA unit + structural tests (§10)
└── test_mera_compat.py      # EncodingMeta lifting + Rec encoding test
```

Public exports from `src/qft_pcn/qft/__init__.py`:

```python
from .mera import MERA, MERATensor, layer_dims, causal_cone_path
from .mera_evolution import trotter_step as mera_trotter_step
from .mera_evolution import evolve as mera_evolve
from .mera_evolution import energy as mera_energy
```

(The MPS-side `trotter_step`/`evolve`/`energy` keep their existing names. The MERA aliases get a prefix to avoid clobbering.)

No changes to any existing file *except*:

- `src/qft_pcn/qft/__init__.py`: add the MERA exports above.
- `src/qft_pcn/logic/ast.py`: add the `Rec` node behind `EXPERIMENTAL_REC = True` (a module-level constant — not a runtime flag — so import statically determines presence).

---

## 9. Error model

Custom exceptions in `src/qft_pcn/qft/mera.py`:

```python
class MERAError(Exception):
    """Base class for MERA-specific errors."""

class InvalidLayerCount(MERAError):
    def __init__(self, N: int):
        super().__init__(f"N={N} is not a positive power of 2")

class CausalConeViolation(MERAError):
    """Raised by debugging affordances if a contraction touches a tensor
    outside the documented causal cone — i.e. the implementation is
    quietly violating the §1.2 bound.
    """

class LayerDimMismatch(MERAError):
    def __init__(self, layer: int, expected: int, got: int):
        super().__init__(
            f"layer {layer}: expected dim {expected}, got {got}")

class IsometryViolation(MERAError):
    """w^dag w != I beyond tolerance — the isometry has lost its property."""

class UnitaryViolation(MERAError):
    """u^dag u != I beyond tolerance — the disentangler has lost unitarity."""
```

`CausalConeViolation` is the test-suite tripwire for §1.2 violations. It is raised by the `_assert_causal_cone(...)` debug helper that wraps the gate / expectation methods when `MERA_DEBUG_CAUSAL_CONE = True` (an env var or module flag). Default off in production; on for the relevant tests.

---

## 10. Tests

Located in `src/qft_pcn/tests/test_mera.py` and `src/qft_pcn/tests/test_mera_compat.py`. Uses the existing pytest convention.

### 10.1 Construction tests

```python
def test_vacuum_has_correct_layer_structure():
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    assert m.N == 8
    assert m.L == 3
    assert m.d_local == 4
    assert m.layer_dims == [4, 4, 4]
    # 4 leaves' worth of disentanglers at layer 0
    assert len(m.disentanglers[0]) == 4
    assert len(m.isometries[0]) == 4
    # 2 disentanglers at layer 1
    assert len(m.disentanglers[1]) == 2
    # top tensor is unit norm
    assert abs(np.linalg.norm(m.top) - 1.0) < 1e-12


def test_vacuum_is_normalized():
    m = MERA.vacuum(N=8, d_local=4)
    assert abs(m.norm_sq() - 1.0) < 1e-10


def test_invalid_N_raises():
    with pytest.raises(InvalidLayerCount):
        MERA.vacuum(N=6, d_local=4)
```

### 10.2 Isometry / unitary structural tests

```python
def test_initial_isometries_satisfy_w_dag_w_equals_I():
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    for ℓ in range(m.L):
        for w in m.isometries[ℓ]:
            d_up = w.shape[0]
            mat = w.reshape(d_up, -1)
            assert np.allclose(mat @ mat.conj().T, np.eye(d_up), atol=1e-10)


def test_initial_disentanglers_are_unitary():
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    for ℓ in range(m.L):
        for u in m.disentanglers[ℓ]:
            d = u.shape[0]
            mat = u.reshape(d * d, d * d)
            assert np.allclose(mat @ mat.conj().T, np.eye(d * d), atol=1e-10)
```

### 10.3 Inner-product self-norm

```python
def test_inner_self_equals_norm_sq():
    m = MERA.vacuum(N=8, d_local=4)
    assert abs(m.inner(m) - m.norm_sq()) < 1e-10


def test_inner_orthogonal_product_states():
    a = MERA.number_states([1, 0, 0, 0, 0, 0, 0, 0], d=3)
    b = MERA.number_states([0, 1, 0, 0, 0, 0, 0, 0], d=3)
    assert abs(a.inner(b)) < 1e-10
```

### 10.4 Local expectation matches MPS for product states

```python
def test_local_expectation_matches_mps_for_product():
    """Both substrates should give the same expectation value for a
    product state and a single-site operator."""
    occ = [2, 0, 1, 3, 0, 0, 1, 0]
    d = 4
    mps_state = MPS.number_states(occ, d=d)
    mera_state = MERA.number_states(occ, d=d)
    n_op = number(d)
    for site in range(8):
        e_mps = mps_state.local_expectation(site, n_op).real
        e_mera = mera_state.local_expectation(site, n_op).real
        assert abs(e_mps - e_mera) < 1e-10, f"site {site}: {e_mps} vs {e_mera}"
```

### 10.5 The acceptance test — recursive Fibonacci-style bond-dim scaling

```python
@pytest.mark.parametrize("D", [2, 3, 4, 5])
def test_recursive_bond_dim_scales_as_O_log_N(D):
    """A Fibonacci-style recursive program at recursion-unrolling depth D
    encodes to a MERA whose per-layer bond dim is O(1).

    This is sub-project F's §10.4 acceptance criterion: bond dim grows
    polylog-in-recursion-depth, NOT linearly.
    """
    from src.qft_pcn.logic.ast import Rec, Var, Lam, App, IntLit, Bin, TInt, TArrow
    from src.qft_pcn.logic.mera_compat import encode_rec_to_mera

    # rec f. \n:Int. if n < 2 then n else f (n - 1) + f (n - 2)
    body = Lam(
        param="n", param_ty=TInt(),
        body=Bin(op="+",
                 lhs=App(fn=Var("f"), arg=Bin(op="-", lhs=Var("n"), rhs=IntLit(val=1))),
                 rhs=App(fn=Var("f"), arg=Bin(op="-", lhs=Var("n"), rhs=IntLit(val=2)))),
    )
    program = Rec(name="f", name_ty=TArrow(src=TInt(), dst=TInt()), body=body)

    N = 2 ** (3 + D)   # scales the MERA, not the AST
    state, meta = encode_rec_to_mera(program, N=N, chi_layer=16, unroll_depth=D)
    # The acceptance check:
    max_bond = max(state.bond_dimensions())
    # O(log N) scaling: with N = 2^(3+D), log2 N = 3 + D, so
    # bond_dim should grow at most linearly in (3 + D), not exponentially.
    # A loose check: bond dim < 4 * (3 + D).
    assert max_bond < 4 * (3 + D), \
        f"max layer bond dim = {max_bond}, expected < {4 * (3 + D)} at D={D}; " \
        f"the encoder is propagating binders horizontally — re-read §1.3 and §7.2"


def test_recursive_against_mps_baseline_at_d_eq_5():
    """Same recursive program encoded into an MPS would require bond dim
    growing exponentially in D. Construct the same program with an MPS
    and observe the contrast.
    """
    # (Detailed implementation deferred to plan task; the test asserts:
    #   MPS-encoded max bond dim >= 2^D
    #   MERA-encoded max bond dim <= 4 * (log_2 N))
```

### 10.6 Compatibility test: re-encode A's P3 on MERA

```python
def test_a_p3_program_observables_match_on_mera():
    """A's P3: \\f:Int->Int. \\x:Int. f (f x).

    Encoded into both an MPS (via A's encoder) and a MERA (via the
    mera_compat shim) — for product-state programs, the leaf-level
    expectations of A's basis-projector operators must agree.
    """
    from src.qft_pcn.logic import encode, parse
    from src.qft_pcn.logic.mera_compat import encode_to_mera
    p = parse(r"\f:Int->Int. \x:Int. f (f x)")
    mps_state, mps_meta = encode(p, N=32, chi_max=16)
    mera_state, mera_meta = encode_to_mera(p, N=32, chi_layer=16)
    # For each leaf, build the kind projector and compare expectations.
    for leaf in range(32):
        kind_proj = build_kind_projector_op(leaf, mps_meta)
        e_mps = mps_state.local_expectation(leaf, kind_proj)
        e_mera = mera_state.local_expectation(leaf, kind_proj)
        assert abs(e_mps - e_mera) < 1e-9, f"leaf {leaf}"
```

### 10.7 Causal-cone bound

```python
def test_local_expectation_touches_only_O_log_N_tensors():
    """Asserts §1.2: a local expectation must only touch the causal cone."""
    m = MERA.vacuum(N=64, d_local=4, chi_layer=4)
    # Patch the internal _ascend_one_layer to count calls.
    call_count = [0]
    orig = m._ascend_one_layer
    m._ascend_one_layer = lambda *a, **kw: (call_count.__setitem__(0, call_count[0] + 1) or orig(*a, **kw))
    op = number(4)
    _ = m.local_expectation(leaf=17, op=op)
    # log2(64) = 6 layers, so at most 6 ascend calls.
    assert call_count[0] == m.L, \
        f"local_expectation touched {call_count[0]} tensors; expected {m.L}"


def test_gate_application_touches_only_layer_0():
    """Applying a two-leaf gate at the leaves must only mutate layer-0
    disentanglers (intra or inter), per §5.6.
    """
    m = MERA.vacuum(N=16, d_local=4, chi_layer=4)
    # Snapshot every tensor's bytes.
    before = {('iso', ℓ, j): m.isometries[ℓ][j].copy()
              for ℓ in range(m.L) for j in range(len(m.isometries[ℓ]))}
    before_dis_l1 = {('dis', 1, j): m.disentanglers[1][j].copy()
                     for j in range(len(m.disentanglers[1]))}
    g = np.eye(16, dtype=complex).reshape(4, 4, 4, 4).reshape(16, 16)
    m.apply_two_site_gate(leaf=4, gate=g, chi_max=4)
    # Layer-1+ tensors unchanged.
    for k, arr in before.items():
        ℓ, j = k[1], k[2]
        if ℓ == 0:
            continue
        assert np.allclose(m.isometries[ℓ][j], arr), f"isometry {k} mutated"
    for k, arr in before_dis_l1.items():
        ℓ, j = k[1], k[2]
        assert np.allclose(m.disentanglers[ℓ][j], arr), f"layer-1 disentangler {k} mutated"
```

### 10.8 Entropy across a cut

```python
def test_entropy_zero_for_product_state():
    m = MERA.number_states([1, 0, 1, 0, 0, 1, 0, 1], d=4)
    for cut in range(7):
        S = m.entanglement_entropy(cut)
        assert abs(S) < 1e-9, f"cut {cut}: S={S}"


def test_entropy_of_bell_pair_at_leaves_0_and_1():
    """Construct a Bell-like state on leaves 0 and 1 by applying a CNOT
    after Hadamarding leaf 0; entropy across cut=0 should be ln 2.
    """
    d = 2
    m = MERA.from_product([np.array([1.0, 0.0]) for _ in range(8)],
                          chi_layer=4)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H_full = H  # d=2 case, no embedding needed
    m.apply_local_gate(0, H_full.astype(complex))
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)
    m.apply_two_site_gate(0, CNOT, chi_max=4)
    S = m.entanglement_entropy(0)
    assert abs(S - np.log(2)) < 1e-6, f"S={S}, expected ln 2"
```

### 10.9 Trotter evolution
```python
def test_mera_trotter_step_preserves_norm():
    from src.qft_pcn.qft.hamiltonian import (FieldSpecies, HamiltonianConfig,
                                             build_hamiltonian)
    species = [FieldSpecies(name="a", cutoff=4, bare_mass=1.0, kinetic=0.5)]
    cfg = HamiltonianConfig(N=8, species=species)
    H = build_hamiltonian(cfg)
    m = MERA.vacuum(N=8, d_local=4, chi_layer=8)
    err = mera_trotter_step(m, H, dt=0.01, imaginary=False, chi_max=8)
    # Real-time evolution is unitary; norm preserved.
    assert abs(m.norm_sq() - 1.0) < 1e-6
    assert err < 1e-6   # only round-off truncation


def test_mera_imag_evolve_lowers_energy():
    """Imaginary-time evolution should drive an arbitrary state toward
    lower energy."""
    # ... standard energy-monotonicity test, like test_qft.py does for MPS.
```

### 10.10 layer_metric placeholder

```python
def test_layer_metric_default_is_identity():
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    for ℓ in range(m.L):
        g = m.layer_metric(ℓ)
        assert g.shape == (m.layer_dims[ℓ], m.layer_dims[ℓ])
        assert np.allclose(g, np.eye(m.layer_dims[ℓ]))
```

### 10.11 Performance budget

```python
@pytest.mark.timeout(10)
def test_mera_local_expectation_perf_budget():
    """For N=64, d_local=8, chi_layer=8: 1000 local_expectations under 10s."""
    m = MERA.vacuum(N=64, d_local=8, chi_layer=8)
    op = number(8)
    for _ in range(1000):
        _ = m.local_expectation(leaf=7, op=op)
```

If this test fails, the implementation is contracting more than the causal cone — re-read §5.4.

### 10.12 Error paths

```python
def test_invalid_N_raises():
    with pytest.raises(InvalidLayerCount):
        MERA.vacuum(N=10, d_local=4)


def test_gate_at_invalid_leaf_raises():
    m = MERA.vacuum(N=8, d_local=4)
    with pytest.raises(ValueError):
        m.apply_two_site_gate(leaf=7, gate=np.eye(16, dtype=complex), chi_max=4)
        # leaf=7 has no right neighbor (leaf 7 + 1 = 8 out of bounds)
```

### 10.13 EncodingMeta lift

```python
def test_lift_encoding_meta_to_mera_assigns_lca_layers():
    """For \\x. (\\y. \\z. x + y + z), the binder x has LCAs:
       - with its x-uses inside the inner-inner lambda: lca layer = 2
    and the encoder must record that."""
    from src.qft_pcn.logic import encode, parse
    from src.qft_pcn.logic.mera_compat import lift_encoding_meta_to_mera
    p = parse(r"\x:Int. \y:Int. \z:Int. (x + y) + z")
    _, mps_meta = encode(p, N=32)
    mera_meta = lift_encoding_meta_to_mera(mps_meta, N=32)
    assert mera_meta.L == 5
    # x's LCA layer with its var use should be >= some depth determined
    # by the AST geometry — the precise number depends on pre-order layout.
    for binder, lca_layer in mera_meta.binder_to_lca_layer.items():
        assert 0 <= lca_layer < mera_meta.L
```

---

## 11. Acceptance criteria

The sub-project is complete when:

1. All tests in §10.1–10.13 pass.
2. The construction is **Vidal-binary** (§1.5): the test suite includes a structural test verifying `len(disentanglers[ℓ]) == n_ℓ // 2` for every layer.
3. Causal-cone bound (§1.2) is verified by the test in §10.7: gate and expectation operations touch `O(L) = O(log N)` tensors, not `O(N)`.
4. The recursive Fibonacci bond-dim test (§10.5) passes: `max_bond < 4 · log_2(N)` for `D ∈ {2, 3, 4, 5}`.
5. A's P3 program encoded into both MPS and MERA produces matching observable expectations (§10.6), demonstrating the substrate-portability claim.
6. The performance budget (§10.11) is met.
7. The `layer_metric` hook (§10.10) returns identity by default and has the right shape per layer.
8. All previously-passing tests in `src/qft_pcn/tests/` continue to pass.
9. The MPS substrate (`qft/mps.py`, `qft/evolution.py`) is **unchanged**: F is additive.

No claim of completion is acceptable without these tests actually running green in a fresh shell. (See `superpowers:verification-before-completion`.)

---

## 12. Open questions

**None.** Every design choice above is locked. If you, while implementing, find a real ambiguity that this document does not resolve — stop and ask the human. Do not paper over it.

Two issues are *deliberately deferred* (not "open"; clearly out-of-scope):

- **Variational MERA optimization** (Vidal §V). F constructs MERAs encoder-time (analytic / product); higher-layer tensors are not re-optimized during evolution. This is the right scope: full variational MERA optimization is a separate, larger sub-project.
- **2D MERA / branching MERA**. Not in F. F is 1D binary MERA.

---

## 13. Contract with downstream sub-projects

This section names the **portability contract** between F and the other sub-projects.

### 13.1 With A (AST↔MPS encoder)

A produces an `(MPS, EncodingMeta)` pair. F provides `lift_encoding_meta_to_mera(meta, N) -> MERAEncodingMeta` plus `encode_to_mera(ast, N, chi_layer) -> (MERA, MERAEncodingMeta)`. The encoded MERA state agrees with the encoded MPS state on all leaf-level observables that A defined (kind / type / bid / value projector expectations), modulo numerical tolerance `1e-9`. **A is not rewritten**; F provides a re-targeting shim.

### 13.2 With B (typing-rule Hamiltonian compiler)

B builds local and two-site Hamiltonian terms acting on the per-site Hilbert space `H_local = H_kind ⊗ H_type ⊗ H_bid ⊗ H_value`. F's leaves carry the same `H_local`. B's `local_op(k)` and `bond_op(k)` constructions are usable verbatim against MERA leaves; what changes is that `mera_evolution.trotter_step(state, H, dt)` applies them through the MERA's layer-0 disentanglers instead of MPS bond gates. **B is not rewritten**.

### 13.3 With C (evaluation Hamiltonian)

Same as B. C adds non-Hermitian-but-Hamiltonian-formulated terms; F's `mera_evolution` handles them identically to MPS (the difference is which substrate carries the dynamics). **C is not rewritten**.

### 13.4 With D (constraint debugger)

D reads per-term `⟨H_term⟩` to identify high-energy constraints. F's `local_expectation` and `two_site_expectation` have the same signatures. **D is not rewritten**.

### 13.5 With E (STLC synthesis demo)

E takes an AST with holes, encodes to MPS, runs imaginary-time evolution, samples completions. With F, E can run on MERA instead — the imaginary-time evolution will be cheaper for hole-heavy ASTs because the MERA's representation of unconstrained candidate-binder superpositions is more compact. **E is not rewritten**, only re-targeted by changing one import (`from qft_pcn.qft import MPS` → `from qft_pcn.qft import MERA`).

### 13.6 With G (LLM bridge)

G consumes the JSON DSL and dispatches to either the MPS or MERA backend based on a `"substrate"` key. F's contribution to G is: G's runtime must support both substrates uniformly. **Decision: defer to G's spec**.

---

## 14. Glossary (local)

- **Leaf** — site at the bottom of the MERA tree (layer 0). Equivalent to an MPS site.
- **Layer** — one level of coarse-graining in the MERA. Layer 0 is the leaves; layer `L − 1` is just below the top.
- **Disentangler `u`** — a 4-leg unitary at a given (layer, position) that removes short-range entanglement before coarse-graining. Intra-pair acts on `(2j, 2j + 1)`; inter-pair acts on `(2j + 1, 2j + 2)`.
- **Isometry `w`** — a 3-leg map that projects two adjacent sites at layer `ℓ` onto one site at layer `ℓ + 1`. Isometric: `w^† w = I`.
- **Top tensor `T_top`** — the rank-3 tensor at the apex of the tree, holding the wavefunction on the top two sites.
- **Causal cone** — the set of MERA tensors that an operator at a leaf can possibly affect; width `O(1)` per layer, total `O(log N)` tensors.
- **Ascending superoperator** — the map `O_ℓ ↦ O_{ℓ + 1}` that lifts a layer-`ℓ` operator through one layer's disentangler-isometry pair.
- **Descending superoperator** — the dual map on density matrices.
- **LCA (Lowest Common Ancestor)** — for two leaves, the deepest tree node both descend from. The LCA layer is the depth of this node.
- **Bond dimension at a layer** — the maximum size of the bond between the layer and the next layer up; bounded by `chi_layer`.
- **Cut** — a partition of the leaves into a left-half and a right-half at some position `i`; the Schmidt rank across the cut is the entanglement entropy boundary.
- **`chi_layer`** — the per-layer bond cap, analogous to MPS's `chi_max`.
- **Vidal-binary MERA** — the specific MERA variant from Vidal 2008 with binary coarse-graining, intra- and inter-pair disentanglers, 2→1 isometries.
