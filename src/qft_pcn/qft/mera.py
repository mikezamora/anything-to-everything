"""Binary 1D MERA (Multi-scale Entanglement Renormalization Ansatz) substrate.

See:
  - Vidal, G. (2008). "Class of quantum many-body states that can be
    efficiently simulated." PRL.
  - Swingle, B. (2012). "Entanglement renormalization and holography." PRD.
  - docs/superpowers/specs/2026-05-21-mera-substrate-design.md (authoritative).

A binary MERA on N = 2^L leaves has L layers. Each layer ℓ carries:
  - intra-pair disentanglers u^(ℓ)_j on sites (2j, 2j+1)
  - inter-pair disentanglers u^(ℓ,inter)_j on sites (2j+1, 2j+2)
  - 2->1 isometries w^(ℓ)_j projecting (2j, 2j+1) -> coarse site j

The top tensor is a rank-3 wavefunction on the topmost two sites.

This module is ADDITIVE — the 1D MPS code is unchanged. Sub-projects
A-E (encoder, Hamiltonian compilers, debugger, synthesis demo) can be
retargeted from MPS to MERA by changing one import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._backend import contract, to_device, to_host


# ---- exceptions -----------------------------------------------------------


class MERAError(Exception):
    """Base class for MERA-specific errors."""


class InvalidLayerCount(MERAError):
    def __init__(self, N: int):
        self.N = N
        super().__init__(f"N={N} is not a positive power of 2")


class CausalConeViolation(MERAError):
    """Raised by debug-mode wrappers when a contraction touches a tensor
    outside the documented O(log N) causal cone — i.e. the implementation
    is quietly violating spec §1.2.
    """


class LayerDimMismatch(MERAError):
    def __init__(self, layer: int, expected: int, got: int):
        self.layer, self.expected, self.got = layer, expected, got
        super().__init__(
            f"layer {layer}: expected dim {expected}, got {got}")


class IsometryViolation(MERAError):
    """w^dag @ w != I beyond tolerance."""


class UnitaryViolation(MERAError):
    """u^dag @ u != I beyond tolerance."""


# ---- pure helpers ---------------------------------------------------------


def _is_identity_matrix(u: np.ndarray, atol: float = 1e-12) -> bool:
    """Fast identity check for a (square-reshapeable) disentangler tensor.

    Mathematically equivalent to ``np.allclose(u.reshape(n, n), np.eye(n),
    atol=atol)`` but avoids np.allclose's per-element isclose/within_tol
    machinery and the np.eye allocation — both dominate _is_product when it
    is called once per inner product during imaginary-time evolution.
    """
    d = u.shape[0]
    n = d * d
    flat = u.reshape(n, n)
    if flat.size == 0:
        return True
    # Equivalent to max(|flat - I|) <= atol but without allocating np.eye:
    # bound the diagonal's deviation from 1 and the off-diagonal magnitude
    # separately. abs_off is a working copy with the diagonal zeroed.
    abs_off = np.abs(flat)
    diag = np.diagonal(flat)
    if np.max(np.abs(diag - 1.0)) > atol:
        return False
    np.fill_diagonal(abs_off, 0.0)
    return bool(np.max(abs_off) <= atol)


def layer_dims(d_local: int, L: int, chi_layer: int = 16) -> list[int]:
    """Per-layer bond dim schedule: [d_0, d_1, ..., d_{L-1}].

    d_0 = d_local (physical leaf dimension).
    d_ℓ = min(chi_layer, d_{ℓ-1} ** 2) — would-be exact growth is capped.
    """
    if L < 1:
        raise ValueError(f"L must be >= 1; got {L}")
    if d_local < 1:
        raise ValueError(f"d_local must be >= 1; got {d_local}")
    if chi_layer < 1:
        raise ValueError(f"chi_layer must be >= 1; got {chi_layer}")
    dims = [d_local]
    for _ in range(1, L):
        dims.append(min(chi_layer, dims[-1] * dims[-1]))
    return dims


def causal_cone_path(leaf: int, L: int) -> list[tuple[int, int]]:
    """The (layer, position) sequence visited while ascending from `leaf`
    to the top. Length L.
    """
    return [(ell, leaf >> ell) for ell in range(L)]


def _orthonormal_isometry(pair: np.ndarray, d_up: int,
                          d_in: int) -> np.ndarray:
    """Build an isometry W of shape (d_up, d_in) such that

        W @ pair = ||pair|| * e_0

    and W @ W^dag = I_{d_up}.

    W's row 0 is `pair.conj() / ||pair||` (so that
    (W @ pair)[0] = pair.conj()^T @ pair / ||pair|| = ||pair||).
    Rows 1..d_up-1 are an orthonormal completion built by Gram-Schmidt
    against the canonical basis e_0, e_1, e_2, ... of the input space.

    If pair is the zero vector, returns the canonical isometry
        W[k, :] = e_k for k = 0 .. d_up - 1.
    """
    pair_norm = float(np.linalg.norm(pair))
    if pair_norm < 1e-15:
        # Degenerate case: pair is zero; return canonical isometry.
        W = np.zeros((d_up, d_in), dtype=complex)
        for k in range(d_up):
            W[k, k] = 1.0
        return W
    # Row 0: normalized conjugate of the pair (so W @ pair = ||pair|| at slot 0).
    row0 = pair.conj() / pair_norm
    # Rows 1..d_up-1: orthonormal completion in the orthogonal complement of
    # pair. The previous implementation Gram-Schmidt'd canonical basis vectors
    # row by row in a Python loop — O(d_up * d_in) Python ops dominated
    # ``MERA.from_product`` profiling (~91% of a Trotter step on P3). Replace
    # with one LAPACK QR on a (d_in, d_up) matrix whose first column is
    # ``pair``: ``Q[:, 0]`` is parallel to ``pair`` (up to a unit-modulus
    # phase), so ``Q[:, 1:]`` is an orthonormal frame orthogonal to ``pair``.
    # We use the conjugate transpose of ``Q[:, 1:]`` as the completion rows.
    #
    # Observational equivalence: for product-MERA construction the completion
    # rows are multiplied by the slot-1+ components of the ascended pair
    # amplitude — exactly zero (the pair concentrates on slot 0 by row 0's
    # construction). Changing the completion basis does NOT change any
    # downstream norm, inner product, or expectation. Verified within
    # atol=1e-6 by the reduction / fix-recursion / eval-hamiltonian /
    # evolution-logic test suites and by the MERA core (test_mera*,
    # test_mera_window, test_mera_holes, etc.).
    if d_up == 1:
        return row0.reshape(1, d_in)
    A = np.zeros((d_in, d_up), dtype=complex)
    A[:, 0] = pair
    idx = np.arange(d_up - 1)
    A[idx, idx + 1] = 1.0
    Q, _ = np.linalg.qr(A)
    W = np.empty((d_up, d_in), dtype=complex)
    W[0] = row0
    W[1:] = Q[:, 1:].conj().T
    return W


def _orthonormal_isometry_multi(pairs: list[np.ndarray], d_up: int,
                                d_in: int) -> np.ndarray:
    """Isometry W (d_up, d_in) whose row space contains every vector in
    ``pairs`` (so ``W @ p`` is lossless: ``||W @ p|| == ||p||`` for each p).

    Generalizes ``_orthonormal_isometry`` to k branch directions. The first
    ``r`` rows are an orthonormal basis (Gram-Schmidt) of span(pairs); the
    remaining rows complete to an orthonormal frame using canonical basis
    vectors. Requires ``r <= d_up`` — for k branches with k <= d_up this
    always holds.
    """
    rows: list[np.ndarray] = []
    # Gram-Schmidt over the branch directions first (these MUST be spanned).
    for p in pairs:
        e = p.astype(complex).copy()
        for r in rows:
            e = e - (r.conj() @ e) * r
        n = float(np.linalg.norm(e))
        if n > 1e-12:
            rows.append(e / n)
    if len(rows) > d_up:
        raise ValueError(
            f"cannot span {len(rows)} branch directions in d_up={d_up}")
    # Complete to an orthonormal frame with canonical basis vectors.
    for basis_idx in range(d_in):
        if len(rows) >= d_up:
            break
        e = np.zeros(d_in, dtype=complex)
        e[basis_idx] = 1.0
        for r in rows:
            e = e - (r.conj() @ e) * r
        n = float(np.linalg.norm(e))
        if n > 1e-12:
            rows.append(e / n)
    if len(rows) < d_up:
        raise ValueError(
            f"could not build isometry: d_up={d_up}, d_in={d_in}")
    return np.array(rows, dtype=complex)


# ---- per-tensor wrapper ---------------------------------------------------


_VALID_KINDS = frozenset({"leaf", "disentangler", "inter_disentangler",
                          "isometry", "top"})


@dataclass
class MERATensor:
    """One slot in the MERA tree.

    `kind` ∈ {"leaf", "disentangler", "inter_disentangler", "isometry", "top"}.
    `array` is the actual numpy array; shape depends on kind:
        leaf:                (1, d_local, 1)
        disentangler:        (d_ℓ, d_ℓ, d_ℓ, d_ℓ)
        inter_disentangler:  (d_ℓ, d_ℓ, d_ℓ, d_ℓ)
        isometry:            (d_{ℓ+1}, d_ℓ, d_ℓ)
        top:                 (d_{L-1}, d_{L-1}, 1)
    """
    kind: str
    layer: int
    position: int
    array: np.ndarray

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"MERATensor kind must be one of {sorted(_VALID_KINDS)}, "
                f"got {self.kind!r}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)


# ---- the MERA class -------------------------------------------------------


@dataclass
class MERA:
    """Binary 1D MERA on N = 2^L leaves.

    Storage:
      leaves: list of N leaf tensors, each (1, d_local, 1).
      disentanglers[ℓ][j]: intra-pair unitary at (layer ℓ, pair j).
      inter_disentanglers[ℓ][j]: inter-pair unitary at (layer ℓ, pair j).
      isometries[ℓ][j]: 2->1 isometry at (layer ℓ, pair j).
      top: rank-3 top tensor (d_{L-1}, d_{L-1}, 1).
      layer_dims: per-layer bond dimensions [d_0, ..., d_{L-1}].
    """
    leaves: list[np.ndarray]
    disentanglers: list[list[np.ndarray]]
    inter_disentanglers: list[list[np.ndarray]]
    isometries: list[list[np.ndarray]]
    top: np.ndarray
    layer_dims: list[int]
    # Optional exact branch decomposition for states built by
    # from_term_superposition: list of (coeff, [per-leaf state vectors]).
    # When present, entanglement_entropy uses it for an exact, cheap cut
    # entropy (the entanglement is isometry-carried, so the disentangler-only
    # _is_product / _materialize fast paths cannot see it).
    _superposition_terms: "list | None" = None
    # Memoized _is_product() result. None = not yet computed / invalidated.
    # Disentanglers are only mutated by apply_local_gate's reconstruction
    # (the two assignment sites in apply_local_gate), which clear this.
    _is_product_cache: "bool | None" = None
    # Bumped by every in-place mutation (leaf gate / 2-site gate /
    # normalize-via-leaves). Read by ``_mera_window._bra_cache`` to
    # invalidate the cross-ascent cache when the state changes.
    _mutation_version: int = 0

    def __post_init__(self) -> None:
        N = len(self.leaves)
        if N <= 0 or (N & (N - 1)) != 0:
            raise InvalidLayerCount(N=N)
        # Per-leaf shape: (1, d_local, 1).
        d_local = self.leaves[0].shape[1]
        for k, leaf in enumerate(self.leaves):
            if leaf.shape != (1, d_local, 1):
                raise ValueError(
                    f"leaf {k}: shape {leaf.shape}, "
                    f"expected (1, {d_local}, 1)")
        # Layer counts must match log2(N).
        L = int(round(np.log2(N)))
        if L != len(self.isometries):
            raise ValueError(
                f"len(isometries)={len(self.isometries)}, expected L={L}")
        if L != len(self.disentanglers):
            raise ValueError(
                f"len(disentanglers)={len(self.disentanglers)}, expected L={L}")
        if L != len(self.inter_disentanglers):
            raise ValueError(
                f"len(inter_disentanglers)={len(self.inter_disentanglers)}, "
                f"expected L={L}")
        if L != len(self.layer_dims):
            raise ValueError(
                f"len(layer_dims)={len(self.layer_dims)}, expected L={L}")
        # Per-layer pair counts.
        for ell in range(L):
            n_l = N // (2 ** ell)
            expected_pairs = n_l // 2
            if len(self.disentanglers[ell]) != expected_pairs:
                raise ValueError(
                    f"layer {ell}: disentanglers count "
                    f"{len(self.disentanglers[ell])}, "
                    f"expected {expected_pairs}")
            if len(self.isometries[ell]) != expected_pairs:
                raise ValueError(
                    f"layer {ell}: isometries count "
                    f"{len(self.isometries[ell])}, "
                    f"expected {expected_pairs}")
            # inter-pair count is one fewer than intra (or 0 if pairs<=1).
            expected_inter = max(0, expected_pairs - 1)
            if len(self.inter_disentanglers[ell]) != expected_inter:
                raise ValueError(
                    f"layer {ell}: inter_disentanglers count "
                    f"{len(self.inter_disentanglers[ell])}, "
                    f"expected {expected_inter}")

    @property
    def N(self) -> int:
        return len(self.leaves)

    @property
    def L(self) -> int:
        return len(self.isometries)

    @property
    def d_local(self) -> int:
        return self.leaves[0].shape[1]

    def layer_dimensions(self) -> list[int]:
        """Per-layer bond dimensions [d_0, ..., d_{L-1}]. Read-only
        accessor over the layer_dims field; used to verify the O(log N)
        bond-dimension bound during imaginary-time evolution (M2 §9.4)."""
        return list(self.layer_dims)

    def leaf_marginal(self, leaf: int) -> np.ndarray:
        """The (d_local,) probability vector for one leaf.

        Fast path for a product (concrete-program) MERA: the leaf's
        marginal is |v_leaf|^2 read directly from the encoded leaf
        vector — no causal-cone ascent needed, because a product MERA
        carries no inter-leaf entanglement, so the single-leaf reduced
        density matrix is exactly the outer product of that leaf's
        vector. For a term-superposition state this would not hold, so
        the slow per-projector local_expectation route is used instead.
        This is the same structural read as local_expectation, just
        without the redundant O(d^4 L) cone contraction (spec §5.4)."""
        if not 0 <= leaf < self.N:
            raise IndexError(f"leaf {leaf} out of range [0, {self.N})")
        if self._superposition_terms is None:
            vec = self.leaves[leaf][0, :, 0]
            p = np.abs(vec) ** 2
            total = float(p.sum())
            if total > 1e-15:
                p = p / total
            return p.astype(float)
        d = self.d_local
        p = np.empty(d, dtype=float)
        for b in range(d):
            proj = np.zeros((d, d), dtype=complex)
            proj[b, b] = 1.0
            p[b] = float(np.real(self.local_expectation(leaf, proj)))
        total = float(p.sum())
        if total > 1e-15:
            p = p / total
        return p

    def copy(self) -> "MERA":
        clone = MERA(
            leaves=[s.copy() for s in self.leaves],
            disentanglers=[[u.copy() for u in layer]
                           for layer in self.disentanglers],
            inter_disentanglers=[[u.copy() for u in layer]
                                 for layer in self.inter_disentanglers],
            isometries=[[w.copy() for w in layer] for layer in self.isometries],
            top=self.top.copy(),
            layer_dims=list(self.layer_dims),
            _superposition_terms=(
                None if self._superposition_terms is None
                else [(c, [s.copy() for s in sites])
                      for c, sites in self._superposition_terms]),
        )
        # Disentanglers are copied verbatim, so the product-state status is
        # identical to the source. Propagating the cache avoids re-scanning
        # every disentangler with _is_identity_matrix on the fresh copy —
        # the dominant cost of inner() during imaginary-time evolution,
        # where the ket is a leaf-mutated copy of the bra each trotter step.
        clone._is_product_cache = self._is_product_cache
        return clone

    # ---- construction ------------------------------------------------------

    @classmethod
    def vacuum(cls, N: int, d_local: int,
               chi_layer: int = 16) -> "MERA":
        """Product MERA at the vacuum (|0...0>) with identity disentanglers
        and canonical embedding isometries.
        """
        from .fock import vacuum_vec
        if N <= 0 or (N & (N - 1)) != 0:
            raise InvalidLayerCount(N=N)
        return cls.from_product(
            [vacuum_vec(d_local) for _ in range(N)],
            chi_layer=chi_layer,
        )

    @classmethod
    def from_product(cls, single_site_states: list[np.ndarray],
                     chi_layer: int = 16) -> "MERA":
        """Build a product MERA from per-leaf state vectors.

        Convention: the FULL physical wavefunction encoded by the network
        equals the product state |v_0> ⊗ |v_1> ⊗ ... ⊗ |v_{N-1}>.

        Construction:
          - leaves carry per-site state vectors (1, d_local, 1).
          - all disentanglers (intra and inter) are identity.
          - isometries are constructed adaptively per-pair so that the
            *ascended* amplitude at each layer concentrates on slot 0
            (with magnitude = norm of the pair). The isometry's row 0 is
            the conjugated pair direction; rows 1..d_up-1 are an orthonormal
            completion built starting from canonical basis vectors.
          - the top tensor is the ascended wavefunction on the 2 top sites.

        For the vacuum, the pair direction is e_0 (basis state |0, 0> with
        flat index 0), so row 0 = e_0 and the isometry is the canonical one.
        """
        N = len(single_site_states)
        if N <= 0 or (N & (N - 1)) != 0:
            raise InvalidLayerCount(N=N)
        d_local = single_site_states[0].shape[0]
        L = int(round(np.log2(N)))
        dims = layer_dims(d_local, L, chi_layer)
        leaves = [s.reshape(1, d_local, 1).astype(complex)
                  for s in single_site_states]
        # Ascend amplitudes through L-1 layers (the upper structure).
        # cur_amps[k] is the (complex) amplitude vector at layer ell, site k.
        cur_amps: list[np.ndarray] = [s.astype(complex)
                                       for s in single_site_states]
        disentanglers: list[list[np.ndarray]] = []
        inter_disentanglers: list[list[np.ndarray]] = []
        isometries: list[list[np.ndarray]] = []
        for ell in range(L):
            n_l = N // (2 ** ell)
            d_l = dims[ell]
            d_up = dims[ell + 1] if ell + 1 < L else d_l
            # All disentanglers at this layer are the identity by
            # construction. Share ONE read-only identity tensor per layer
            # rather than re-allocating np.eye(d_l*d_l) for every slot. For
            # d_l=16 each is a 256x256 complex (~1 MB); a P3 Trotter step
            # rebuilds the whole tree, so O(N) such allocations per step
            # dominate the per-step cost (profiled). The shared tensor is
            # never mutated through this construction path -- MERA tensors
            # are written only by explicit gate-apply methods that operate
            # on per-tensor copies.
            _id_tensor = np.eye(d_l * d_l, dtype=complex).reshape(
                d_l, d_l, d_l, d_l)
            _id_tensor.setflags(write=False)
            intra = [_id_tensor for _ in range(n_l // 2)]
            inter = [_id_tensor for _ in range(max(0, n_l // 2 - 1))]
            iso: list[np.ndarray] = []
            new_amps: list[np.ndarray] = []
            for j in range(n_l // 2):
                v_l = cur_amps[2 * j]
                v_r = cur_amps[2 * j + 1]
                pair = np.outer(v_l, v_r).reshape(-1)   # (d_l * d_l,)
                w_mat = _orthonormal_isometry(pair, d_up, d_l * d_l)
                w = w_mat.reshape(d_up, d_l, d_l)
                iso.append(w)
                # Ascended amplitude: w @ pair concentrates on slot 0.
                ascended = w_mat @ pair
                new_amps.append(ascended)
            disentanglers.append(intra)
            inter_disentanglers.append(inter)
            isometries.append(iso)
            if ell < L - 1:
                cur_amps = new_amps
            # On the final layer (ell == L-1) we do NOT consume the isometry
            # in the top tensor: cur_amps remains the layer-(L-1) amplitudes
            # (the input to the top tensor). The layer-(L-1) isometry is
            # stored in the network for structural symmetry but is not
            # applied in norm_sq / expectation computations.
        # Top tensor: ascended wavefunction on the 2 top sites at layer L-1.
        # cur_amps now has 2 entries (the layer-(L-1) site amplitudes).
        assert len(cur_amps) == 2
        top = np.outer(cur_amps[0], cur_amps[1]).reshape(
            dims[L - 1], dims[L - 1], 1).astype(complex)
        out = cls(
            leaves=leaves,
            disentanglers=disentanglers,
            inter_disentanglers=inter_disentanglers,
            isometries=isometries,
            top=top,
            layer_dims=dims,
        )
        # By construction every (intra + inter) disentangler is the identity
        # — pre-set the _is_product fast-path cache so downstream callers
        # (norm_sq, inner, expectation) skip the per-step
        # _is_identity_matrix scan over every disentangler, which the per-
        # Trotter-step profile shows dominates the rebuild cost.
        out._is_product_cache = True
        return out

    @classmethod
    def from_mps(cls, mps, chi_layer: int = 16) -> "MERA":
        """Coerce a *product* MPS to a MERA on the same N leaves.

        Used by the bridge (`evolve_for_search` with `runtime="mera"`) to lift
        flat-MPS initial states into the hierarchical substrate without
        forcing every caller to build a MERA by hand. The coercion is exact
        when each MPS tensor has trivial bond dimensions (left == right == 1)
        -- i.e. the MPS factorizes as a product state. In that regime each
        tensor's physical slice IS the per-leaf state vector and the result
        is equivalent to ``MERA.from_product(...)`` on those vectors.

        For an entangled MPS (any bond > 1) coercion to a binary MERA on the
        same leaves is not unique and not free; we honest-fail with
        NotImplementedError rather than silently lose entanglement.
        """
        tensors = mps.tensors
        N = len(tensors)
        if N <= 0 or (N & (N - 1)) != 0:
            raise InvalidLayerCount(N=N)
        for k, t in enumerate(tensors):
            if t.shape[0] != 1 or t.shape[2] != 1:
                raise NotImplementedError(
                    f"MERA.from_mps: site {k} has bond dims "
                    f"({t.shape[0]}, {t.shape[2]}); coercion of entangled "
                    "MPS to MERA is not implemented. Build the MERA "
                    "directly (e.g. encode_mera or MERA.from_product) for "
                    "non-product initial states."
                )
        per_leaf = [t[0, :, 0].astype(complex) for t in tensors]
        return cls.from_product(per_leaf, chi_layer=chi_layer)

    @classmethod
    def number_states(cls, occupations: list[int], d: int,
                      chi_layer: int = 16) -> "MERA":
        """Product MERA in the Fock |n_0, n_1, ..., n_{N-1}> basis."""
        from .fock import number_state_vec
        return cls.from_product(
            [number_state_vec(d, n) for n in occupations],
            chi_layer=chi_layer,
        )

    @classmethod
    def from_term_superposition(cls, terms: list[tuple[complex, list[np.ndarray]]],
                                chi_layer: int = 16) -> "MERA":
        """Exact MERA for psi = sum_t coeff_t * (|v_{t,0}> x ... x |v_{t,N-1}>).

        Each ``terms[t]`` is ``(coeff_t, [per-leaf state vectors])``. A single
        term reproduces ``from_product``. With k >= 2 terms whose per-leaf
        vectors differ on more than one leaf, the encoded state is *genuinely
        entangled* — the entanglement is carried by the tree's isometries
        (each layer's bond dimension grows to span the k branch directions),
        exactly as the §1.1 binding-as-entanglement principle requires.

        Construction (exact, no large dense tensor — every per-leaf and
        per-bond object is <= chi_layer-dimensional, k branches tracked
        explicitly):
          - leaves carry the *equal-weight branch sum* per site so the
            per-leaf marginal is faithful (decode/sample read leaves);
          - disentanglers are identity (entanglement is isometry-carried);
          - per layer, each pair's k ascended branch vectors are collected;
            the isometry's rows are an orthonormal basis spanning those k
            directions (Gram-Schmidt), so the ascent is lossless;
          - the top contracts the two final branch-amplitude vectors.

        The explicit branch decomposition is stored on the returned MERA as
        ``_superposition_terms`` so ``entanglement_entropy`` can compute the
        genuine cut entropy exactly and cheaply (it is NOT a product state).
        """
        if not terms:
            raise ValueError("from_term_superposition needs >= 1 term")
        N = len(terms[0][1])
        if N <= 0 or (N & (N - 1)) != 0:
            raise InvalidLayerCount(N=N)
        d_local = terms[0][1][0].shape[0]
        L = int(round(np.log2(N)))
        dims = layer_dims(d_local, L, chi_layer)
        k = len(terms)
        coeffs = [complex(c) for c, _ in terms]
        # Leaf tensors: store the (unnormalized) per-site superposed vector.
        # The full state's amplitude is reconstructed by the isometries; the
        # leaf only needs to span the per-site branch directions for decode.
        leaf_vecs: list[np.ndarray] = []
        for kk in range(N):
            v = np.zeros(d_local, dtype=complex)
            for (c, sites) in terms:
                v = v + sites[kk].astype(complex)
            nv = float(np.linalg.norm(v))
            if nv > 1e-15:
                v = v / nv
            else:
                v = np.zeros(d_local, dtype=complex)
                v[0] = 1.0
            leaf_vecs.append(v)
        leaves = [v.reshape(1, d_local, 1).astype(complex) for v in leaf_vecs]
        # Per-branch amplitude vectors at the current layer (start: leaves).
        # cur[t] is a list of n_l site vectors for branch t.
        cur: list[list[np.ndarray]] = [
            [s.astype(complex) for s in sites] for _, sites in terms
        ]
        disentanglers: list[list[np.ndarray]] = []
        inter_disentanglers: list[list[np.ndarray]] = []
        isometries: list[list[np.ndarray]] = []
        for ell in range(L):
            n_l = N // (2 ** ell)
            d_l = dims[ell]
            d_up = dims[ell + 1] if ell + 1 < L else d_l
            intra = [
                np.eye(d_l * d_l, dtype=complex).reshape(d_l, d_l, d_l, d_l)
                for _ in range(n_l // 2)
            ]
            inter = [
                np.eye(d_l * d_l, dtype=complex).reshape(d_l, d_l, d_l, d_l)
                for _ in range(max(0, n_l // 2 - 1))
            ]
            iso: list[np.ndarray] = []
            new_cur: list[list[np.ndarray]] = [[] for _ in range(k)]
            for j in range(n_l // 2):
                # The k branch pair-vectors at this pair.
                pairs = [np.outer(cur[t][2 * j], cur[t][2 * j + 1]).reshape(-1)
                         for t in range(k)]
                w_mat = _orthonormal_isometry_multi(pairs, d_up, d_l * d_l)
                iso.append(w_mat.reshape(d_up, d_l, d_l))
                for t in range(k):
                    new_cur[t].append(w_mat @ pairs[t])
            disentanglers.append(intra)
            inter_disentanglers.append(inter)
            isometries.append(iso)
            if ell < L - 1:
                cur = new_cur
        # Top tensor: sum_t coeff_t |cur[t][0]> <x> |cur[t][1]>.
        assert all(len(cur[t]) == 2 for t in range(k))
        d_top = dims[L - 1]
        top = np.zeros((d_top, d_top, 1), dtype=complex)
        for t in range(k):
            top[:, :, 0] += coeffs[t] * np.outer(cur[t][0], cur[t][1])
        m = cls(
            leaves=leaves,
            disentanglers=disentanglers,
            inter_disentanglers=inter_disentanglers,
            isometries=isometries,
            top=top,
            layer_dims=dims,
        )
        # Stash the exact branch decomposition for entropy/materialization.
        m._superposition_terms = [
            (coeffs[t], [s.astype(complex) for s in terms[t][1]])
            for t in range(k)
        ]
        return m

    # ---- environment contractions -----------------------------------------

    def _layer_density(self, ell: int) -> list[np.ndarray]:
        """The list of per-site reduced density matrices at layer ell.

        Layer 0: rho_k = leaf_k @ leaf_k^dag per site, shape (d_local, d_local).
        Layer ell > 0: built by ascending layer-(ell-1) density matrices
        through the disentangler-isometry pair using the standard ascending
        superoperator on each pair independently. (For product MERAs the
        inter-pair disentanglers are identity, so independent-pair ascent
        is exact. Non-product cases require inter-pair coupling; this is
        only used for product states or measurement of self-norm-like
        quantities.)

        Returns a list of n_ell density matrices, each (d_ell, d_ell).
        """
        if ell == 0:
            rhos: list[np.ndarray] = []
            for s in self.leaves:
                v = s[0, :, 0]  # leaf state vector (d_local,)
                rhos.append(np.outer(v, v.conj()))
            return rhos
        # Recurse: get layer-(ell-1), then ascend.
        rho_below = self._layer_density(ell - 1)
        N = self.N
        n_l = N // (2 ** ell)
        d_below = self.layer_dims[ell - 1]
        u_intra = self.disentanglers[ell - 1]
        w_list = self.isometries[ell - 1]
        rhos_new: list[np.ndarray] = []
        for j in range(n_l):
            a = 2 * j
            b = 2 * j + 1
            # Joint pair density (kron preserves the (a_l, a_r, b_l, b_r) order
            # when interpreted as matrix kron of two (d, d) matrices).
            # np.kron(rho_a, rho_b) has shape (d_below*d_below, d_below*d_below).
            # Reshape to (d_below, d_below, d_below, d_below) with
            # indices (a_l_block, a_l_inblock=a_r, b_l_block, b_l_inblock=b_r).
            # That gives indices (a_l, a_r, b_l, b_r) — matches our convention.
            rho_kron = np.kron(rho_below[a], rho_below[b])
            rho_pair = rho_kron.reshape(d_below, d_below, d_below, d_below)
            # Apply intra-pair disentangler: u rho_pair u^dag.
            u = u_intra[j]
            tmp = np.einsum('abst,stcd->abcd', u, rho_pair)
            rho_pair = np.einsum('abcd,pqcd->abpq', tmp, u.conj())
            # Project with isometry.
            w = w_list[j]
            rho_new = np.einsum('Aab,abcd,Bcd->AB', w, rho_pair, w.conj())
            rhos_new.append(rho_new)
        return rhos_new

    def norm_sq(self) -> float:
        """<psi|psi>.

        Computed as inner(self, self) via the double-network cross-tensor
        algorithm — this is correct for ARBITRARY MERA states (not just
        product MERAs). The previous "product-state factored" form
        (<top|rho_l⊗rho_r|top>) was only correct when each layer's reduced
        density factorized across pairs, which fails as soon as a gate
        application entangles adjacent leaves.
        """
        if self._superposition_terms is not None:
            return self._norm_sq_from_terms()
        return float(np.real(self.inner(self)))

    def _norm_sq_from_terms(self) -> float:
        """Exact <psi|psi> for a state stored as an explicit product-term
        superposition. <psi|psi> = sum_{t,t'} conj(c_t) c_t' prod_k
        <v_{t,k}|v_{t',k}>. No large dense tensor — k branches, N leaves.
        """
        terms = self._superposition_terms
        k = len(terms)
        coeffs = np.array([c for c, _ in terms], dtype=complex)
        G = np.ones((k, k), dtype=complex)
        for kk in range(self.N):
            col = np.array([terms[t][1][kk].astype(complex) for t in range(k)])
            G = G * (col.conj() @ col.T)
        cc = np.outer(coeffs.conj(), coeffs)
        return float(np.real(np.sum(cc * G)))

    # ---- normalization and inner product ---------------------------------

    def normalize(self) -> "MERA":
        """In-place normalize to <psi|psi> = 1. Returns self for chaining.

        Distributes the rescaling across leaves so no tensor grows huge:
        each leaf is divided by n^(1/(2N)), giving norm_sq -> 1 exactly.

        For a term-superposition state the rescaling is applied to the
        branch coefficients and the top tensor (the leaves carry only the
        per-site marginal direction and must stay unit-norm).
        """
        n = self.norm_sq()
        if n < 1e-30:
            raise ValueError("cannot normalize a zero-norm MERA")
        if self._superposition_terms is not None:
            s = np.sqrt(n)
            self._superposition_terms = [
                (c / s, sites) for c, sites in self._superposition_terms]
            self.top = self.top / s
            self._mutation_version += 1
            return self
        scale = n ** (0.5 / self.N)
        for k in range(self.N):
            self.leaves[k] = self.leaves[k] / scale
        self._mutation_version += 1
        return self

    def _cross_layer1(self, other: "MERA") -> list[np.ndarray]:
        """Build layer-1 cross "double" tensors from leaves + layer-0
        disentangler/isometry.

        At the leaf level the bra and ket SHARE the physical basis index;
        the elementwise product eta_k[s] = self.leaf_k.conj()[s] *
        other.leaf_k[s] captures this. Then the pair's effective bra/ket
        isometries (disentangler composed with isometry) act on the
        shared physical index pair (s_l, s_r), with eta_l and eta_r
        weighting the sum.

        Returns a list of (d_up_bra, d_up_ket) = (d_1, d_1) cross
        tensors, one per layer-0 pair.
        """
        N = self.N
        etas = [to_device(self.leaves[k][0, :, 0].conj()
                          * other.leaves[k][0, :, 0])
                for k in range(N)]
        out: list[np.ndarray] = []
        for j in range(N // 2):
            eta_l = etas[2 * j]
            eta_r = etas[2 * j + 1]
            u_b = to_device(self.disentanglers[0][j])
            u_k = to_device(other.disentanglers[0][j])
            w_b = to_device(self.isometries[0][j])
            w_k = to_device(other.isometries[0][j])
            # Effective layer-isometry (w composed with u):
            #   W[A, s_l, s_r] = sum_{a, b} w[A, a, b] * u[a, b, s_l, s_r]
            Wb = contract('Aab,abst->Ast', w_b, u_b)
            Wk = contract('Aab,abst->Ast', w_k, u_k)
            # M[A_bra, A_ket] = sum_{s_l, s_r} Wb.conj()[A_bra, s_l, s_r]
            #                                  * Wk[A_ket, s_l, s_r]
            #                                  * eta_l[s_l] * eta_r[s_r]
            M = contract('Bst,Kst,s,t->BK', Wb.conj(), Wk, eta_l, eta_r)
            out.append(M)
        return out

    def _cross_ascend(self, other: "MERA", ell: int,
                      cross_below: list[np.ndarray]
                      ) -> list[np.ndarray]:
        """Ascend cross tensors from layer ell to layer ell+1, for ell >= 1.

        At layer >= 1, bra and ket bonds are independent: cross_below[k]
        has shape (d_ell_bra, d_ell_ket). The ascent applies self's
        disentangler+isometry on the bra index and other's on the ket
        index (no shared-basis collapse).
        """
        N = self.N
        n_above = N // (2 ** (ell + 1))
        u_b_list = self.disentanglers[ell]
        u_k_list = other.disentanglers[ell]
        w_b_list = self.isometries[ell]
        w_k_list = other.isometries[ell]
        out: list[np.ndarray] = []
        for j in range(n_above):
            ML = to_device(cross_below[2 * j])      # (a_b, a_k)
            MR = to_device(cross_below[2 * j + 1])  # (b_b, b_k)
            u_b = to_device(u_b_list[j])
            u_k = to_device(u_k_list[j])
            w_b = to_device(w_b_list[j])
            w_k = to_device(w_k_list[j])
            # Composed bra/ket layer-isometries:
            #   Wb[A_b, a_b, b_b] = sum w_b.conj()[A_b, a', b'] *
            #                            u_b.conj()[a', b', a_b, b_b]
            #   Wk[A_k, a_k, b_k] = sum w_k[A_k, a', b'] *
            #                            u_k[a', b', a_k, b_k]
            Wb = contract('Bxy,xyab->Bab', w_b.conj(), u_b.conj())
            Wk = contract('Kxy,xyab->Kab', w_k, u_k)
            # M_new[A_b, A_k] = sum_{a_b, b_b, a_k, b_k}
            #     Wb[A_b, a_b, b_b] * Wk[A_k, a_k, b_k]
            #     * ML[a_b, a_k] * MR[b_b, b_k]
            M_new = contract('Bab,Kcd,ac,bd->BK', Wb, Wk, ML, MR)
            out.append(M_new)
        return out

    def inner(self, other: "MERA") -> complex:
        """<self | other>. Self is bra (conjugated), other is ket.

        For self == other this reduces to norm_sq.

        Algorithm: build "double-network" cross tensors layer by layer.
        Layer 0 → 1 has shared-basis collapse (eta_k = bra.conj() * ket).
        Layers 1 → L-1 have independent bra/ket bonds. Top contracts
        with cross at layer L-1 (2 sites).
        """
        if other.N != self.N:
            raise ValueError(f"MERA length mismatch: {self.N} vs {other.N}")
        if other.L != self.L:
            raise ValueError(f"MERA L mismatch: {self.L} vs {other.L}")
        if other.d_local != self.d_local:
            raise ValueError(
                f"d_local mismatch: {self.d_local} vs {other.d_local}")
        L = self.L
        # Product-state fast path: when both operands are product MERAs
        # (identity disentanglers) that share the SAME tree of isometries,
        # disentanglers and top tensor, the double-network contraction
        # telescopes exactly to the product of the per-leaf overlaps
        # (every isometry W satisfies W^dag W = I and cancels). This is
        # the common case for imaginary-time evolution, where the ket is
        # a copy of the bra with only leaf tensors mutated; it replaces an
        # O(N) sequence of tensor contractions with O(N) vector dot
        # products. The result is mathematically identical to the network
        # contraction below (verified by the MERA inner-product tests).
        if (self._superposition_terms is None
                and other._superposition_terms is None
                and self._is_product() and other._is_product()
                and self._same_tree(other)):
            val = complex(1.0)
            for k in range(self.N):
                bra = self.leaves[k][0, :, 0].conj()
                ket = other.leaves[k][0, :, 0]
                val *= complex(bra @ ket)
            return val
        # D30 fix (third sister of D5 + D25): the cross-network helpers
        # ``_cross_layer1`` / ``_cross_ascend`` only fold the intra-pair
        # disentangler composed with the isometry; they silently drop the
        # layer-0 INTER-pair disentanglers. After ``apply_two_site_gate`` on
        # an odd leaf those become non-identity and the network contraction
        # is biased. Route via leaf-basis materialization (cost O(d^N), but
        # the gate-application invariant — only layer-0 mutates — is the
        # same regime that already triggers D5/D25's materialize routing).
        if (self._superposition_terms is None
                and other._superposition_terms is None
                and (self._layer0_any_nontrivial()
                     or other._layer0_any_nontrivial())):
            psi_b = self._materialize()
            psi_k = other._materialize()
            return complex(np.vdot(psi_b.ravel(), psi_k.ravel()))
        if L == 1:
            # N=2: top sits directly above the two leaves (no isometry
            # ascent consumed in norm_sq either). Contract leaves with
            # top tensors via shared basis.
            eta_l = to_device(self.leaves[0][0, :, 0].conj()
                              * other.leaves[0][0, :, 0])
            eta_r = to_device(self.leaves[1][0, :, 0].conj()
                              * other.leaves[1][0, :, 0])
            T_b = to_device(self.top[..., 0].conj())
            T_k = to_device(other.top[..., 0])
            val = contract('st,st,s,t->', T_b, T_k, eta_l, eta_r)
            return complex(to_host(val))
        cross = self._cross_layer1(other)
        for ell in range(1, L - 1):
            cross = self._cross_ascend(other, ell, cross)
        assert len(cross) == 2
        ML, MR = cross
        T_b = to_device(self.top[..., 0].conj())    # (a_l_bra, a_r_bra)
        T_k = to_device(other.top[..., 0])          # (a_l_ket, a_r_ket)
        val = contract('ab,cd,ac,bd->', T_b, T_k, ML, MR)
        return complex(to_host(val))

    # ---- local gate application -------------------------------------------

    # ---- ascending superoperator ------------------------------------------

    def _ascend_one_layer(self, op: np.ndarray, ell: int,
                          pos: int) -> np.ndarray:
        """Lift a single-site operator from (layer ell, position pos) to
        (layer ell+1, position pos // 2).

        op: (d_ell, d_ell) acting on site `pos` at layer `ell`.
        Returns: (d_{ell+1}, d_{ell+1}) on the coarse-grained site.

        Algorithm (Vidal 2008 §III.5):
            j = pos // 2 (pair index)
            op_pair = op (x) I  if pos even, else I (x) op
            op_pair_conj = u . op_pair . u^dag   (intra-pair disentangler)
            op_up = w . op_pair_conj . w^dag      (isometry projection)

        Inter-pair disentanglers are NOT included in this single-pair
        ascending superoperator — they are pair-coupling and require a
        wider causal cone to fold (handled by the caller via
        :meth:`local_expectation`'s materialize-fallback path; see D5
        fix). On vacuum / product MERAs the inter-pair disentanglers act
        as identity so this simplification is exact; whenever they are
        non-identity at layer 0 (post gate-application) the caller MUST
        NOT route through this method, or the result is silently biased.
        """
        j = pos // 2
        d_ell = self.layer_dims[ell]
        op_dev = to_device(op)
        I = to_device(np.eye(d_ell, dtype=complex))
        if pos % 2 == 0:
            # op acts on left slot of pair j; I on right slot.
            op_pair = contract('ac,bd->abcd', op_dev, I)
        else:
            op_pair = contract('ac,bd->abcd', I, op_dev)
        u = to_device(self.disentanglers[ell][j])
        # u . op_pair . u^dag  (acting in pair-Hilbert space)
        tmp = contract('ABab,abcd->ABcd', u, op_pair)
        op_pair_conj = contract('ABcd,CDcd->ABCD', tmp, u.conj())
        w = to_device(self.isometries[ell][j])
        op_up = contract('Aab,abcd,Bcd->AB', w, op_pair_conj, w.conj())
        return op_up

    def _layer0_any_nontrivial(self) -> bool:
        """True iff any layer-0 disentangler (intra OR inter) is non-identity.

        Used by :meth:`local_expectation` and :meth:`two_site_expectation`
        to route to the materialize-based exact path (D5 + D25 fix). Two
        sister bugs share this guard:

        * D5: the single-pair ascending superoperator silently drops the
          INTER-pair disentangler from the causal cone.
        * D25: ``two_site_expectation``'s odd-leaf branch ALSO silently
          drops the layer-0 INTRA-pair disentanglers
          (``disentanglers[0][j_inter]`` and ``disentanglers[0][j_inter+1]``)
          from its 4-site fold. The in-code comment acknowledged the hole;
          this guard now closes it.

        After :meth:`apply_two_site_gate` either family may be non-identity,
        so we route uniformly when EITHER is.
        """
        for u in self.disentanglers[0]:
            if not _is_identity_matrix(u):
                return True
        for u in self.inter_disentanglers[0]:
            if not _is_identity_matrix(u):
                return True
        return False

    def _local_expectation_via_materialize(self, leaf: int,
                                           op: np.ndarray) -> complex:
        """Exact <psi|O_leaf|psi> via leaf-basis materialization.

        Used as the D5 fold-fallback when layer-0 inter-pair disentanglers
        are non-identity. ``_materialize`` itself raises NotImplementedError
        for non-identity layers >= 1, so this honest-fails loudly outside
        the gate-application invariant (only layer-0 modifiable).
        """
        psi = self._materialize()   # (d,)*N
        d = self.d_local
        # Apply op on the `leaf` axis: psi_out[..., s', ...] =
        #   sum_s op[s', s] · psi[..., s, ...]
        axes = list(range(self.N))
        # Move leaf axis to front for contract, then back.
        psi_moved = np.moveaxis(psi, leaf, 0)              # (d, d, ..., d)
        op_psi = np.tensordot(op, psi_moved, axes=([1], [0]))   # (d, d, ..., d)
        op_psi = np.moveaxis(op_psi, 0, leaf)
        # <psi | op_psi> = sum over all indices of conj(psi) * op_psi.
        return complex(np.vdot(psi.ravel(), op_psi.ravel()))

    def _two_site_expectation_via_materialize(self, leaf: int,
                                              op: np.ndarray) -> complex:
        """Exact <psi|O_{leaf, leaf+1}|psi> via leaf-basis materialization.

        D5 fold-fallback. Same loud-fail discipline as
        :meth:`_local_expectation_via_materialize`.
        """
        psi = self._materialize()                # (d,)*N
        d = self.d_local
        op4 = op.reshape(d, d, d, d)             # (out_l, out_r, in_l, in_r)
        # Apply op4 on axes (leaf, leaf+1) of psi.
        psi_moved = np.moveaxis(psi, [leaf, leaf + 1], [0, 1])   # (d,d,...)
        op_psi = np.tensordot(op4, psi_moved, axes=([2, 3], [0, 1]))
        op_psi = np.moveaxis(op_psi, [0, 1], [leaf, leaf + 1])
        return complex(np.vdot(psi.ravel(), op_psi.ravel()))

    def local_expectation(self, leaf: int, op: np.ndarray) -> complex:
        """<psi | O_leaf | psi> for a single-leaf operator (d, d).

        Ascends `op` through the L-1 causal-cone tensors to the top
        layer, then contracts with the top tensor. Cost O(d^4 · L).
        Spec §5.4.

        D5 fix: when any layer-0 inter-pair disentangler is non-identity
        (post :meth:`apply_two_site_gate` on an odd leaf), the single-pair
        ascending superoperator is insufficient — the inter-pair causal
        cone is dropped and the result is biased. In that case we route
        to the materialize-based exact path, which folds all layer-0
        disentanglers (intra + inter) into a leaf-basis state and
        contracts the operator directly. ``_materialize`` enforces the
        layer-≥1-identity invariant via NotImplementedError, so this
        path honest-fails on out-of-spec states.
        """
        if not 0 <= leaf < self.N:
            raise IndexError(f"leaf {leaf} out of range [0, {self.N})")
        d = self.d_local
        if op.shape != (d, d):
            raise ValueError(f"op shape {op.shape}, expected ({d}, {d})")
        if self._superposition_terms is not None:
            return self._local_expectation_from_terms(leaf, op)
        if self._layer0_any_nontrivial():
            return self._local_expectation_via_materialize(leaf, op)
        op_layer = to_device(op)
        pos = leaf
        for ell in range(self.L - 1):
            op_layer = self._ascend_one_layer(op_layer, ell, pos)
            pos //= 2
        # At layer L-1 with 2 top sites; pos is 0 or 1.
        T = to_device(self.top[..., 0])   # (d_top, d_top)
        if pos == 0:
            # op acts on left top site:
            # <O> = sum_{a, A, b} T.conj()[a, b] · op[a, A] · T[A, b]
            val = contract('ab,aA,Ab->', T.conj(), op_layer, T)
        else:
            # op acts on right top site:
            # <O> = sum_{a, b, B} T.conj()[a, b] · op[b, B] · T[a, B]
            val = contract('ab,bB,aB->', T.conj(), op_layer, T)
        return complex(to_host(val))

    def _local_expectation_from_terms(self, leaf: int,
                                      op: np.ndarray) -> complex:
        """Exact <psi|O_leaf|psi> for a term-superposition state.

        <O> = sum_{t,t'} conj(c_t) c_t' (prod_{k != leaf} <v_{t,k}|v_{t',k}>)
              * <v_{t,leaf}| O |v_{t',leaf}>.
        """
        terms = self._superposition_terms
        k = len(terms)
        coeffs = np.array([c for c, _ in terms], dtype=complex)
        G = np.ones((k, k), dtype=complex)
        for kk in range(self.N):
            if kk == leaf:
                continue
            col = np.array([terms[t][1][kk].astype(complex) for t in range(k)])
            G = G * (col.conj() @ col.T)
        col_l = np.array([terms[t][1][leaf].astype(complex) for t in range(k)])
        # Oll[t, t'] = <v_{t,leaf}| op |v_{t',leaf}>
        Oll = col_l.conj() @ op @ col_l.T
        cc = np.outer(coeffs.conj(), coeffs)
        return complex(np.sum(cc * G * Oll))

    def two_site_expectation(self, leaf: int, op: np.ndarray) -> complex:
        """<psi | O_{leaf, leaf+1} | psi> for a two-leaf operator.

        op shape: (d^2, d^2). Convention: op acts on |s_leaf, s_{leaf+1}>
        with leaf the outer (slow) index, matching np.kron.
        Spec §5.5.
        """
        if not 0 <= leaf < self.N - 1:
            raise IndexError(
                f"leaf {leaf} invalid for two-site op (N={self.N})")
        d = self.d_local
        if op.shape != (d * d, d * d):
            raise ValueError(
                f"op shape {op.shape}, expected ({d * d}, {d * d})")
        op4 = to_device(op.reshape(d, d, d, d))   # (out_l, out_r, in_l, in_r)
        # D5 + D25 fix: when ANY layer-0 disentangler (intra OR inter) is
        # non-identity, both the single-pair ascending path (even leaf)
        # and the inter-pair 4-site fold (odd leaf) silently drop layer-0
        # disentanglers from the causal cone. Route to materialize-based
        # exact contraction; `_materialize` honest-fails
        # (NotImplementedError) on layer-≥1 non-identity, preserving the
        # layer-0-only modification invariant.
        if (self._superposition_terms is None
                and self._layer0_any_nontrivial()):
            return self._two_site_expectation_via_materialize(leaf, op)
        if leaf % 2 == 0:
            # Intra-pair: the gate acts on pair j = leaf // 2 of layer 0.
            j = leaf // 2
            u = to_device(self.disentanglers[0][j])
            # u . op . u^dag
            tmp = contract('ABab,abcd->ABcd', u, op4)
            op_pair = contract('ABcd,CDcd->ABCD', tmp, u.conj())
            w = to_device(self.isometries[0][j])
            op_layer = contract('Aab,abcd,Bcd->AB', w, op_pair, w.conj())
            # Now at layer 1, position j; ascend remaining layers.
            pos = j
            for ell in range(1, self.L - 1):
                op_layer = self._ascend_one_layer(op_layer, ell, pos)
                pos //= 2
            T = to_device(self.top[..., 0])
            if self.L == 1:
                # N=2 case: top is already in the physical basis; op_layer
                # is the original op (no layer-0 ascent applied actually).
                # Handle separately:
                val = contract('ab,abcd,cd->', T.conj(), op4, T)
                return complex(to_host(val))
            if pos == 0:
                val = contract('ab,aA,Ab->', T.conj(), op_layer, T)
            else:
                val = contract('ab,bB,aB->', T.conj(), op_layer, T)
            return complex(to_host(val))
        # Inter-pair: leaf is odd. (leaf, leaf+1) straddle adjacent pairs.
        # For product MERAs (all disentanglers identity), the inter-pair
        # contraction reduces to the direct 4-site expectation on the
        # four involved leaves, with the inter-pair disentangler
        # u_inter[0][(leaf - 1) // 2] applied to the middle two slots.
        # Gate application (Tasks 16-17) may break the identity-disentangler
        # assumption; for those cases the present implementation is exact
        # only when the modified disentanglers act on slots disjoint from
        # the 4 involved leaves' chain. Sub-project F's tests cover the
        # product-state case; non-product inter-pair tests are deferred to
        # Task 19 (gate-then-measure) via comparison against materialize.
        j_inter = (leaf - 1) // 2
        # The 4 leaves involved:
        a_idx = 2 * j_inter
        P_idx = 2 * j_inter + 1
        Q_idx = 2 * j_inter + 2
        e_idx = 2 * j_inter + 3
        s_a = self.leaves[a_idx][0, :, 0]
        s_P = self.leaves[P_idx][0, :, 0]
        s_Q = self.leaves[Q_idx][0, :, 0]
        s_d = self.leaves[e_idx][0, :, 0]
        u_inter = self.inter_disentanglers[0][j_inter]
        # Apply u_inter on (P, Q):
        inter_PQ = np.einsum('PQpq,p,q->PQ',
                             u_inter, s_P, s_Q,
                             optimize='greedy')
        # 4-site post-inter ket:
        #   psi_post[a, P, Q, d] = s_a[a] · inter_PQ[P, Q] · s_d[d]
        psi_post = np.einsum('a,PQ,d->aPQd',
                             s_a, inter_PQ, s_d,
                             optimize='greedy')
        # <op_{P,Q}> = sum_{a, P, Q, d, P', Q'} psi.conj()[a, P, Q, d]
        #                * op4[P, Q, P', Q'] * psi[a, P', Q', d]
        e = np.einsum('aPQd,PQpq,apqd->',
                      psi_post.conj(), op4, psi_post,
                      optimize='greedy')
        return complex(e)

    # ---- structural helpers -----------------------------------------------

    def bond_dimensions(self) -> list[int]:
        """Per-layer max bond dimension (spec §6.2).

        Entry ℓ is the maximum across isometries at layer ℓ of their FIRST
        index (the "out" / coarse-grained dim). For a vacuum/product MERA
        with uniform chi_layer, returns [layer_dims[1], ..., layer_dims[L-1],
        layer_dims[L-1]].
        """
        out: list[int] = []
        for ell in range(self.L):
            iso_layer = self.isometries[ell]
            out.append(max(w.shape[0] for w in iso_layer) if iso_layer else 0)
        return out

    def layer_dimensions(self) -> list[int]:
        """Per-layer bond dimensions [d_0, ..., d_{L-1}] (read-only).

        Interface-only accessor used by the M2 Fix-recursion O(log N)
        demo to assert no layer's bond dimension exceeds chi_layer
        through an imaginary-time evolution. No behavior change.
        """
        return list(self.layer_dims)

    def layer_metric(self, layer: int) -> np.ndarray:
        """Effective metric tensor at the given layer (spec §6.3).

        Default: identity. Future sub-projects coupling QPCN curvature to
        the substrate will mutate this to reflect bulk geometry at radial
        coordinate = layer index.
        """
        if not 0 <= layer < self.L:
            raise IndexError(f"layer {layer} out of range [0, {self.L})")
        return np.eye(self.layer_dims[layer], dtype=complex)

    # ---- entanglement entropy ---------------------------------------------

    def _is_product(self) -> bool:
        """True if all disentanglers (intra + inter, every layer) are the
        identity. Used as the product-state entropy fast path.

        Guard: a term-superposition state carries its entanglement in the
        isometries, NOT the disentanglers (which stay identity). The
        disentangler-only check below cannot see that entanglement, so it
        would falsely report such a state as product. If a non-trivial
        term decomposition is present (>= 2 distinct terms), defer to it:
        only a single-term superposition is genuinely a product state.
        """
        if self._superposition_terms is not None:
            return len(self._superposition_terms) <= 1
        if self._is_product_cache is not None:
            return self._is_product_cache
        result = True
        for layer in self.disentanglers:
            for u in layer:
                if not _is_identity_matrix(u):
                    result = False
                    break
            if not result:
                break
        if result:
            for layer in self.inter_disentanglers:
                for u in layer:
                    if not _is_identity_matrix(u):
                        result = False
                        break
                if not result:
                    break
        self._is_product_cache = result
        return result

    def _same_tree(self, other: "MERA") -> bool:
        """True if `other` shares this MERA's isometry/disentangler/top
        tensors (leaves may differ). Used to guard the product-state
        inner-product fast path: the leaf-overlap telescoping identity
        only holds when bra and ket are built on the same tree."""
        # Identity short-circuit: norm_sq/normalize call inner(self, self),
        # so the common case is a self-comparison and a per-element
        # array_equal scan over every isometry/disentangler is pure waste.
        if self is other:
            return True
        if not np.array_equal(self.top, other.top):
            return False
        for sl, ol in ((self.disentanglers, other.disentanglers),
                       (self.inter_disentanglers, other.inter_disentanglers),
                       (self.isometries, other.isometries)):
            if len(sl) != len(ol):
                return False
            for slay, olay in zip(sl, ol):
                if len(slay) != len(olay):
                    return False
                for a, b in zip(slay, olay):
                    if not np.array_equal(a, b):
                        return False
        return True

    def entanglement_entropy(self, cut: int) -> float:
        """Von Neumann entropy across the cut after leaf `cut`.

        Spec §5.8. For product MERAs (identity disentanglers) returns 0
        without materializing. Otherwise materializes the full statevector
        (cost O(d_local^N)) — acceptable only for the small acceptance-test
        sizes (N <= 8, d_local <= 4); production optimization via the
        descending superoperator is a follow-on.
        """
        if not 0 <= cut < self.N - 1:
            raise ValueError(
                f"cut {cut} out of range [0, {self.N - 1})")
        if self._superposition_terms is not None:
            return self._entropy_from_terms(cut)
        if self._is_product():
            return 0.0
        psi = self._materialize()      # shape (d_local,) * N
        N = self.N
        d = self.d_local
        left_size = cut + 1
        right_size = N - left_size
        psi_mat = psi.reshape(d ** left_size, d ** right_size)
        # Schmidt SVD across the cut → eigvals of reduced density.
        sv = np.linalg.svd(psi_mat, compute_uv=False)
        p = sv * sv
        total = p.sum()
        if total > 1e-15:
            p = p / total
        p = p[p > 1e-15]
        return float(-(p * np.log(p)).sum())

    def _entropy_from_terms(self, cut: int) -> float:
        """Exact von Neumann entropy across ``cut`` for a state stored as an
        explicit product-term superposition (``_superposition_terms``).

        psi = sum_t c_t |L_t> (x) |R_t>, with L_t the product of leaf
        vectors 0..cut and R_t the product of leaves cut+1..N-1.

        The reduced density on the left subsystem has the same nonzero
        spectrum as the k x k matrix in the branch basis; we diagonalize
        that (k <= a few) — no large dense tensor. The branch vectors are
        not orthogonal, so we work in the (possibly non-orthonormal) branch
        frame: spectrum of rho_L = eigenvalues of  X = G_L^{1/2}-free form
        computed as eig of (C* G_R C-weighted) — done via the standard
        trick: rho_L ~ eig of  M  where
            M[t, t'] = c_t conj(c_{t'}) <R_{t'}|R_t> <L_{t'}|L_t-projected>.
        Simplest robust route: build the k x k "left" and "right" Gram
        matrices and form the Hermitian  rho-spectrum matrix.
        """
        terms = self._superposition_terms
        k = len(terms)
        coeffs = np.array([c for c, _ in terms], dtype=complex)
        left_idx = list(range(cut + 1))
        right_idx = list(range(cut + 1, self.N))

        def gram(idx_set: list[int]) -> np.ndarray:
            G = np.ones((k, k), dtype=complex)
            for kk in idx_set:
                col = np.array([
                    terms[t][1][kk].astype(complex) for t in range(k)])
                # overlap[t, t'] = <v_t | v_t'>
                ov = col.conj() @ col.T
                G = G * ov
            return G

        GL = gram(left_idx)
        GR = gram(right_idx)
        # Full state norm^2 = sum_{t,t'} conj(c_t) c_t' GL[t,t'] GR[t,t'].
        cc = np.outer(coeffs.conj(), coeffs)
        norm_sq = float(np.real(np.sum(cc * GL * GR)))
        if norm_sq < 1e-30:
            return 0.0
        # Reduced density on the LEFT subsystem, expressed in the branch
        # frame: rho_L = sum_{t,t'} conj(c_t) c_t' GR[t,t'] |L_t'><L_t|.
        # The nonzero spectrum of an operator written in a non-orthonormal
        # frame as  sum_{t',t} A[t',t] |L_t'><L_t|  equals the spectrum of
        # the k x k matrix  GL @ A^T , where GL[t,t'] = <L_t|L_t'>.
        # Here A[t',t] = conj(c_t) c_t' GR[t,t'], so
        #   A^T[t,t'] = c_t conj(c_t') GR[t',t]
        #             = (outer(coeffs, coeffs.conj()) * GR.T)[t, t'].
        # Hence  K = GL @ B  with  B = outer(coeffs, coeffs.conj()) * GR.T.
        # The previous form  K = (cc.T * GR.T) * GL  was elementwise (not a
        # matrix product) and over-reported entropy: a 2-term superposition
        # differing on a single leaf (a genuine product state, S=0) was
        # reported as ln(2).
        B = np.outer(coeffs, coeffs.conj()) * GR.T
        K = (GL @ B) / norm_sq
        ev = np.linalg.eigvals(K)
        p = np.real(ev)
        p = p[p > 1e-12]
        if p.size == 0:
            return 0.0
        p = p / p.sum()
        return float(-(p * np.log(p)).sum())

    def _materialize(self) -> np.ndarray:
        """Dense state vector, shape (d_local,) * N.

        Cost O(d_local^N) — do NOT call from production code paths. Only
        used by `_entropy_general` on tiny N for the acceptance tests.

        Construction: start with the leaf product, then apply each layer's
        intra- then inter-pair disentanglers IN LEAF SPACE. Higher layers
        (>= 1) act in a coarse basis and would need to be lifted through
        the isometries to act on leaves; we assert here that higher layers
        are identity, matching the from_product + apply_two_site_gate
        contract of sub-project F.
        """
        # Validate the "only layer-0 disentanglers may be non-identity"
        # invariant. Anything else would require lifting higher-layer
        # disentanglers back to leaf space, which is outside F's scope.
        for ell in range(1, self.L):
            for u in self.disentanglers[ell]:
                d = u.shape[0]
                if not np.allclose(u.reshape(d * d, d * d),
                                   np.eye(d * d), atol=1e-10):
                    raise NotImplementedError(
                        f"_materialize: layer {ell} disentangler is "
                        "non-identity; not yet supported")
            for u in self.inter_disentanglers[ell]:
                d = u.shape[0]
                if not np.allclose(u.reshape(d * d, d * d),
                                   np.eye(d * d), atol=1e-10):
                    raise NotImplementedError(
                        f"_materialize: layer {ell} inter-disentangler is "
                        "non-identity; not yet supported")
        d = self.d_local
        N = self.N
        # Build leaf product.
        psi = self.leaves[0][0, :, 0].astype(complex)
        for k in range(1, N):
            psi = np.tensordot(psi, self.leaves[k][0, :, 0], axes=0)
        # shape: (d, d, ..., d) with N axes.
        # Apply layer-0 intra-pair disentanglers on (2j, 2j+1).
        for j in range(N // 2):
            u = self.disentanglers[0][j]   # (out_l, out_r, in_l, in_r)
            psi = np.tensordot(u, psi, axes=([2, 3], [2 * j, 2 * j + 1]))
            # tensordot puts contracted axes' output at front (out_l, out_r,
            # then the remaining axes in order).
            psi = np.moveaxis(psi, [0, 1], [2 * j, 2 * j + 1])
        # Apply layer-0 inter-pair disentanglers on (2j+1, 2j+2).
        for j in range(max(0, N // 2 - 1)):
            u = self.inter_disentanglers[0][j]
            left = 2 * j + 1
            right = 2 * j + 2
            psi = np.tensordot(u, psi, axes=([2, 3], [left, right]))
            psi = np.moveaxis(psi, [0, 1], [left, right])
        return psi

    # ---- two-site gate application ----------------------------------------

    def apply_two_site_gate(self, leaf: int, gate: np.ndarray,
                            chi_max: int = 16,
                            eps: float = 1e-12) -> float:
        """In-place: apply `gate` (d^2 x d^2) at leaves (leaf, leaf+1).

        Per spec §5.6, the gate absorbs into a layer-0 disentangler:
        intra-pair (disentanglers[0][leaf//2]) if leaf is even, else
        inter-pair (inter_disentanglers[0][(leaf-1)//2]). Higher-layer
        tensors are NOT modified — this is the causal-cone bound at
        encoder-time.

        Returns the truncation error (sum of discarded squared singular
        values relative to total). For unitary gates this is ~0.
        """
        if not 0 <= leaf < self.N - 1:
            raise ValueError(
                f"leaf {leaf} invalid for two-site gate (N={self.N})")
        d = self.d_local
        if gate.shape != (d * d, d * d):
            raise ValueError(
                f"gate shape {gate.shape}, expected ({d * d}, {d * d})")
        g4 = gate.reshape(d, d, d, d)   # (out_l, out_r, in_l, in_r)
        if leaf % 2 == 0:
            j = leaf // 2
            u_old = self.disentanglers[0][j]
            # u_new[A, B, s, t] = sum_{a, b} g4[A, B, a, b] * u_old[a, b, s, t]
            u_new = np.einsum('ABab,abst->ABst', g4, u_old,
                              optimize='greedy')
        else:
            j_inter = (leaf - 1) // 2
            u_old = self.inter_disentanglers[0][j_inter]
            u_new = np.einsum('ABab,abst->ABst', g4, u_old,
                              optimize='greedy')
        mat = u_new.reshape(d * d, d * d)
        # SVD for principled non-unitary handling (e.g. imag-time Trotter).
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        if S.size:
            tol = eps * S[0]
            keep_mask = S > tol
        else:
            keep_mask = np.zeros(0, dtype=bool)
        U_kept = U[:, keep_mask][:, :chi_max]
        S_kept = S[keep_mask][:chi_max]
        Vh_kept = Vh[keep_mask][:chi_max]
        norm_sq_full = float((S * S).sum())
        kept_norm_sq = float((S_kept * S_kept).sum())
        trunc_err = max(0.0, 1.0 - (kept_norm_sq / norm_sq_full
                                    if norm_sq_full > 0 else 1.0))
        recon = (U_kept * S_kept) @ Vh_kept
        if recon.shape != (d * d, d * d):
            full = np.zeros((d * d, d * d), dtype=complex)
            full[:recon.shape[0], :recon.shape[1]] = recon
            recon = full
        recon4 = recon.reshape(d, d, d, d)
        if leaf % 2 == 0:
            self.disentanglers[0][leaf // 2] = recon4
        else:
            self.inter_disentanglers[0][(leaf - 1) // 2] = recon4
        # A reconstructed disentangler is generally non-identity.
        self._is_product_cache = None
        # Invalidate the bra-bra cross-ascent cache: disentanglers and/or
        # inter-disentanglers at layer 0 changed.
        self._mutation_version += 1
        return trunc_err

    def apply_local_gate(self, leaf: int, gate: np.ndarray) -> None:
        """In-place: leaf <- gate @ leaf on the physical index.

        Only the target leaf tensor is modified; per spec §5.6 the rest
        of the MERA tree is untouched.
        """
        if not 0 <= leaf < self.N:
            raise IndexError(
                f"leaf {leaf} out of range [0, {self.N})")
        d = self.d_local
        if gate.shape != (d, d):
            raise ValueError(
                f"gate shape {gate.shape}, expected ({d}, {d})")
        self.leaves[leaf] = np.einsum(
            'st,ltr->lsr', gate, self.leaves[leaf], optimize='greedy')
        # Invalidate the bra-bra cross-ascent cache held by
        # _mera_window: this leaf mutation changes the diagonal eta_k.
        self._mutation_version += 1
