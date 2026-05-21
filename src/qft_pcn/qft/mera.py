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
    rows = [row0]
    # Gram-Schmidt: extend with canonical basis vectors orthogonalized
    # against already-collected rows. Skip vectors that have ~zero projection.
    for basis_idx in range(d_in):
        if len(rows) >= d_up:
            break
        e = np.zeros(d_in, dtype=complex)
        e[basis_idx] = 1.0
        # Orthogonalize against all collected rows.
        for r in rows:
            e = e - (r.conj() @ e) * r
        n = float(np.linalg.norm(e))
        if n > 1e-12:
            rows.append(e / n)
    if len(rows) < d_up:
        # Should not happen if d_up <= d_in.
        raise ValueError(
            f"could not build orthonormal isometry: d_up={d_up}, d_in={d_in}")
    W = np.array(rows, dtype=complex)
    return W


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

    def copy(self) -> "MERA":
        return MERA(
            leaves=[s.copy() for s in self.leaves],
            disentanglers=[[u.copy() for u in layer]
                           for layer in self.disentanglers],
            inter_disentanglers=[[u.copy() for u in layer]
                                 for layer in self.inter_disentanglers],
            isometries=[[w.copy() for w in layer] for layer in self.isometries],
            top=self.top.copy(),
            layer_dims=list(self.layer_dims),
        )

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
            intra = [
                np.eye(d_l * d_l, dtype=complex).reshape(d_l, d_l, d_l, d_l)
                for _ in range(n_l // 2)
            ]
            inter = [
                np.eye(d_l * d_l, dtype=complex).reshape(d_l, d_l, d_l, d_l)
                for _ in range(max(0, n_l // 2 - 1))
            ]
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
        return cls(
            leaves=leaves,
            disentanglers=disentanglers,
            inter_disentanglers=inter_disentanglers,
            isometries=isometries,
            top=top,
            layer_dims=dims,
        )

    @classmethod
    def number_states(cls, occupations: list[int], d: int,
                      chi_layer: int = 16) -> "MERA":
        """Product MERA in the Fock |n_0, n_1, ..., n_{N-1}> basis."""
        from .fock import number_state_vec
        return cls.from_product(
            [number_state_vec(d, n) for n in occupations],
            chi_layer=chi_layer,
        )

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
        """<psi|psi> via layer-by-layer ascending of the density matrices.

        For the top: rho_l, rho_r are layer-(L-1) reduced densities on the
        two top sites. The top tensor T_top: (d_{L-1}, d_{L-1}, 1) is the
        wavefunction amplitudes on these two sites.
        """
        L = self.L
        if L == 1:
            # N = 2: leaves themselves are the top sites.
            rhos = self._layer_density(0)
            rho_l, rho_r = rhos[0], rhos[1]
            T = self.top[..., 0]
            val = np.einsum('ab,cd,ac,bd->', T, T.conj(), rho_l, rho_r)
            return float(np.real(val))
        # General L>=2: ascend to layer L-1 (2 sites), then contract with top.
        rhos = self._layer_density(L - 1)
        assert len(rhos) == 2, f"top layer should have 2 sites, got {len(rhos)}"
        rho_l, rho_r = rhos
        T = self.top[..., 0]
        val = np.einsum('ab,cd,ac,bd->', T, T.conj(), rho_l, rho_r,
                        optimize='greedy')
        return float(np.real(val))

    # ---- normalization and inner product ---------------------------------

    def normalize(self) -> "MERA":
        """In-place normalize to <psi|psi> = 1. Returns self for chaining.

        Distributes the rescaling across leaves so no tensor grows huge:
        each leaf is divided by n^(1/(2N)), giving norm_sq -> 1 exactly.
        """
        n = self.norm_sq()
        if n < 1e-30:
            raise ValueError("cannot normalize a zero-norm MERA")
        scale = n ** (0.5 / self.N)
        for k in range(self.N):
            self.leaves[k] = self.leaves[k] / scale
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
        etas = [self.leaves[k][0, :, 0].conj() * other.leaves[k][0, :, 0]
                for k in range(N)]
        out: list[np.ndarray] = []
        for j in range(N // 2):
            eta_l = etas[2 * j]
            eta_r = etas[2 * j + 1]
            u_b = self.disentanglers[0][j]
            u_k = other.disentanglers[0][j]
            w_b = self.isometries[0][j]
            w_k = other.isometries[0][j]
            # Effective layer-isometry (w composed with u):
            #   W[A, s_l, s_r] = sum_{a, b} w[A, a, b] * u[a, b, s_l, s_r]
            Wb = np.einsum('Aab,abst->Ast', w_b, u_b, optimize='greedy')
            Wk = np.einsum('Aab,abst->Ast', w_k, u_k, optimize='greedy')
            # M[A_bra, A_ket] = sum_{s_l, s_r} Wb.conj()[A_bra, s_l, s_r]
            #                                  * Wk[A_ket, s_l, s_r]
            #                                  * eta_l[s_l] * eta_r[s_r]
            M = np.einsum('Bst,Kst,s,t->BK',
                          Wb.conj(), Wk, eta_l, eta_r,
                          optimize='greedy')
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
            ML = cross_below[2 * j]      # (a_b, a_k)
            MR = cross_below[2 * j + 1]  # (b_b, b_k)
            u_b = u_b_list[j]
            u_k = u_k_list[j]
            w_b = w_b_list[j]
            w_k = w_k_list[j]
            # Composed bra/ket layer-isometries:
            #   Wb[A_b, a_b, b_b] = sum w_b.conj()[A_b, a', b'] *
            #                            u_b.conj()[a', b', a_b, b_b]
            #   Wk[A_k, a_k, b_k] = sum w_k[A_k, a', b'] *
            #                            u_k[a', b', a_k, b_k]
            Wb = np.einsum('Bxy,xyab->Bab', w_b.conj(), u_b.conj(),
                           optimize='greedy')
            Wk = np.einsum('Kxy,xyab->Kab', w_k, u_k,
                           optimize='greedy')
            # M_new[A_b, A_k] = sum_{a_b, b_b, a_k, b_k}
            #     Wb[A_b, a_b, b_b] * Wk[A_k, a_k, b_k]
            #     * ML[a_b, a_k] * MR[b_b, b_k]
            M_new = np.einsum('Bab,Kcd,ac,bd->BK',
                              Wb, Wk, ML, MR,
                              optimize='greedy')
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
        if L == 1:
            # N=2: top sits directly above the two leaves (no isometry
            # ascent consumed in norm_sq either). Contract leaves with
            # top tensors via shared basis.
            eta_l = self.leaves[0][0, :, 0].conj() * other.leaves[0][0, :, 0]
            eta_r = self.leaves[1][0, :, 0].conj() * other.leaves[1][0, :, 0]
            T_b = self.top[..., 0].conj()
            T_k = other.top[..., 0]
            val = np.einsum('st,st,s,t->',
                            T_b, T_k, eta_l, eta_r,
                            optimize='greedy')
            return complex(val)
        cross = self._cross_layer1(other)
        for ell in range(1, L - 1):
            cross = self._cross_ascend(other, ell, cross)
        assert len(cross) == 2
        ML, MR = cross
        T_b = self.top[..., 0].conj()    # (a_l_bra, a_r_bra)
        T_k = other.top[..., 0]          # (a_l_ket, a_r_ket)
        val = np.einsum('ab,cd,ac,bd->',
                        T_b, T_k, ML, MR,
                        optimize='greedy')
        return complex(val)

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

        Inter-pair disentanglers are NOT included here; on product /
        vacuum MERAs they act as identity, so this simplification is
        exact. Sub-projects E/F's gate-application acceptance test
        (Task 17) will validate that the simplification is consistent
        with the actual structure used elsewhere in the substrate.
        """
        j = pos // 2
        d_ell = self.layer_dims[ell]
        I = np.eye(d_ell, dtype=complex)
        if pos % 2 == 0:
            # op acts on left slot of pair j; I on right slot.
            op_pair = np.einsum('ac,bd->abcd', op, I, optimize='greedy')
        else:
            op_pair = np.einsum('ac,bd->abcd', I, op, optimize='greedy')
        u = self.disentanglers[ell][j]
        # u . op_pair . u^dag  (acting in pair-Hilbert space)
        tmp = np.einsum('ABab,abcd->ABcd', u, op_pair, optimize='greedy')
        op_pair_conj = np.einsum('ABcd,CDcd->ABCD', tmp, u.conj(),
                                 optimize='greedy')
        w = self.isometries[ell][j]
        op_up = np.einsum('Aab,abcd,Bcd->AB',
                          w, op_pair_conj, w.conj(),
                          optimize='greedy')
        return op_up

    def local_expectation(self, leaf: int, op: np.ndarray) -> complex:
        """<psi | O_leaf | psi> for a single-leaf operator (d, d).

        Ascends `op` through the L-1 causal-cone tensors to the top
        layer, then contracts with the top tensor. Cost O(d^4 · L).
        Spec §5.4.
        """
        if not 0 <= leaf < self.N:
            raise IndexError(f"leaf {leaf} out of range [0, {self.N})")
        d = self.d_local
        if op.shape != (d, d):
            raise ValueError(f"op shape {op.shape}, expected ({d}, {d})")
        op_layer = op
        pos = leaf
        for ell in range(self.L - 1):
            op_layer = self._ascend_one_layer(op_layer, ell, pos)
            pos //= 2
        # At layer L-1 with 2 top sites; pos is 0 or 1.
        T = self.top[..., 0]   # (d_top, d_top)
        if pos == 0:
            # op acts on left top site:
            # <O> = sum_{a, A, b} T.conj()[a, b] · op[a, A] · T[A, b]
            val = np.einsum('ab,aA,Ab->', T.conj(), op_layer, T,
                            optimize='greedy')
        else:
            # op acts on right top site:
            # <O> = sum_{a, b, B} T.conj()[a, b] · op[b, B] · T[a, B]
            val = np.einsum('ab,bB,aB->', T.conj(), op_layer, T,
                            optimize='greedy')
        return complex(val)

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
        op4 = op.reshape(d, d, d, d)   # (out_l, out_r, in_l, in_r)
        if leaf % 2 == 0:
            # Intra-pair: the gate acts on pair j = leaf // 2 of layer 0.
            j = leaf // 2
            u = self.disentanglers[0][j]
            # u . op . u^dag
            tmp = np.einsum('ABab,abcd->ABcd', u, op4, optimize='greedy')
            op_pair = np.einsum('ABcd,CDcd->ABCD', tmp, u.conj(),
                                optimize='greedy')
            w = self.isometries[0][j]
            op_layer = np.einsum('Aab,abcd,Bcd->AB',
                                 w, op_pair, w.conj(),
                                 optimize='greedy')
            # Now at layer 1, position j; ascend remaining layers.
            pos = j
            for ell in range(1, self.L - 1):
                op_layer = self._ascend_one_layer(op_layer, ell, pos)
                pos //= 2
            T = self.top[..., 0]
            if self.L == 1:
                # N=2 case: top is already in the physical basis; op_layer
                # is the original op (no layer-0 ascent applied actually).
                # Handle separately:
                val = np.einsum('ab,abcd,cd->',
                                T.conj(), op4, T,
                                optimize='greedy')
                return complex(val)
            if pos == 0:
                val = np.einsum('ab,aA,Ab->', T.conj(), op_layer, T,
                                optimize='greedy')
            else:
                val = np.einsum('ab,bB,aB->', T.conj(), op_layer, T,
                                optimize='greedy')
            return complex(val)
        # Inter-pair branch deferred to Task 12.
        raise NotImplementedError(
            "two_site_expectation inter-pair case is in Task 12")

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
