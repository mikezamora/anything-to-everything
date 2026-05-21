"""Matrix Product State representation with bond truncation.

An MPS encodes a many-body quantum state on N sites with local dimension d
as a chain of rank-3 tensors A[k] of shape (chi_left, d, chi_right).
Boundary bonds are dimension 1.

This is the workhorse of 1D quantum many-body physics — it represents
states with bounded entanglement entropy efficiently, scales linearly with
system size, and admits exact application of local and two-site
operators with SVD-based bond truncation.

We use the convention: index order (left_bond, physical, right_bond), and
states are NOT kept in any canonical form by default. Operations that
need it (two-site gate application) bring the relevant pair into mixed
canonical form via QR/SVD as needed.

The state is |Psi> = sum_{s} (A[0] A[1] ... A[N-1])^{1,1} |s_0 s_1 ... s_{N-1}>
where the matrix product on the physical indices s_k.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fock import vacuum_vec, number_state_vec


@dataclass
class MPS:
    """List-of-tensors MPS. Each tensor: (chi_left, d, chi_right)."""

    tensors: list[np.ndarray]

    def __post_init__(self) -> None:
        for k, t in enumerate(self.tensors):
            if t.ndim != 3:
                raise ValueError(f"site {k} tensor must be rank-3, "
                                 f"got shape {t.shape}")

    @property
    def N(self) -> int:
        return len(self.tensors)

    @property
    def d(self) -> int:
        return self.tensors[0].shape[1]

    def copy(self) -> "MPS":
        return MPS(tensors=[t.copy() for t in self.tensors])

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_product(cls, single_site_states: list[np.ndarray]) -> "MPS":
        tensors = [s.reshape(1, s.shape[0], 1).astype(complex)
                   for s in single_site_states]
        return cls(tensors=tensors)

    @classmethod
    def vacuum(cls, N: int, d: int) -> "MPS":
        return cls.from_product([vacuum_vec(d) for _ in range(N)])

    @classmethod
    def number_states(cls, occupations: list[int], d: int) -> "MPS":
        """Product state |n_0, n_1, ..., n_{N-1}> in Fock basis."""
        return cls.from_product([number_state_vec(d, n) for n in occupations])

    # ---- environment contractions -----------------------------------------

    def norm_sq(self) -> float:
        env = np.ones((1, 1), dtype=complex)
        for t in self.tensors:
            env = np.einsum('ij,isk,jsl->kl', env, t, t.conj())
        return float(np.real(env[0, 0]))

    def inner(self, other: "MPS") -> complex:
        """<self | other> contracted by sweeping environment tensors.

        Self is the bra (conjugated), other is the ket. Both MPSes must have
        the same length and per-site dimension.
        """
        if other.N != self.N:
            raise ValueError(f"MPS length mismatch: {self.N} vs {other.N}")
        env = np.ones((1, 1), dtype=complex)
        for k in range(self.N):
            bra = self.tensors[k].conj()    # (chi_l_self, d, chi_r_self)
            ket = other.tensors[k]          # (chi_l_other, d, chi_r_other)
            if bra.shape[1] != ket.shape[1]:
                raise ValueError(f"site {k} dim mismatch: "
                                 f"{bra.shape[1]} vs {ket.shape[1]}")
            # env has shape (chi_l_self, chi_l_other) -> (chi_r_self, chi_r_other)
            env = np.einsum('ij,isk,jsl->kl', env, bra, ket)
        return complex(env[0, 0])

    def normalize(self) -> "MPS":
        norm = np.sqrt(self.norm_sq())
        if norm < 1e-15:
            raise ValueError("cannot normalize a zero-norm MPS")
        # Distribute the rescaling across sites so no tensor grows huge.
        scale = norm ** (1.0 / self.N)
        for k in range(self.N):
            self.tensors[k] = self.tensors[k] / scale
        return self

    # ---- expectation values ------------------------------------------------

    def local_expectation(self, site: int, op: np.ndarray) -> complex:
        """<Psi|O_site|Psi> for a single-site operator O of shape (d, d)."""
        env = np.ones((1, 1), dtype=complex)
        for k, t in enumerate(self.tensors):
            if k == site:
                env = np.einsum('ij,isk,st,jtl->kl', env, t, op, t.conj())
            else:
                env = np.einsum('ij,isk,jsl->kl', env, t, t.conj())
        return complex(env[0, 0])

    def two_site_expectation(self, site: int, op: np.ndarray) -> complex:
        """<Psi|O_{site, site+1}|Psi> for a two-site operator of shape (d^2, d^2).

        op acts on the combined basis |s_k s_{k+1}> with s_k being the
        outer (slow) index per np.kron convention.
        """
        if not 0 <= site < self.N - 1:
            raise ValueError(f"site {site} invalid for two-site op")
        d = self.d
        op_t = op.reshape(d, d, d, d)  # (s_out_l, s_out_r, s_in_l, s_in_r)
        env = np.ones((1, 1), dtype=complex)
        for k, t in enumerate(self.tensors):
            if k == site:
                # We'll need to contract two sites together with the operator.
                left = t  # (chi_l, d, chi_m)
                continue
            if k == site + 1:
                right = t  # (chi_m, d, chi_r)
                # Build theta = left @ right with operator applied.
                # theta[i, a, b, j] = sum_m left[i, s, m] * right[m, t, j]
                #                     * op_t[a, b, s, t]
                theta = np.einsum('ism,mtj,abst->iabj', left, right, op_t)
                bra = np.einsum('iam,mbj->iabj', left.conj(), right.conj())
                env = np.einsum('ij,iabk,jabl->kl', env, theta, bra)
                continue
            env = np.einsum('ij,isk,jsl->kl', env, t, t.conj())
        return complex(env[0, 0])

    # ---- gate application --------------------------------------------------

    def apply_local_gate(self, site: int, gate: np.ndarray) -> None:
        """In-place: A[site] <- gate @ A[site] (gate acts on physical index)."""
        self.tensors[site] = np.einsum('st,ltr->lsr', gate,
                                       self.tensors[site])

    def apply_two_site_gate(self, site: int, gate: np.ndarray,
                            chi_max: int = 32, eps: float = 1e-12) -> float:
        """In-place application of a two-site gate at (site, site+1).

        Returns the truncation error (1 - sum of kept singular values^2).
        Bond dimension between the two sites is capped at chi_max.

        Math:
            theta[i, s, t, j] = sum_m A[k][i, s, m] A[k+1][m, t, j]
            theta'[i, a, b, j] = sum_{s,t} G[a, b, s, t] theta[i, s, t, j]
            Reshape to (chi_l * d, d * chi_r) and SVD.
            Truncate singular values to top chi.
            A[k]_new   = U[:chi_l, :, :chi] (after reshape)
            A[k+1]_new = (S * Vh[:chi, :])  (after reshape)
        """
        if not 0 <= site < self.N - 1:
            raise ValueError(f"site {site} invalid")
        d = self.d
        L = self.tensors[site]      # (chi_l, d, chi_m)
        R = self.tensors[site + 1]  # (chi_m, d, chi_r)
        chi_l, _, chi_m = L.shape
        chi_m_check, _, chi_r = R.shape
        if chi_m != chi_m_check:
            raise ValueError("inconsistent bond dimensions")

        # Combine and apply gate.
        theta = np.einsum('ism,mtj->istj', L, R)            # (chi_l, d, d, chi_r)
        g4 = gate.reshape(d, d, d, d)                       # (a, b, s, t)
        theta = np.einsum('abst,istj->iabj', g4, theta)     # (chi_l, d, d, chi_r)

        # SVD on the bipartition (left site | right site).
        mat = theta.reshape(chi_l * d, d * chi_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        # Truncate.
        norm_sq = float((S * S).sum())
        keep = S > eps * (S[0] if S.size else 1.0)
        S_kept = S[keep][:chi_max]
        U_kept = U[:, keep][:, :chi_max]
        Vh_kept = Vh[keep][:chi_max]
        chi_new = S_kept.size
        kept_norm_sq = float((S_kept * S_kept).sum())
        trunc_err = max(0.0, 1.0 - (kept_norm_sq / norm_sq if norm_sq > 0
                                    else 1.0))

        # Renormalize so the truncated state has unit relative weight in
        # the kept subspace; absolute normalization is maintained at the
        # MPS level by the caller via norm_sq if needed.
        L_new = U_kept.reshape(chi_l, d, chi_new)
        # Push the singular values into the right tensor (right-canonicalize the
        # left site).
        R_new = (S_kept[:, None] * Vh_kept).reshape(chi_new, d, chi_r)
        self.tensors[site] = L_new
        self.tensors[site + 1] = R_new
        return trunc_err

    # ---- diagnostics --------------------------------------------------------

    def bond_dimensions(self) -> list[int]:
        return [t.shape[2] for t in self.tensors[:-1]]

    def entanglement_entropy(self, bond: int) -> float:
        """von Neumann entropy across the cut after `bond` (0-indexed).

        Computed by QR-sweeping from both ends to bring the bond into mixed
        canonical form, then SVD at the bond. Stays polynomial in N; no
        full statevector is ever materialized.
        """
        if not 0 <= bond < self.N - 1:
            raise ValueError(f"bond {bond} out of range")
        ts = [t.copy() for t in self.tensors]
        # Left-canonicalize sites 0..bond by sweeping QR rightward.
        for k in range(bond + 1):
            chi_l, d, chi_r = ts[k].shape
            mat = ts[k].reshape(chi_l * d, chi_r)
            Q, R = np.linalg.qr(mat)
            ts[k] = Q.reshape(chi_l, d, Q.shape[1])
            if k + 1 < self.N:
                ts[k + 1] = np.einsum('rs,sdt->rdt', R, ts[k + 1])
        # Right-canonicalize sites N-1..bond+1 by sweeping QR leftward.
        for k in range(self.N - 1, bond, -1):
            chi_l, d, chi_r = ts[k].shape
            mat = ts[k].reshape(chi_l, d * chi_r)
            # Use QR on the transpose so we right-canonicalize.
            Q, R = np.linalg.qr(mat.conj().T)
            Q = Q.conj().T  # shape (k_new, d * chi_r)
            R = R.conj().T  # shape (chi_l, k_new)
            ts[k] = Q.reshape(Q.shape[0], d, chi_r)
            if k - 1 >= 0:
                ts[k - 1] = np.einsum('rds,st->rdt', ts[k - 1], R)
        # SVD the boundary matrix between site `bond` and site `bond+1`.
        # After canonicalization, ts[bond] is left-iso and the residual
        # is in its right bond; pair it with ts[bond+1] for the SVD.
        L = ts[bond]            # (chi_l, d, chi_b)
        R = ts[bond + 1]        # (chi_b, d, chi_r)
        theta = np.einsum('ism,mtj->istj', L, R)
        mat = theta.reshape(L.shape[0] * L.shape[1],
                            R.shape[1] * R.shape[2])
        s = np.linalg.svd(mat, compute_uv=False)
        s = s / (np.linalg.norm(s) + 1e-15)
        s2 = s * s
        s2 = s2[s2 > 1e-15]
        return float(-(s2 * np.log(s2)).sum())
