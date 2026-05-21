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
