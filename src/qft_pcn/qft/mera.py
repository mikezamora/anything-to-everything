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
