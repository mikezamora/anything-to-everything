"""Lemma library: storage, indexing, registration. Spec §3, §4.

A lemma is a solved sub-problem -- a MERA ground state with residual
energy below eps_register. By Curry-Howard it is a proof object: it
inhabits the proposition its Hamiltonian encodes. The library caches it
so a future QPCN run clamps it rather than re-deriving it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta

FINGERPRINT_DIM = 32


@dataclass(frozen=True)
class MeraTensorBundle:
    """A serialization-friendly snapshot of a MERA's tensors. NOT a live
    MERA; LemmaLibrary.materialize rebuilds a MERA from it."""
    n_leaves: int
    leaf_dim: int
    n_layers: int
    leaf_vectors: list[np.ndarray]
    disentanglers: list[np.ndarray]
    isometries: list[np.ndarray]


@dataclass(frozen=True)
class DerivationMetadata:
    """Provenance of a lemma (spec §3.2)."""
    hamiltonian_id: str
    residual_energy: float
    energy_gap: float
    trotter_steps: int
    assumptions: tuple[str, ...]
    lemma_deps: tuple[str, ...]
    conditional: bool
    source_run_id: str


@dataclass(frozen=True)
class Lemma:
    """A cached proof object (spec §3.2)."""
    lemma_id: str
    proposition_type: str
    mera_tensors: MeraTensorBundle
    encoding_meta: MeraEncodingMeta
    derivation: DerivationMetadata
    fingerprint: np.ndarray
