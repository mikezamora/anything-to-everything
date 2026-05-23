"""Promotion: the use_lemma DSL constraint compiler. Spec §5.

Promotion is operator-algebraic, NOT syntactic (spec §1.5). A use_lemma
constraint compiles to a tensor-network clamp of a cached lemma's MERA
state -- either an init clamp of the host MERA's lemma window or a
``-W |Psi_L><Psi_L|`` projector term. It never decodes the lemma to an
AST and re-derives the region (that would be a classical rewrite,
forbidden by §1.6).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, mera_from_bundle)
from src.qft_pcn.composition.errors import (
    LemmaLeafCountMismatch, LemmaSpeciesMismatch, ConditionalLemmaRefused)
from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta
from src.qft_pcn.qft.mera import MERA

LEMMA_PROJECTOR_WEIGHT = 1e3


@dataclass(frozen=True)
class PromotedLemma:
    """A compiled use_lemma constraint, ready to imprint onto a host
    MERA. The compilation result is purely referential: it names the
    lemma to clamp, the host leaf indices the clamp covers, the clamp
    mode (init clamp vs. projector energy term), and the projector
    weight. No tensors are materialized at compile time."""
    lemma_id: str
    host_leaves: tuple[int, ...]
    mode: str            # "init_clamp" | "projector"
    weight: float


class Promoter:
    """Compiles ``{"kind": "use_lemma", ...}`` constraints into
    operator-algebraic clamps (spec §5).

    The Promoter is the single seam between the DSL and the physics
    substrate: it loads the cached lemma's MERA state and either
       (a) writes the lemma's leaf tensors into the host MERA's lemma
           window so subsequent relaxation never deforms them
           (``mode="init_clamp"``, spec §5.2a), or
       (b) emits a ``-W |Psi_L><Psi_L|`` projector term on the host's
           causal cone over the window (``mode="projector"``,
           spec §5.2b).
    The Promoter NEVER decodes the lemma to an AST and re-derives the
    region.
    """

    def __init__(self, library: LemmaLibrary, mode: str = "init_clamp"):
        if mode not in ("init_clamp", "projector"):
            raise ValueError(f"unknown promotion mode: {mode}")
        # NOTE: full projector-mode machinery (Hamiltonian term emission,
        # evolution under -W|Psi_L><Psi_L|) is deferred. The energy-eval
        # probe `projector_energy` IS implemented (spec §5.2b's overlap
        # functional, used for acceptance §8.7). `apply_init_clamp` itself
        # guards against being driven from a projector-mode Promoter.
        self.library = library
        self.mode = mode

    def compile_constraint(self, constraint: dict) -> PromotedLemma:
        """Compile a ``use_lemma`` DSL constraint into a ``PromotedLemma``.

        Validates referential consistency only -- the lemma exists, its
        leaf count matches the host window, and (unless explicitly opted
        in) the lemma is not conditional. Tensor content is not touched
        until ``apply_init_clamp`` / ``projector_energy``.
        """
        assert constraint.get("kind") == "use_lemma"
        lemma_id = constraint["lemma_id"]
        leaves = tuple(constraint["leaves"])
        weight = float(constraint.get("weight", LEMMA_PROJECTOR_WEIGHT))
        allow_conditional = bool(constraint.get("allow_conditional", False))

        lemma = self.library.load(lemma_id)  # raises LemmaNotFound

        if lemma.derivation.conditional and not allow_conditional:
            raise ConditionalLemmaRefused(
                f"{lemma_id} is conditional (assumptions="
                f"{lemma.derivation.assumptions}); pass "
                f"allow_conditional=True to opt in")

        n_leaves_L = lemma.encoding_meta.n_leaves
        if len(leaves) != n_leaves_L:
            raise LemmaLeafCountMismatch(
                f"lemma {lemma_id} occupies {n_leaves_L} leaves, "
                f"constraint named {len(leaves)}")

        return PromotedLemma(lemma_id=lemma_id, host_leaves=leaves,
                             mode=self.mode, weight=weight)

    def _check_species(self, lemma_meta: MeraEncodingMeta,
                       host_meta: MeraEncodingMeta,
                       host_leaves: tuple[int, ...]) -> None:
        """Verify the host window's species pattern matches the lemma's.

        A mismatch would mean the clamp is being applied to leaves with
        the wrong physical role (e.g. binder vs. use vs. type vs. bid),
        which is meaningless even though the tensor shapes line up.
        """
        lemma_species = lemma_meta.species_of_leaf
        host_species = host_meta.species_of_leaf
        for j, hl in enumerate(host_leaves):
            if j >= len(lemma_species) or hl >= len(host_species):
                raise LemmaSpeciesMismatch(
                    f"leaf index out of range: lemma j={j} "
                    f"(len={len(lemma_species)}), host hl={hl} "
                    f"(len={len(host_species)})")
            if lemma_species[j] != host_species[hl]:
                raise LemmaSpeciesMismatch(
                    f"lemma leaf {j} species {lemma_species[j]!r} != "
                    f"host leaf {hl} species {host_species[hl]!r}")

    def apply_init_clamp(self, host: MERA, host_meta: MeraEncodingMeta,
                         promoted: PromotedLemma) -> set[int]:
        """Write the lemma's leaf tensors into the host's lemma window;
        return the set of host leaf indices that the caller must freeze
        during subsequent relaxation (spec §5.2a, acceptance §8.6).

        This is the operator-algebraic clamp: a direct tensor copy from
        the cached MERA's leaves to the host MERA's leaves. The cached
        MERA is never decoded to an AST, the lemma's Hamiltonian is
        never re-run, and the host MERA's AST is never rewritten -- the
        clamp is a referential imprint of a previously-solved state
        into the host's tensor network. See spec §1.5/§1.6.
        """
        if self.mode != "init_clamp":
            raise NotImplementedError(
                "apply_init_clamp requires mode='init_clamp'; "
                f"this Promoter is in mode={self.mode!r}.")
        lemma = self.library.load(promoted.lemma_id)
        self._check_species(lemma.encoding_meta, host_meta,
                            promoted.host_leaves)
        cached = mera_from_bundle(lemma.mera_tensors)
        frozen: set[int] = set()
        for j, hl in enumerate(promoted.host_leaves):
            host.leaves[hl] = np.asarray(cached.leaves[j]).copy()
            frozen.add(hl)
        return frozen

    def projector_energy(self, host: MERA, host_meta: MeraEncodingMeta,
                         promoted: PromotedLemma) -> float:
        """Compute the projector-mode energy contribution of `promoted`
        on `host` (spec §5.2b). The projector term is
        ``H_L = -W |Psi_L><Psi_L|`` and its expectation on `host` is
        ``<host| H_L |host> = -W * |<Psi_L|host>|^2``.

        This is an operator-algebraic probe: it computes a real overlap
        between two MERAs, never decoding the lemma to an AST. Used by
        the §8.7 acceptance test (init_clamp vs. projector convergence)
        and as the energy contribution emitted into the composed
        Hamiltonian at solve time.
        """
        lemma = self.library.load(promoted.lemma_id)
        self._check_species(lemma.encoding_meta, host_meta,
                            promoted.host_leaves)
        cached = mera_from_bundle(lemma.mera_tensors)
        overlap = host.inner(cached)
        return float(-promoted.weight * abs(overlap) ** 2)

    def composition_residual(self, host: MERA,
                             host_meta: MeraEncodingMeta,
                             promoted_list: list[PromotedLemma],
                             hamiltonian) -> float:
        """Theorem 13.3 / acceptance §8.9: for disjoint lemma windows
        that have already been init-clamped onto the host, the per-leaf
        L2 distance between the host's leaf and the cached lemma's leaf
        must be zero — the composed state is the exact tensor product
        of the cached lemmas on those windows.

        Returns the sum over all (lemma, host_leaf) pairs of
        ``||host.leaves[hl][0,:,0] - cached.leaves[j][0,:,0]||_2``. A
        nonzero residual means a clamp was overwritten or the windows
        overlap.
        """
        total = 0.0
        for promoted in promoted_list:
            lemma = self.library.load(promoted.lemma_id)
            self._check_species(lemma.encoding_meta, host_meta,
                                promoted.host_leaves)
            cached = mera_from_bundle(lemma.mera_tensors)
            for j, hl in enumerate(promoted.host_leaves):
                h_vec = np.asarray(host.leaves[hl])[0, :, 0]
                c_vec = np.asarray(cached.leaves[j])[0, :, 0]
                total += float(np.linalg.norm(h_vec - c_vec))
        return float(total)
