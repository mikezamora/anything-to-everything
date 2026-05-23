"""§12.11 modular Hamiltonian + entanglement-spectrum classifier tests.

Per §12.11 the entanglement spectrum is an entanglement-equivalence
class read off the bond Schmidt coefficients — never an AST hash. The
three acceptance tests below verify:

  1. alpha-equivalent proofs have IDENTICAL spectra (the binding-as-
     entanglement principle from §1.1: renaming a bound variable does not
     touch the bond entanglement, so the spectrum is invariant);
  2. distinct theorems have DIFFERENT spectra (the spectrum is a true
     fingerprint, not a trivial constant);
  3. the modular Hamiltonian K = -log rho_A is self-adjoint (it is the
     logarithm of a positive Hermitian operator).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.entanglement_spectrum import (
    compute_entanglement_spectrum,
    compute_modular_hamiltonian,
    classify_proof_by_spectrum,
    ProofClassification,
    ProofTopology,
)
from src.qft_pcn.logic.ast import (
    parse, Lam, Var, HoleVar, TInt, TArrow, Bin, IntLit, If, BoolLit, TBool,
)
from src.qft_pcn.logic.mera_encoder import encode_mera


def _encode(src):
    """Encode either a surface string or a pre-built AST node to MERA."""
    ast = parse(src) if isinstance(src, str) else src
    state, _ = encode_mera(ast)
    return state


# ---- structurally entangled fixtures --------------------------------------

def _identity_hole():
    """\\x:Int. ? where ? has candidate {x} — a rank-1 binder hole."""
    return Lam(param="x", param_ty=TInt(),
               body=HoleVar(candidates=["x"]))


def _two_binder_hole():
    """\\f:Int->Int. \\x:Int. ? where ? has candidates {f, x} — rank-2."""
    return Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=["f", "x"])),
    )


# ---- §12.11 acceptance tests ----------------------------------------------


def test_spectrum_invariant_under_alpha_equivalence():
    """Alpha-equivalent proofs encode to the same MERA bond entanglement,
    so their entanglement spectra (and modular Hamiltonians) must be
    identical to floating-point precision. This is §1.1 in action: a
    bound-variable rename is a no-op on the bond structure."""
    s1 = _encode(r"\x:Int. x")
    s2 = _encode(r"\y:Int. y")        # alpha-equivalent to s1
    cut = (s1.N // 2) - 1

    p1 = compute_entanglement_spectrum(s1, cut)
    p2 = compute_entanglement_spectrum(s2, cut)

    assert p1.shape == p2.shape
    assert np.allclose(p1, p2, atol=1e-10)

    K1 = compute_modular_hamiltonian(s1, cut)["K_eigenvalues"]
    K2 = compute_modular_hamiltonian(s2, cut)["K_eigenvalues"]
    assert np.allclose(K1, K2, atol=1e-10)

    c1 = classify_proof_by_spectrum(s1)
    c2 = classify_proof_by_spectrum(s2)
    assert c1.signature() == c2.signature()


def test_spectrum_differs_for_distinct_theorems():
    """Two structurally distinct programs produce distinguishable spectra.
    They need not occupy different topology classes (both may be TRIVIAL
    on a product MERA), but the numeric fingerprint — entropy + spectrum
    tuple — must separate them, otherwise the classifier is useless as a
    fingerprint."""
    # Two hole-bearing sketches with distinct candidate cardinalities
    # produce distinct bond entanglement on the MERA (rank-1 vs rank-2
    # branch superposition). Their spectra must therefore differ.
    s1 = _encode(_identity_hole())
    s2 = _encode(_two_binder_hole())

    c1 = classify_proof_by_spectrum(s1)
    c2 = classify_proof_by_spectrum(s2)

    # Distinct programs must yield distinguishable signatures (the
    # entanglement equivalence class differs). We check the *signature*
    # tuple rather than raw arrays to mirror what the classifier actually
    # exposes as the equivalence-class identity.
    assert c1.signature() != c2.signature(), (
        f"distinct theorems collapsed to the same entanglement "
        f"equivalence class:\n  s1={c1.signature()}\n  s2={c2.signature()}"
    )

    # Sanity: at least one of the two must have nontrivial entanglement
    # (rank > 1), otherwise we would not be testing the spectrum itself.
    p1 = compute_entanglement_spectrum(s1, (s1.N // 2) - 1)
    p2 = compute_entanglement_spectrum(s2, (s2.N // 2) - 1)
    assert max(p1.size, p2.size) >= 2


def test_modular_h_is_self_adjoint():
    """K = -log rho_A must satisfy K^dagger = K. We verify this on a
    program whose MERA has nontrivial bond entanglement (a hole-bearing
    or multi-leaf program), so the spectrum is genuinely multi-rank and
    the test is non-degenerate."""
    s = _encode(_two_binder_hole())
    cut = (s.N // 2) - 1
    out = compute_modular_hamiltonian(s, cut)

    K = out["K_matrix"]
    assert K.ndim == 2 and K.shape[0] == K.shape[1]
    # Self-adjoint: K† == K (within fp tolerance).
    assert np.allclose(K, K.conj().T, atol=1e-12)

    # And rho_A is positive Hermitian with unit trace.
    rho = out["rho_A"]
    assert np.allclose(rho, rho.conj().T, atol=1e-12)
    eigs_rho = np.linalg.eigvalsh(rho)
    assert np.all(eigs_rho >= -1e-12)
    assert abs(np.trace(rho) - 1.0) < 1e-10

    # Modular eigenvalues are real and finite (no -log of a zero
    # eigenvalue, because the spectrum is truncated at _SPEC_TOL).
    K_eigs = out["K_eigenvalues"]
    assert np.all(np.isfinite(K_eigs))
    assert np.all(np.imag(K_eigs) == 0) or np.allclose(np.imag(K_eigs), 0)


# ---- sanity / API tests ----------------------------------------------------


def test_classification_is_a_dataclass_with_signature():
    s = _encode(r"\x:Int. x")
    c = classify_proof_by_spectrum(s)
    assert isinstance(c, ProofClassification)
    assert isinstance(c.topology, ProofTopology)
    assert c.entropy >= 0.0
    sig = c.signature()
    assert isinstance(sig, tuple)
    # Frozen dataclass → hashable, suitable for clustering proofs.
    assert hash(c) is not None


def test_trivial_proof_classifies_as_trivial():
    """A bare identity is a product MERA: rank-1 spectrum → TRIVIAL."""
    s = _encode(r"\x:Int. x")
    c = classify_proof_by_spectrum(s)
    assert c.topology == ProofTopology.TRIVIAL
    assert c.entropy < 1e-9


def test_spectrum_is_normalized_and_descending():
    s = _encode(_two_binder_hole())
    cut = (s.N // 2) - 1
    p = compute_entanglement_spectrum(s, cut)
    assert abs(p.sum() - 1.0) < 1e-10
    assert np.all(np.diff(p) <= 1e-12)        # descending
    assert np.all(p > 0)


def test_spectrum_rejects_out_of_range_cut():
    s = _encode(r"\x:Int. x")
    with pytest.raises(ValueError):
        compute_entanglement_spectrum(s, -1)
    with pytest.raises(ValueError):
        compute_entanglement_spectrum(s, s.N)
