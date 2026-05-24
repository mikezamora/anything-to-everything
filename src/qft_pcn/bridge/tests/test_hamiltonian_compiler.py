"""§2.4 four new constraint-kind compilers — Hermiticity + ground-state checks."""
import numpy as np

from src.qft_pcn.bridge.runtime.hamiltonian_compiler import compile_vocabulary


def test_vocabulary_hermitian_psd():
    # 4 sites, node_kind cutoff 8, allowed primitives = first 3 labels
    H = compile_vocabulary(
        primitives=["a", "b", "c"],
        vocab=["a", "b", "c", "d", "e", "f", "g", "h"],
        n_sites=4, kind_cutoff=8, weight=1.0,
    )
    assert H.shape == (8**4, 8**4)
    assert np.allclose(H, H.conj().T), "vocabulary H must be Hermitian"
    eigs = np.linalg.eigvalsh(H)
    assert eigs[0] >= -1e-10, "PSD: ground eigenvalue must be ≥ 0"
    assert eigs[0] < 1e-10, "must have a zero ground state (some valid configurations exist)"


def test_vocabulary_disallowed_label_costs_one_weight():
    # 2 sites, cutoff 4, only label 0 allowed → all-disallowed state has energy 2*weight
    H = compile_vocabulary(
        primitives=["a"], vocab=["a", "b", "c", "d"],
        n_sites=2, kind_cutoff=4, weight=3.0,
    )
    # |3,3⟩ basis vector (both sites disallowed) has energy 2*3.0 = 6.0
    dim = 4 ** 2
    idx_3_3 = 3 * 4 + 3
    state = np.zeros(dim); state[idx_3_3] = 1.0
    e = float(state @ H @ state)
    assert abs(e - 6.0) < 1e-9, f"|3,3⟩ should cost 2 violations × weight 3.0 = 6.0, got {e}"
