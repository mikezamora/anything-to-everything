"""Tests for the multi-field network with learnable coupling.

These verify the *structural* claims: a coupled pair of fields with
correlated inputs should grow a nonzero coupling constant; an uncoupled
pair with independent inputs should keep its coupling near zero.
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.layer import LayerConfig
from src.qft_pcn.multifield import MultiFieldConfig, MultiFieldNetwork


def test_correlated_inputs_grow_coupling():
    rng = np.random.default_rng(11)
    xs = np.arange(12)[:, None] - 6
    ys = np.arange(12)[None, :] - 6
    blob = np.exp(-(xs ** 2 + ys ** 2) / 4.0)[None]

    cfg = MultiFieldConfig(
        field_names=["a", "b"],
        layer_configs={
            "a": LayerConfig(channels=1, learn_rate=0.02),
            "b": LayerConfig(channels=1, learn_rate=0.02),
        },
        coupling={("a", "b"): 0.0},
        learn_coupling=True,
        coupling_lr=0.05,
        dt=0.4,
    )
    net = MultiFieldNetwork(12, 12, cfg, rng=rng)

    # Both fields see the same spatial pattern (perfectly correlated).
    for _ in range(60):
        net.step({"a": blob, "b": blob}, learn=True)

    g = abs(net.couplings[("a", "b")])
    assert g > 0.05, f"correlated fields should grow coupling, got |g|={g:.3f}"


def test_uncorrelated_inputs_keep_coupling_small():
    rng = np.random.default_rng(23)
    xs = np.arange(12)[:, None] - 6
    ys = np.arange(12)[None, :] - 6
    blob_a = np.exp(-(xs ** 2 + ys ** 2) / 4.0)[None]
    # Field b sees random noise that doesn't correlate with the blob.
    rng_b = np.random.default_rng(99)

    cfg = MultiFieldConfig(
        field_names=["a", "b"],
        layer_configs={
            "a": LayerConfig(channels=1, learn_rate=0.02),
            "b": LayerConfig(channels=1, learn_rate=0.02),
        },
        coupling={("a", "b"): 0.0},
        learn_coupling=True,
        coupling_lr=0.05,
        dt=0.4,
    )
    net = MultiFieldNetwork(12, 12, cfg, rng=rng)

    couplings = []
    for _ in range(80):
        blob_b = rng_b.standard_normal((1, 12, 12)) * 0.3
        net.step({"a": blob_a, "b": blob_b}, learn=True)
        couplings.append(net.couplings[("a", "b")])

    # The running |g| should stay much smaller than the correlated case.
    final_g = abs(couplings[-1])
    assert final_g < 0.2, (
        f"uncoupled fields shouldn't grow large |g|, got {final_g:.3f}")


def test_multifield_shares_one_manifold():
    rng = np.random.default_rng(0)
    cfg = MultiFieldConfig(
        field_names=["a", "b"],
        layer_configs={
            "a": LayerConfig(channels=1),
            "b": LayerConfig(channels=1),
        },
    )
    net = MultiFieldNetwork(8, 8, cfg, rng=rng)
    assert net.fields["a"].manifold is net.fields["b"].manifold


if __name__ == "__main__":
    test_correlated_inputs_grow_coupling()
    test_uncorrelated_inputs_keep_coupling_small()
    test_multifield_shares_one_manifold()
    print("ok")
