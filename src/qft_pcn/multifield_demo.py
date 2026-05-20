"""Multi-field demo: two coupled fields tracking a joint pattern.

Setup:
  - "shape" field: classical generative map (3x3 conv), sees a Gaussian
    blob at position (cx, cy).
  - "motion" field: quantum generative map (4-qubit VQC on 2x2 patches),
    sees a velocity-encoded pattern at the same position.

The pattern is correlated across modalities — wherever the shape blob is,
the motion vector is there too. With learnable coupling, the network
should discover a nonzero coupling constant g_{shape,motion}, and the
joint free energy should decrease faster than the sum of independent
energies would.

This is the QFT "interaction vertex" idea in cortical-stream form: two
modalities that explain each other's residuals develop strong coupling;
modalities that don't, don't.

Run:  python -m src.qft_pcn.multifield_demo
"""

from __future__ import annotations

import numpy as np

from .layer import LayerConfig
from .multifield import MultiFieldConfig, MultiFieldNetwork
from .quantum import QuantumConvMap


def make_shape_obs(nx: int, ny: int, t: float) -> np.ndarray:
    xs = np.arange(nx)[:, None] - nx / 2
    ys = np.arange(ny)[None, :] - ny / 2
    cx = 2.5 * np.cos(0.07 * t)
    cy = 2.5 * np.sin(0.07 * t)
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / 4.0)[None]


def make_motion_obs(nx: int, ny: int, t: float) -> np.ndarray:
    """Same spatial pattern, but with a sign that flips with motion direction.

    Conceptually: shape says "object is here"; motion says "object is moving
    rightward / leftward". Spatially co-located, semantically different —
    exactly the kind of pair where cross-field coupling is informative.
    """
    xs = np.arange(nx)[:, None] - nx / 2
    ys = np.arange(ny)[None, :] - ny / 2
    cx = 2.5 * np.cos(0.07 * t)
    cy = 2.5 * np.sin(0.07 * t)
    mag = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / 4.0)
    direction = float(np.sign(np.sin(0.07 * t)) or 1.0)
    return (direction * mag)[None]


def run(steps: int = 60, nx: int = 8, ny: int = 8, verbose: bool = True
        ) -> list[dict]:
    rng = np.random.default_rng(123)

    # Shape field uses the classical conv generative map.
    shape_cfg = LayerConfig(channels=1, diffusion=0.1, learn_rate=0.01)

    # Motion field uses a VQC on 2x2 patches: 4 qubits, 2 ansatz layers.
    # On an 8x8 grid that means 16 patches per forward pass.
    q_map = QuantumConvMap(channels=1, patch=2, n_layers=2,
                           input_scale=1.0, rng=rng)
    motion_cfg = LayerConfig(channels=1, diffusion=0.1, learn_rate=0.05,
                             learn_every=5, gen_map=q_map)

    mf_cfg = MultiFieldConfig(
        field_names=["shape", "motion"],
        layer_configs={"shape": shape_cfg, "motion": motion_cfg},
        coupling={("shape", "motion"): 0.0},  # start with no coupling
        learn_coupling=True,
        coupling_lr=0.02,
        dt=0.4,
    )
    net = MultiFieldNetwork(nx, ny, mf_cfg, rng=rng)

    history: list[dict] = []
    for t in range(steps):
        obs = {
            "shape": make_shape_obs(nx, ny, t),
            "motion": make_motion_obs(nx, ny, t),
        }
        info = net.step(obs, learn=True)
        history.append(info)
        if verbose and (t < 3 or t % 10 == 0 or t == steps - 1):
            g = info["couplings"][("motion", "shape")]
            print(f"step={t:3d}  F_total={info['total_F']:+.3e}  "
                  f"F_shape={info['per_field_F']['shape']:+.3e}  "
                  f"F_motion={info['per_field_F']['motion']:+.3e}  "
                  f"g_sm={g:+.4f}")

    if verbose:
        final_g = history[-1]["couplings"][("motion", "shape")]
        initial_F = history[0]["total_F"]
        final_F = history[-1]["total_F"]
        print(f"\nLearned coupling g_shape-motion: {final_g:+.4f}")
        print(f"Free energy: {initial_F:+.3e} -> {final_F:+.3e} "
              f"(reduction: {initial_F - final_F:+.3e})")

    return history


if __name__ == "__main__":
    run()
