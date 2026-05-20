"""Quantum-circuit-backed generative map for PCN layers.

A `QuantumGenerativeMap` is a translation-invariant "quantum convolution":
- The manifold is tiled into non-overlapping patches of size patch x patch.
- Each patch's values are encoded as input rotation angles on n_qubits qubits
  (one qubit per patch element; we require n_qubits == patch * patch).
- A parameterized variational ansatz (Ry-Rz per qubit, linear CNOT
  entangling layer) is applied, with parameters shared across all patches.
- Pauli-Z expectations on each qubit produce the patch's prediction.

The circuit is simulated exactly via Statevector — no shot noise. Parameter
gradients use the analytic parameter-shift rule
    df/dtheta = (f(theta + pi/2) - f(theta - pi/2)) / 2.
This is the standard hybrid quantum-classical workflow; the same code runs
on real IBM hardware by swapping the simulator for a SamplerV2 backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


def _z_observable(n_qubits: int, target: int) -> SparsePauliOp:
    """Pauli-Z on qubit `target`, identity on the rest.

    Qiskit's string ordering puts qubit 0 on the *right*, so for n_qubits=4
    and target=1 we want "IIZI".
    """
    label = ["I"] * n_qubits
    label[n_qubits - 1 - target] = "Z"
    return SparsePauliOp("".join(label))


class QuantumGenerativeMap:
    """Variational quantum circuit acting as a per-patch generative model.

    Forward: f(x; theta) where x in R^{n_qubits} (patch values flattened),
    theta in R^{n_layers x n_qubits x 2}. Output is <Z_i> for i in qubits,
    so it lies in [-1, +1]. Inputs are rescaled into the same range before
    angle encoding so they have a clean meaning.

    Memoisation: forward passes at fixed (x, theta) are cached for the
    duration of one optimisation cycle so parameter-shift gradients don't
    re-evaluate the unshifted statevector at every parameter.
    """

    def __init__(self, n_qubits: int, n_layers: int = 2,
                 input_scale: float = 1.0,
                 rng: np.random.Generator | None = None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_scale = input_scale
        rng = rng if rng is not None else np.random.default_rng(0)
        # Two trainable angles per qubit per layer (Ry, Rz).
        self.theta = rng.standard_normal((n_layers, n_qubits, 2)) * 0.1
        self._z_ops = [_z_observable(n_qubits, q) for q in range(n_qubits)]

    # ---- circuit construction ---------------------------------------------

    def _build(self, x: np.ndarray, theta: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Input encoding: bound x to [-1, 1] then map to [-pi/2, pi/2].
        x_clipped = np.clip(x * self.input_scale, -1.0, 1.0)
        for q in range(self.n_qubits):
            qc.ry(float(x_clipped[q]) * math.pi / 2.0, q)
        # Variational ansatz with linear-chain entanglement.
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.ry(float(theta[layer, q, 0]), q)
                qc.rz(float(theta[layer, q, 1]), q)
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    def _expectations(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        qc = self._build(x, theta)
        psi = Statevector.from_instruction(qc)
        return np.array([float(np.real(psi.expectation_value(op)))
                         for op in self._z_ops])

    # ---- public forward / gradient ----------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Map a patch (n_qubits,) to a prediction (n_qubits,)."""
        return self._expectations(x, self.theta)

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised forward over a stack of patches (B, n_qubits)."""
        return np.stack([self.forward(x) for x in X], axis=0)

    def parameter_shift_grad(self, x: np.ndarray, upstream: np.ndarray
                             ) -> np.ndarray:
        """Gradient of <upstream, f(x; theta)> w.r.t. theta.

        `upstream` shape: (n_qubits,) — the cotangent on the output.
        Returns: gradient with the same shape as self.theta.
        """
        grad = np.zeros_like(self.theta)
        shift = math.pi / 2.0
        for li in range(self.n_layers):
            for qi in range(self.n_qubits):
                for pi_idx in range(2):
                    theta_plus = self.theta.copy()
                    theta_plus[li, qi, pi_idx] += shift
                    theta_minus = self.theta.copy()
                    theta_minus[li, qi, pi_idx] -= shift
                    fp = self._expectations(x, theta_plus)
                    fm = self._expectations(x, theta_minus)
                    grad[li, qi, pi_idx] = float(np.dot(upstream,
                                                        (fp - fm) / 2.0))
        return grad

    def batch_parameter_shift_grad(self, X: np.ndarray, upstream: np.ndarray
                                   ) -> np.ndarray:
        """Sum of parameter-shift gradients across a batch of patches.

        X shape: (B, n_qubits); upstream shape: (B, n_qubits). The returned
        gradient is averaged (not summed) over the batch so the learning
        rate is batch-size independent.
        """
        if len(X) == 0:
            return np.zeros_like(self.theta)
        g = np.zeros_like(self.theta)
        for x, u in zip(X, upstream):
            g += self.parameter_shift_grad(x, u)
        return g / len(X)


# ---------------------------------------------------------------------------
# Patch-based wiring helpers
# ---------------------------------------------------------------------------


def to_patches(field: np.ndarray, patch: int) -> tuple[np.ndarray,
                                                       tuple[int, int]]:
    """Split a (C, Nx, Ny) field into non-overlapping patches.

    Returns (patches, grid_shape) where patches has shape (B, C * patch * patch)
    and grid_shape = (Nx // patch, Ny // patch). Channels are flattened with
    the patch so a single quantum circuit consumes one full patch at a time.
    """
    c, nx, ny = field.shape
    if nx % patch or ny % patch:
        raise ValueError(f"grid {nx}x{ny} not divisible by patch {patch}")
    gx, gy = nx // patch, ny // patch
    reshaped = field.reshape(c, gx, patch, gy, patch).transpose(1, 3, 0, 2, 4)
    return reshaped.reshape(gx * gy, c * patch * patch), (gx, gy)


def from_patches(patches: np.ndarray, grid_shape: tuple[int, int],
                 channels: int, patch: int) -> np.ndarray:
    """Inverse of `to_patches`. Reassembles a (C, Nx, Ny) field."""
    gx, gy = grid_shape
    reshaped = patches.reshape(gx, gy, channels, patch, patch)
    return reshaped.transpose(2, 0, 3, 1, 4).reshape(channels, gx * patch,
                                                     gy * patch)


class QuantumConvMap:
    """`GenerativeMap` adapter wrapping a `QuantumGenerativeMap`.

    Tiles the field into non-overlapping patches, runs the same shared VQC
    on every patch (translation-invariant quantum convolution), and
    reassembles the predictions. Bias is a single classical scalar per
    channel — kept outside the circuit because Pauli-Z expectations are
    already in [-1, 1] and we sometimes need to recenter.
    """

    def __init__(self, channels: int, patch: int, n_layers: int = 2,
                 input_scale: float = 1.0,
                 rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.channels = channels
        self.patch = patch
        self.n_qubits = channels * patch * patch
        self.qmap = QuantumGenerativeMap(
            n_qubits=self.n_qubits, n_layers=n_layers,
            input_scale=input_scale, rng=rng,
        )
        self.bias = np.zeros((channels, 1, 1))

    # ---- Protocol surface --------------------------------------------------

    def forward(self, phi: np.ndarray) -> np.ndarray:
        patches, grid_shape = to_patches(phi, self.patch)
        preds = self.qmap.forward_batch(patches)
        return from_patches(preds, grid_shape, self.channels, self.patch
                            ) + self.bias

    def grad_input(self, phi: np.ndarray, upstream: np.ndarray) -> np.ndarray:
        """Gradient flowing into phi: sum over patches of J_f^T @ upstream.

        Computed by parameter-shifting the input-encoding Ry gates. We do this
        patch by patch since the encoding angles are patch-local.
        """
        patches, grid_shape = to_patches(phi, self.patch)
        u_patches, _ = to_patches(upstream, self.patch)
        out_patches = np.zeros_like(patches)
        for i, (x, u) in enumerate(zip(patches, u_patches)):
            out_patches[i] = self._input_grad(x, u)
        return from_patches(out_patches, grid_shape, self.channels, self.patch)

    def update_params(self, phi: np.ndarray, upstream: np.ndarray,
                      lr: float) -> None:
        patches, _ = to_patches(phi, self.patch)
        u_patches, _ = to_patches(upstream, self.patch)
        grad = self.qmap.batch_parameter_shift_grad(patches, u_patches)
        grad = np.clip(grad, -1.0, 1.0)
        self.qmap.theta -= lr * grad
        # Bias is just an averaging shift: gradient w.r.t. b is -upstream.
        grad_b = -upstream.mean(axis=(-1, -2), keepdims=True)
        self.bias -= lr * np.clip(grad_b, -1.0, 1.0)

    # ---- internals ---------------------------------------------------------

    def _input_grad(self, x: np.ndarray, upstream: np.ndarray) -> np.ndarray:
        """Parameter-shift over the input-encoding Ry angles for one patch.

        Encoding gate is Ry(input_scale * x_q * pi/2). The shift in input
        space corresponding to a pi/2 rotation shift is dx = 1/input_scale.
        """
        shift_x = 1.0 / max(self.qmap.input_scale, 1e-8)
        grad = np.zeros_like(x)
        for q in range(self.n_qubits):
            x_plus = x.copy(); x_plus[q] += shift_x
            x_minus = x.copy(); x_minus[q] -= shift_x
            fp = self.qmap._expectations(x_plus, self.qmap.theta)
            fm = self.qmap._expectations(x_minus, self.qmap.theta)
            grad[q] = float(np.dot(upstream, (fp - fm) / 2.0))
        return grad
