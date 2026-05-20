"""Field containers for the PCN-QFT hybrid.

Fields live on a 2D grid that represents a discretized Riemannian manifold.
Channels are the field's internal degrees of freedom (analogous to spinor
indices). All numerics are numpy; there are no autograd graphs — gradients
needed by the PDE are computed analytically in `dynamics.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Field:
    """Multi-channel scalar field Phi(x, y) on a 2D grid.

    Shape convention: values has shape (C, Nx, Ny). Channel-first matches the
    way convolutions and per-channel reductions are typically written.
    """

    values: np.ndarray

    @classmethod
    def zeros(cls, channels: int, nx: int, ny: int) -> "Field":
        return cls(values=np.zeros((channels, nx, ny), dtype=np.float64))

    @classmethod
    def randn(cls, channels: int, nx: int, ny: int, scale: float = 0.01,
              rng: np.random.Generator | None = None) -> "Field":
        rng = rng if rng is not None else np.random.default_rng(0)
        return cls(values=rng.standard_normal((channels, nx, ny)) * scale)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape  # type: ignore[return-value]

    @property
    def channels(self) -> int:
        return self.values.shape[0]

    def copy(self) -> "Field":
        return Field(values=self.values.copy())

    def __iadd__(self, other: "Field | np.ndarray") -> "Field":
        self.values += other.values if isinstance(other, Field) else other
        return self


@dataclass
class PrecisionField:
    """Scalar precision field Pi(x, y) >= eps.

    Represents the inverse variance of prediction error at each manifold point.
    Stored in log-space (`log_pi`) to keep updates additive and Pi positive.
    The Pi getter exponentiates lazily.
    """

    log_pi: np.ndarray
    eps: float = 1e-4

    @classmethod
    def ones(cls, nx: int, ny: int) -> "PrecisionField":
        return cls(log_pi=np.zeros((nx, ny), dtype=np.float64))

    @property
    def pi(self) -> np.ndarray:
        return np.exp(self.log_pi) + self.eps

    @property
    def shape(self) -> tuple[int, int]:
        return self.log_pi.shape  # type: ignore[return-value]

    def copy(self) -> "PrecisionField":
        return PrecisionField(log_pi=self.log_pi.copy(), eps=self.eps)
