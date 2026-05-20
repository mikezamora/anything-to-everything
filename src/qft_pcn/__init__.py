"""PCN-QFT hybrid architecture.

A predictive-coding network whose substrate is a dynamic Riemannian manifold:
prediction-error fields source curvature via a stress-energy tensor, and belief
diffusion uses the Laplace-Beltrami operator of the resulting metric.

Public surface:
    Manifold2D       - dynamic 2D metric field with LB operator and curvature
    Field            - scalar/multi-channel field on the manifold
    PrecisionField   - non-negative precision (inverse variance) field
    QFTPCNLayer      - single hierarchical layer (Phi, E, Pi + generative map)
    QFTPCNNetwork    - stack of coupled layers minimizing variational free energy
"""

from .manifold import Manifold2D
from .fields import Field, PrecisionField
from .layer import QFTPCNLayer
from .network import QFTPCNNetwork

__all__ = [
    "Manifold2D",
    "Field",
    "PrecisionField",
    "QFTPCNLayer",
    "QFTPCNNetwork",
]
