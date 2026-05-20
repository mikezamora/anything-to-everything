"""PCN-QFT hybrid architecture.

A predictive-coding network whose substrate is a dynamic Riemannian manifold,
with optional quantum-circuit generative models and multi-field coupling.

Public surface:
    Manifold2D       - dynamic 2D metric field with LB operator and curvature
    Field            - scalar/multi-channel field on the manifold
    PrecisionField   - non-negative precision (inverse variance) field
    QFTPCNLayer      - single hierarchical layer (Phi, E, Pi + generative map)
    QFTPCNNetwork    - stack of coupled layers
    MultiFieldNetwork - several field types sharing one manifold, with
                       learnable pairwise interactions
    QuantumGenerativeMap - variational quantum circuit
    QuantumConvMap   - VQC tiled as a quantum convolution, slots into a layer
"""

from .manifold import Manifold2D
from .fields import Field, PrecisionField
from .layer import QFTPCNLayer, LayerConfig, ClassicalConvMap, GenerativeMap
from .network import QFTPCNNetwork, NetworkConfig
from .multifield import MultiFieldNetwork, MultiFieldConfig
from .quantum import QuantumGenerativeMap, QuantumConvMap

__all__ = [
    "Manifold2D",
    "Field",
    "PrecisionField",
    "QFTPCNLayer",
    "LayerConfig",
    "ClassicalConvMap",
    "GenerativeMap",
    "QFTPCNNetwork",
    "NetworkConfig",
    "MultiFieldNetwork",
    "MultiFieldConfig",
    "QuantumGenerativeMap",
    "QuantumConvMap",
]
