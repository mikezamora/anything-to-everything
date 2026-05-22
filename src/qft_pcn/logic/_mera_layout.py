"""Node-major species-leaf layout for the MERA-native encoder (spec §4.3).

An N-node AST -> 5N species-leaves, node i at leaves [5i, 5i+5), padded
with PAD-leaves to the next power of two. Pure computation; no tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

from .mera_encoding import SPECIES_ORDER, LEAVES_PER_NODE, SPECIES_LEAF_OFFSET


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@dataclass
class MeraLayout:
    n_nodes: int
    n_leaves: int                  # 5*n_nodes padded to a power of two
    L: int                         # log2(n_leaves)
    species_of_leaf: list[str]     # length n_leaves; species name or "PAD"
    node_of_leaf: list[int]        # length n_leaves; AST node index or -1

    def leaf_of(self, node: int, species: str) -> int:
        """Absolute leaf index of `species` of AST `node`."""
        return LEAVES_PER_NODE * node + SPECIES_LEAF_OFFSET[species]


def compute_layout(n_nodes: int) -> MeraLayout:
    """Build the MeraLayout for an n_nodes-node AST."""
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    raw_leaves = LEAVES_PER_NODE * n_nodes
    n_leaves = _next_power_of_two(raw_leaves)
    L = n_leaves.bit_length() - 1   # log2 for an exact power of two
    species_of_leaf: list[str] = []
    node_of_leaf: list[int] = []
    for leaf in range(n_leaves):
        if leaf < raw_leaves:
            node = leaf // LEAVES_PER_NODE
            species = SPECIES_ORDER[leaf % LEAVES_PER_NODE]
            species_of_leaf.append(species)
            node_of_leaf.append(node)
        else:
            species_of_leaf.append("PAD")
            node_of_leaf.append(-1)
    return MeraLayout(
        n_nodes=n_nodes, n_leaves=n_leaves, L=L,
        species_of_leaf=species_of_leaf, node_of_leaf=node_of_leaf,
    )
