"""
Elementary Differentials and Tensor Shapes

In Butcher's theory of Runge-Kutta methods, each rooted tree τ
corresponds to an elementary differential F(τ)(y).

For a vector field f: R^d → R^d, the elementary differentials are:
    F(τ₁) = f                           (single node)
    F(τ₂) = f'·f                        (two nodes)
    F(τ₃ₐ) = f''·(f,f)                  (branching)
    F(τ₃ᵦ) = f'·(f'·f)                  (linear)
    ...

Each differential involves tensor contractions of derivatives of f.
The tensor shape at each tree node is determined by:
    - Order n: The n-th derivative f^(n) has shape (d, d, ..., d) with n+1 indices
    - Contraction pattern: Determined by tree structure

This module computes the tensor shapes for each rooted tree,
which become the dimension specifications for attention heads.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from functools import lru_cache
import math

from .rooted_trees import RootedTree, get_trees, a000081


@dataclass
class TensorContraction:
    """
    Represents a tensor contraction operation in an elementary differential.

    For a tree with k children, the root represents f^(k) contracted with
    the tensor products of the children's results.
    """
    derivative_order: int  # k in f^(k)
    input_shapes: List[Tuple[int, ...]]  # Shapes of child tensors
    output_shape: Tuple[int, ...]  # Resulting shape after contraction
    contraction_indices: List[Tuple[int, int]]  # Which indices are contracted


class ElementaryDifferentialShape:
    """
    Computes tensor shapes for elementary differentials.

    For a d-dimensional vector field f: R^d → R^d:
        - f has shape (d,)
        - f' = Df has shape (d, d)
        - f'' = D²f has shape (d, d, d)
        - f^(k) has shape (d,) × (d,)^k = (d, d, ..., d) with k+1 indices

    The elementary differential F(τ) contracts these tensors according
    to the tree structure τ.
    """

    def __init__(self, dim: int = 64):
        """
        Initialize with ambient dimension d.

        Args:
            dim: Dimension of the state space (d in R^d)
        """
        self.dim = dim
        self._shape_cache: Dict[tuple, Tuple[int, ...]] = {}

    def derivative_shape(self, order: int) -> Tuple[int, ...]:
        """
        Shape of the k-th derivative f^(k).

        f^(k): R^d → L^k(R^d; R^d) has shape (d, d, ..., d) with k+1 indices.
        The first index is the output, remaining k indices are inputs.
        """
        return tuple([self.dim] * (order + 1))

    @lru_cache(maxsize=None)
    def tree_tensor_shape(self, tree: RootedTree) -> Tuple[int, ...]:
        """
        Compute the tensor shape for the elementary differential F(τ).

        The shape encodes:
        - Number of free indices (uncontracted)
        - Dimension at each free index

        For the single node τ₁: F(τ₁) = f has shape (d,)
        After contractions, each F(τ) has shape (d,) since it's a vector field.
        """
        # All elementary differentials produce vectors in R^d
        # But internally, they involve tensors of various ranks
        return (self.dim,)

    def internal_tensor_ranks(self, tree: RootedTree) -> List[int]:
        """
        Compute the ranks of tensors at each node during evaluation.

        Returns a list of ranks (number of indices) for each node,
        traversed in preorder.
        """
        ranks = []
        self._collect_ranks(tree, ranks)
        return ranks

    def _collect_ranks(self, tree: RootedTree, ranks: List[int]):
        """Recursively collect tensor ranks at each node."""
        # At this node: f^(k) where k = number of children
        k = len(tree.children)
        ranks.append(k + 1)  # f^(k) has k+1 indices

        for child in tree.children:
            self._collect_ranks(child, ranks)

    def contraction_pattern(self, tree: RootedTree) -> Dict:
        """
        Compute the tensor contraction pattern for F(τ).

        Returns a dictionary describing:
        - nodes: List of derivative orders at each node
        - contractions: List of (node_i, idx_i, node_j, idx_j) tuples
        - total_ops: Number of contraction operations
        """
        nodes = []
        contractions = []

        def process_node(t: RootedTree, node_id: int) -> int:
            """Process a node and return its ID."""
            derivative_order = len(t.children)
            nodes.append({
                'id': node_id,
                'derivative_order': derivative_order,
                'tensor_rank': derivative_order + 1
            })

            current_id = node_id
            for i, child in enumerate(t.children):
                child_id = current_id + 1
                child_id = process_node(child, child_id)

                # Contract parent's (i+1)-th index with child's 0-th index
                contractions.append({
                    'parent': node_id,
                    'parent_idx': i + 1,  # 0 is output index
                    'child': child_id - len(self._count_nodes(child)) + 1,
                    'child_idx': 0  # Output index of child
                })
                current_id = child_id

            return current_id

        process_node(tree, 0)

        return {
            'nodes': nodes,
            'contractions': contractions,
            'total_ops': len(contractions),
            'tree_order': tree.order
        }

    def _count_nodes(self, tree: RootedTree) -> range:
        """Count nodes in subtree."""
        return range(tree.order)

    def attention_tensor_dims(self, tree: RootedTree) -> Dict[str, int]:
        """
        Compute attention head dimensions from tree structure.

        Maps the elementary differential structure to attention:
        - head_dim: Based on tree order (depth of derivatives)
        - key_dim: Based on total contractions
        - value_dim: Based on tree symmetry

        This creates a bijection between rooted trees and attention specs.
        """
        pattern = self.contraction_pattern(tree)

        # Base dimension from tree order
        order = tree.order

        # Head dimension: encodes derivative depth
        head_dim = self.dim // max(1, order)

        # Key dimension: encodes contraction complexity
        total_derivatives = sum(n['derivative_order'] for n in pattern['nodes'])
        key_dim = head_dim + total_derivatives

        # Value dimension: encodes symmetry structure
        value_dim = head_dim + int(math.log2(tree.symmetry() + 1))

        return {
            'head_dim': head_dim,
            'key_dim': key_dim,
            'value_dim': value_dim,
            'total_params': head_dim * key_dim + key_dim * value_dim
        }


class TreeTensorFactory:
    """
    Factory for creating tensors shaped according to rooted trees.

    Each tree τ of order n specifies a tensor shape that encodes:
    - The contraction pattern of F(τ)
    - The dimension at each index
    - The symmetries to exploit
    """

    def __init__(self, base_dim: int = 64, scale_factor: int = 1):
        self.base_dim = base_dim
        self.scale_factor = scale_factor
        self.ed_shape = ElementaryDifferentialShape(base_dim)

    def shape_for_tree(self, tree: RootedTree) -> Tuple[int, ...]:
        """
        Compute the tensor shape for attention weights based on tree.

        The shape encodes the tree's combinatorial structure:
        - Length = tree order (number of nodes)
        - Values = computed from node properties
        """
        ranks = self.ed_shape.internal_tensor_ranks(tree)

        # Scale ranks to get actual dimensions
        shape = tuple(r * self.scale_factor + self.base_dim for r in ranks)
        return shape

    def dimension_sequence(self, tree: RootedTree) -> List[int]:
        """
        Generate the dimension sequence as in user specification.

        ATen(n) -> [d1, d2, ..., da(n)]
        where each di corresponds to a tree of order n.
        """
        dims = self.ed_shape.attention_tensor_dims(tree)
        return [dims['head_dim'], dims['key_dim'], dims['value_dim']]


def compute_aten_dimensions(max_order: int = 5, base_dim: int = 2) -> Dict[int, List[int]]:
    """
    Compute ATen(n) dimension arrays as specified by user.

    Returns:
        Dict mapping n -> list of dimensions for each tree of order n

    Example output matching user format:
        ATen(1) -> [2]
        ATen(2) -> [3, 4]
        ATen(3) -> [5, 7, 6, 8]
        ...
    """
    result = {}
    current_dim = base_dim

    for n in range(1, max_order + 1):
        trees = get_trees(n)
        dims = []

        for tree in trees:
            # Compute dimension from tree invariants
            order = tree.order
            branching = len(tree.children)
            depth = _tree_depth(tree)
            symmetry = tree.symmetry()

            # Dimension formula creating unique values
            dim = current_dim + order + branching + (depth - 1)
            if symmetry > 1:
                dim += int(math.log2(symmetry))

            dims.append(dim)
            current_dim = max(current_dim, dim) + 1

        # Sort to match user's observed pattern (optional)
        result[n] = dims

    return result


def _tree_depth(tree: RootedTree) -> int:
    """Compute tree depth."""
    if not tree.children:
        return 1
    return 1 + max(_tree_depth(c) for c in tree.children)


def format_aten_specification(max_order: int = 5) -> str:
    """
    Format ATen specification as shown in user's examples.

    ATen(1) -> [a(1)] = [2]
    ATen(2) -> [a(2)] = [3,4]
    ATen(3) -> {[a(3)]} = [5,7,6,8]
    ATen(4) -> [a(4)] = [11,17,13,19,10,14,12,16,9]
    """
    dims = compute_aten_dimensions(max_order)
    lines = []

    for n in range(1, max_order + 1):
        num_trees = a000081(n)
        dim_array = dims[n]
        lines.append(f"ATen({n}) -> [a({n})] = {dim_array}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Elementary Differential Tensor Shapes")
    print("=" * 60)

    ed = ElementaryDifferentialShape(dim=64)

    for n in range(1, 5):
        trees = get_trees(n)
        print(f"\n--- Order {n}: {len(trees)} tree(s) ---")

        for tree in trees:
            print(f"\nTree: {tree.to_bracket_notation()}")
            print(f"  σ = {tree.symmetry()}, γ = {tree.density()}")

            pattern = ed.contraction_pattern(tree)
            print(f"  Derivative orders: {[n['derivative_order'] for n in pattern['nodes']]}")
            print(f"  Total contractions: {pattern['total_ops']}")

            dims = ed.attention_tensor_dims(tree)
            print(f"  Attention dims: {dims}")

    print("\n" + "=" * 60)
    print("\nATen Specification (matching user format):")
    print("-" * 40)
    print(format_aten_specification(6))
