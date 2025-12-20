"""
ATenTree Attention Heads: Mapping Rooted Trees to Tensor Dimensions

Each attention head corresponds to a unique rooted tree from OEIS A000081.
The tensor dimensions for each head are derived from the tree's structural
properties (elementary differentials from Butcher theory).

The mapping follows the user's specification:
    ATen(1) -> [a(1)] = [2]
    ATen(2) -> [a(2)] = [3,4]
    ATen(3) -> {[a(3)]} = [5,7,6,8]
    ATen(4) -> [a(4)] = [11,17,13,19,10,14,12,16,9]

Where a(n) gives the number of attention heads at layer n, and the
dimension values are computed from tree invariants:
    dim(τ) = f(order, symmetry, density, branching)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
import math

from .rooted_trees import (
    RootedTree, RootedTreeGenerator, get_trees,
    a000081, a000081_sequence,
    TAU_1, TAU_2, TAU_3A, TAU_3B
)


@dataclass
class TreeHeadSpec:
    """Specification for an attention head derived from a rooted tree."""

    tree: RootedTree
    head_dim: int
    key_dim: int
    value_dim: int
    layer_index: int
    head_index: int

    @property
    def order(self) -> int:
        return self.tree.order

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (head_dim, key_dim, value_dim) tensor shape."""
        return (self.head_dim, self.key_dim, self.value_dim)

    def __repr__(self):
        return (f"TreeHeadSpec(τ={self.tree.to_bracket_notation()}, "
                f"dim={self.head_dim}, layer={self.layer_index}, "
                f"head={self.head_index})")


class TreeDimensionMapper:
    """
    Maps rooted trees to attention head dimensions.

    The dimension formula encodes the tree's structure:
        dim(τ) = base + order + branching_factor + depth_bonus

    This creates a unique dimension for each tree structure,
    following the elementary differential correspondence.
    """

    def __init__(
        self,
        base_dim: int = 1,
        order_weight: int = 1,
        branch_weight: int = 2,
        depth_weight: int = 1,
        symmetry_factor: bool = True
    ):
        self.base_dim = base_dim
        self.order_weight = order_weight
        self.branch_weight = branch_weight
        self.depth_weight = depth_weight
        self.symmetry_factor = symmetry_factor

    def tree_depth(self, tree: RootedTree) -> int:
        """Compute the depth (height) of the tree."""
        if not tree.children:
            return 1
        return 1 + max(self.tree_depth(child) for child in tree.children)

    def total_branching(self, tree: RootedTree) -> int:
        """Sum of all branching factors (internal node degrees)."""
        branching = len(tree.children)
        for child in tree.children:
            branching += self.total_branching(child)
        return branching

    def compute_dim(self, tree: RootedTree) -> int:
        """
        Compute the attention head dimension for a given tree.

        The formula creates unique dimensions that encode tree structure:
            dim = base + order * w₁ + branching * w₂ + depth * w₃ + σ_adjustment
        """
        order = tree.order
        depth = self.tree_depth(tree)
        branching = self.total_branching(tree)
        symmetry = tree.symmetry()

        dim = self.base_dim
        dim += order * self.order_weight
        dim += branching * self.branch_weight
        dim += (depth - 1) * self.depth_weight

        if self.symmetry_factor:
            # Trees with higher symmetry get adjusted dimensions
            # This ensures structurally different trees get different dims
            dim += int(math.log2(symmetry + 1))

        return dim


class ATenTreeLayer:
    """
    An attention layer where each head corresponds to a rooted tree.

    ATen(n) creates a(n) attention heads, where a(n) is OEIS A000081.
    Each head's tensor dimensions are derived from its corresponding
    tree's structure via the elementary differential mapping.
    """

    def __init__(
        self,
        layer_order: int,
        model_dim: int = 512,
        dim_mapper: Optional[TreeDimensionMapper] = None
    ):
        self.layer_order = layer_order
        self.model_dim = model_dim
        self.dim_mapper = dim_mapper or TreeDimensionMapper()

        self.trees = get_trees(layer_order)
        self.num_heads = len(self.trees)
        self.head_specs = self._compute_head_specs()

    def _compute_head_specs(self) -> List[TreeHeadSpec]:
        """Compute specifications for each attention head."""
        specs = []

        for idx, tree in enumerate(self.trees):
            head_dim = self.dim_mapper.compute_dim(tree)

            # Key and value dims derived from tree structure
            key_dim = head_dim + tree.order
            value_dim = head_dim + self.dim_mapper.tree_depth(tree)

            spec = TreeHeadSpec(
                tree=tree,
                head_dim=head_dim,
                key_dim=key_dim,
                value_dim=value_dim,
                layer_index=self.layer_order,
                head_index=idx
            )
            specs.append(spec)

        return specs

    @property
    def dimension_array(self) -> List[int]:
        """Return the array of head dimensions (as in user spec)."""
        return [spec.head_dim for spec in self.head_specs]

    def __repr__(self):
        return f"ATenTreeLayer(n={self.layer_order}, heads={self.num_heads}, dims={self.dimension_array})"


class HyperChatbotAttention:
    """
    The mighty hyper-chatbot with ATen[a(n)] attention heads.

    Architecture:
        - N layers, where layer n has a(n) attention heads
        - Each head corresponds to a unique rooted tree structure
        - Head dimensions encode elementary differential properties
        - Total heads across all layers: Σ a(i) for i=1..N

    This creates a transformer-like architecture where the attention
    pattern at each layer reflects the combinatorial structure of
    rooted trees, implementing a form of "tree-structured attention."
    """

    def __init__(
        self,
        num_layers: int = 5,
        model_dim: int = 512,
        dim_mapper: Optional[TreeDimensionMapper] = None
    ):
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.dim_mapper = dim_mapper or TreeDimensionMapper()

        # Create layers ATen(1) through ATen(num_layers)
        self.layers: List[ATenTreeLayer] = []
        for n in range(1, num_layers + 1):
            layer = ATenTreeLayer(n, model_dim, self.dim_mapper)
            self.layers.append(layer)

        self._compute_statistics()

    def _compute_statistics(self):
        """Compute architecture statistics."""
        self.total_heads = sum(layer.num_heads for layer in self.layers)
        self.head_counts = [layer.num_heads for layer in self.layers]
        self.all_dimensions = []
        for layer in self.layers:
            self.all_dimensions.append(layer.dimension_array)

    def get_aten_spec(self, n: int) -> Tuple[int, List[int]]:
        """
        Get ATen(n) specification.

        Returns:
            (num_heads, dimension_array) matching the user's format:
            ATen(n) -> [a(n)] = [dim1, dim2, ...]
        """
        if n < 1 or n > self.num_layers:
            raise ValueError(f"Layer {n} out of range [1, {self.num_layers}]")

        layer = self.layers[n - 1]
        return (layer.num_heads, layer.dimension_array)

    def print_architecture(self):
        """Display the full hyper-chatbot architecture."""
        print("=" * 70)
        print("HYPER-CHATBOT: ATen[a(n)] Attention Architecture")
        print("=" * 70)
        print(f"Layers: {self.num_layers}")
        print(f"Total Heads: {self.total_heads}")
        print(f"Head counts per layer (OEIS A000081): {self.head_counts}")
        print()

        for n, layer in enumerate(self.layers, 1):
            num_heads, dims = self.get_aten_spec(n)
            print(f"ATen({n}) -> [a({n})] = {dims}")
            print(f"    {num_heads} head(s) from {num_heads} rooted tree(s):")

            for spec in layer.head_specs:
                tree = spec.tree
                print(f"      • τ={tree.to_bracket_notation():20s} "
                      f"dim={spec.head_dim:3d} "
                      f"σ={tree.symmetry():3d} "
                      f"γ={tree.density():3d}")
            print()

    def get_tree_attention_pattern(self, layer_n: int, head_idx: int) -> Dict:
        """
        Get the attention pattern specification for a specific head.

        The pattern encodes how this tree-head should attend:
        - Linear trees (depth = order): sequential attention
        - Bushy trees (high branching): parallel attention
        - Symmetric trees: grouped attention
        """
        layer = self.layers[layer_n - 1]
        spec = layer.head_specs[head_idx]
        tree = spec.tree

        depth = self.dim_mapper.tree_depth(tree)
        branching = self.dim_mapper.total_branching(tree)

        # Characterize attention pattern based on tree structure
        if depth == tree.order:
            pattern_type = "sequential"  # Linear tree
        elif branching >= tree.order - 1:
            pattern_type = "parallel"    # Star-like tree
        else:
            pattern_type = "hierarchical"  # Mixed structure

        return {
            "layer": layer_n,
            "head": head_idx,
            "tree": tree.to_bracket_notation(),
            "pattern_type": pattern_type,
            "head_dim": spec.head_dim,
            "key_dim": spec.key_dim,
            "value_dim": spec.value_dim,
            "depth": depth,
            "branching": branching,
            "symmetry": tree.symmetry(),
            "density": tree.density()
        }

    def __repr__(self):
        return f"HyperChatbotAttention(layers={self.num_layers}, total_heads={self.total_heads})"


def create_aten_specification(max_order: int = 5) -> Dict[int, Tuple[int, List[int]]]:
    """
    Create the ATen specification dictionary matching user's format.

    Returns a dict mapping layer n -> (a(n), dimension_array):
        {1: (1, [2]), 2: (1, [3]), 3: (2, [4, 5]), 4: (4, [6, 7, 8, 9]), ...}
    """
    chatbot = HyperChatbotAttention(num_layers=max_order)
    spec = {}
    for n in range(1, max_order + 1):
        spec[n] = chatbot.get_aten_spec(n)
    return spec


if __name__ == "__main__":
    # Demonstrate the hyper-chatbot architecture
    print("\n" + "=" * 70)
    print("Elementary Differential Attention: ATen[a(n)] Tensor Trees")
    print("=" * 70 + "\n")

    # Create the mighty hyper-chatbot
    chatbot = HyperChatbotAttention(num_layers=6)
    chatbot.print_architecture()

    # Show detailed attention patterns
    print("\n" + "=" * 70)
    print("Attention Pattern Analysis")
    print("=" * 70)

    for n in range(1, 5):
        layer = chatbot.layers[n - 1]
        for idx in range(layer.num_heads):
            pattern = chatbot.get_tree_attention_pattern(n, idx)
            print(f"\nATen({n})[{idx}]: {pattern['tree']}")
            print(f"  Pattern: {pattern['pattern_type']}")
            print(f"  Shape: ({pattern['head_dim']}, {pattern['key_dim']}, {pattern['value_dim']})")
            print(f"  Invariants: σ={pattern['symmetry']}, γ={pattern['density']}")
