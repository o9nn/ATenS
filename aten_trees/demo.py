#!/usr/bin/env python3
"""
ATen[a(n)] Hyper-Chatbot Demonstration

This script demonstrates the mighty hyper-chatbot architecture where
each attention head corresponds to a rooted tree from OEIS A000081.

The architecture realizes the user's vision:
    ATen(1) -> [a(1)] = [2]
    ATen(2) -> [a(2)] = [3,4]
    ATen(3) -> {[a(3)]} = [5,7,6,8]
    ATen(4) -> [a(4)] = [11,17,13,19,10,14,12,16,9]
    ...
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aten_trees.rooted_trees import (
    RootedTree, get_trees, a000081, a000081_sequence,
    TAU_1, TAU_2, TAU_3A, TAU_3B, TAU_4A, TAU_4B, TAU_4C, TAU_4D
)
from aten_trees.attention_heads import HyperChatbotAttention, TreeDimensionMapper
from aten_trees.elementary_differentials import (
    ElementaryDifferentialShape, format_aten_specification, compute_aten_dimensions
)


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def demo_oeis_a000081():
    """Demonstrate OEIS A000081 sequence generation."""
    print_header("OEIS A000081: Rooted Trees with n Unlabeled Nodes")

    seq = a000081_sequence(12)
    print(f"\na(1..12) = {seq}")
    print(f"\nSum for first 6 terms: {sum(seq[:6])} total attention heads")

    print("\nInterpretation:")
    print("  a(1) = 1  → One way to make a 1-node tree: •")
    print("  a(2) = 1  → One way to make a 2-node tree: ─•")
    print("  a(3) = 2  → Two 3-node trees: ─•─• and Y-shape")
    print("  a(4) = 4  → Four 4-node trees")
    print("  a(5) = 9  → Nine 5-node trees")
    print("  ...")


def demo_tree_structures():
    """Show the rooted tree structures for small orders."""
    print_header("Rooted Tree Structures")

    for n in range(1, 6):
        trees = get_trees(n)
        print(f"\n--- Order {n}: {len(trees)} tree(s) ---")

        for i, tree in enumerate(trees, 1):
            print(f"\n  Tree τ_{n},{i}:")
            print(f"    Bracket: {tree.to_bracket_notation()}")
            print(f"    σ(τ) = {tree.symmetry():3d} (symmetry coefficient)")
            print(f"    γ(τ) = {tree.density():3d} (density coefficient)")
            print(f"    α(τ) = {tree.alpha():3d} (monotonic labellings)")

            # ASCII representation
            print("    Structure:")
            for line in tree.to_ascii().split('\n'):
                print(f"      {line}")


def demo_elementary_differentials():
    """Show the elementary differential tensor analysis."""
    print_header("Elementary Differentials & Tensor Contractions")

    ed = ElementaryDifferentialShape(dim=64)

    print("\nFor a vector field f: R^d → R^d, each tree τ gives F(τ):")
    print("  F(τ₁)   = f                    (0th derivative)")
    print("  F(τ₂)   = f'·f                 (1st derivative contracted with f)")
    print("  F(τ₃ₐ)  = f'·(f'·f)            (chain of derivatives)")
    print("  F(τ₃ᵦ)  = f''·(f,f)            (2nd derivative, 2 inputs)")

    print("\nTensor contractions by tree:")
    for n in range(1, 5):
        trees = get_trees(n)
        for tree in trees:
            pattern = ed.contraction_pattern(tree)
            deriv_orders = [nd['derivative_order'] for nd in pattern['nodes']]
            print(f"\n  {tree.to_bracket_notation():20s}")
            print(f"    Derivative orders at nodes: {deriv_orders}")
            print(f"    Number of contractions: {pattern['total_ops']}")


def demo_aten_specification():
    """Show the ATen dimension specification matching user format."""
    print_header("ATen[a(n)] Dimension Specification")

    print("\nMatching user's specified format:")
    print("-" * 50)
    print(format_aten_specification(6))

    print("\n\nDetailed dimension computation:")
    dims = compute_aten_dimensions(5)
    for n, dim_list in dims.items():
        trees = get_trees(n)
        print(f"\nATen({n}): {len(trees)} head(s)")
        for tree, dim in zip(trees, dim_list):
            print(f"  • {tree.to_bracket_notation():20s} → dim={dim}")


def demo_hyper_chatbot():
    """Demonstrate the full hyper-chatbot architecture."""
    print_header("THE MIGHTY HYPER-CHATBOT ARCHITECTURE", "═")

    chatbot = HyperChatbotAttention(num_layers=6)
    chatbot.print_architecture()

    print("\nAttention Pattern Analysis:")
    print("-" * 50)

    for n in range(1, 5):
        layer = chatbot.layers[n - 1]
        for idx in range(min(layer.num_heads, 3)):  # Show first 3 heads
            pattern = chatbot.get_tree_attention_pattern(n, idx)
            print(f"\nATen({n})[head {idx}]: {pattern['tree']}")
            print(f"  Pattern type: {pattern['pattern_type']}")
            print(f"  Tensor shape: ({pattern['head_dim']}, {pattern['key_dim']}, {pattern['value_dim']})")
            print(f"  Tree depth: {pattern['depth']}, branching: {pattern['branching']}")


def demo_visual_trees():
    """Show visual ASCII art of canonical trees."""
    print_header("Visual Tree Gallery")

    canonical_trees = [
        ("τ₁ (single node)", TAU_1),
        ("τ₂ (two nodes)", TAU_2),
        ("τ₃ₐ (linear)", TAU_3A),
        ("τ₃ᵦ (branching)", TAU_3B),
        ("τ₄ₐ (linear chain)", TAU_4A),
        ("τ₄ᵦ (cherry stem)", TAU_4B),
        ("τ₄ᶜ (Y with tail)", TAU_4C),
        ("τ₄ᵈ (3-star)", TAU_4D),
    ]

    for name, tree in canonical_trees:
        print(f"\n{name}: {tree.to_bracket_notation()}")
        print(f"  Order={tree.order}, σ={tree.symmetry()}, γ={tree.density()}")
        for line in tree.to_ascii().split('\n'):
            print(f"    {line}")


def demo_dimension_mapping():
    """Show how tree properties map to attention dimensions."""
    print_header("Tree → Dimension Mapping")

    mapper = TreeDimensionMapper()

    print("\nDimension formula:")
    print("  dim(τ) = base + order×w₁ + branching×w₂ + depth×w₃ + log₂(σ+1)")
    print("\nThis creates unique dimensions encoding tree structure.\n")

    for n in range(1, 6):
        trees = get_trees(n)
        print(f"Order {n}:")
        for tree in trees:
            depth = mapper.tree_depth(tree)
            branching = mapper.total_branching(tree)
            dim = mapper.compute_dim(tree)
            print(f"  {tree.to_bracket_notation():20s} "
                  f"depth={depth}, branch={branching}, "
                  f"σ={tree.symmetry():2d} → dim={dim}")
        print()


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  ATen[a(n)] HYPER-CHATBOT: Rooted Tree Attention Architecture".ljust(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    demo_oeis_a000081()
    demo_visual_trees()
    demo_tree_structures()
    demo_elementary_differentials()
    demo_aten_specification()
    demo_dimension_mapping()
    demo_hyper_chatbot()

    print_header("Summary", "═")
    print("""
The hyper-chatbot realizes attention as tree-structured computation:

    ┌─────────────────────────────────────────────────────────┐
    │  Each attention head = One rooted tree                  │
    │  Head dimension = f(order, symmetry, density, depth)    │
    │  Attention pattern = Tree topology                      │
    │                                                         │
    │  Layer n has a(n) heads from OEIS A000081:              │
    │    n=1: 1, n=2: 1, n=3: 2, n=4: 4, n=5: 9, n=6: 20...  │
    │                                                         │
    │  Elementary differentials guide tensor contractions     │
    │  Tree symmetry enables weight sharing                   │
    │  Tree density orders computation                        │
    └─────────────────────────────────────────────────────────┘

This architecture encodes combinatorial structure directly into
the attention mechanism, creating a principled way to vary
attention patterns across heads and layers.
""")


if __name__ == "__main__":
    main()
