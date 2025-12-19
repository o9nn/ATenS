"""
OEIS A000081: Rooted Trees and Elementary Differentials

This module generates rooted trees corresponding to OEIS A000081,
the sequence counting unlabeled rooted trees with n nodes:
    a(n): 0, 1, 1, 2, 4, 9, 20, 48, 115, 286, 719, 1842, ...

Each rooted tree represents an elementary differential in the
Butcher theory of Runge-Kutta methods, and here we map them
to attention head tensor shapes for the hyper-chatbot architecture.

The mapping ATen(n) -> [a(n)] creates attention heads where:
- ATen(1) has 1 head with shape derived from the single 1-node tree
- ATen(2) has 1 head from the 2-node tree
- ATen(3) has 2 heads from 2 distinct 3-node trees
- ATen(4) has 4 heads from 4 distinct 4-node trees
- ATen(5) has 9 heads from 9 distinct 5-node trees
- ...

The tensor dimension at each head is computed from the tree's
structural properties: order (n), symmetry (σ), and density (γ).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterator, Optional
from functools import lru_cache
from collections import Counter
import math


@dataclass(frozen=True)
class RootedTree:
    """
    A rooted tree represented as a tuple of subtrees.

    The empty tuple () represents the single-node tree (just a root).
    A tree with children is represented as a sorted tuple of subtrees.

    Examples:
        τ₁ = RootedTree(()) - single node: •
        τ₂ = RootedTree((τ₁,)) - two nodes: ─•
        τ₃ₐ = RootedTree((τ₂,)) - linear: ─•─•
        τ₃ᵦ = RootedTree((τ₁, τ₁)) - branching: <•
    """
    children: Tuple['RootedTree', ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Ensure children are sorted for canonical form
        if self.children:
            sorted_children = tuple(sorted(self.children, key=lambda t: t.to_tuple()))
            object.__setattr__(self, 'children', sorted_children)

    @property
    def order(self) -> int:
        """Number of nodes in the tree (|τ|)."""
        return 1 + sum(child.order for child in self.children)

    @property
    def num_children(self) -> int:
        """Number of branches from root."""
        return len(self.children)

    @lru_cache(maxsize=None)
    def symmetry(self) -> int:
        """
        Symmetry coefficient σ(τ) - counts automorphisms.

        σ(τ) = ∏ᵢ (mᵢ! · σ(τᵢ)^mᵢ)
        where mᵢ is the multiplicity of subtree τᵢ.
        """
        if not self.children:
            return 1

        child_counts = Counter(self.children)
        result = 1
        for child, count in child_counts.items():
            result *= math.factorial(count) * (child.symmetry() ** count)
        return result

    @lru_cache(maxsize=None)
    def density(self) -> int:
        """
        Density coefficient γ(τ) - appears in B-series.

        γ(τ) = |τ| · ∏ᵢ γ(τᵢ)
        """
        if not self.children:
            return 1
        return self.order * math.prod(child.density() for child in self.children)

    @lru_cache(maxsize=None)
    def alpha(self) -> int:
        """
        α(τ) = |τ|! / (σ(τ) · γ(τ))

        The number of monotonic labellings of the tree.
        """
        return math.factorial(self.order) // (self.symmetry() * self.density())

    def to_tuple(self) -> tuple:
        """Convert to nested tuple representation for hashing/comparison."""
        return tuple(child.to_tuple() for child in self.children)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        if not isinstance(other, RootedTree):
            return False
        return self.to_tuple() == other.to_tuple()

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()

    def __le__(self, other):
        return self.to_tuple() <= other.to_tuple()

    def __gt__(self, other):
        return self.to_tuple() > other.to_tuple()

    def __ge__(self, other):
        return self.to_tuple() >= other.to_tuple()

    def to_ascii(self, prefix: str = "", is_last: bool = True) -> str:
        """Generate ASCII art representation of the tree."""
        if not self.children:
            return "●"

        lines = ["●"]
        for i, child in enumerate(self.children):
            is_last_child = (i == len(self.children) - 1)
            connector = "└── " if is_last_child else "├── "
            child_prefix = "    " if is_last_child else "│   "

            child_str = child.to_ascii(prefix + child_prefix, is_last_child)
            child_lines = child_str.split('\n')
            lines.append(connector + child_lines[0])
            for line in child_lines[1:]:
                lines.append(child_prefix + line)

        return '\n'.join(lines)

    def to_bracket_notation(self) -> str:
        """
        Butcher's bracket notation: τ = [τ₁, τ₂, ..., τₖ]
        The single node is represented as τ or ∅.
        """
        if not self.children:
            return "τ"
        return "[" + ", ".join(c.to_bracket_notation() for c in self.children) + "]"

    def elementary_weight(self, order_map: Dict[int, int]) -> List[int]:
        """
        Compute the elementary differential weight vector.

        For a tree τ of order n, returns a vector of length n
        representing the tensor contraction pattern.
        """
        weights = [self.order]
        for child in self.children:
            weights.extend(child.elementary_weight(order_map))
        return weights


class RootedTreeGenerator:
    """
    Generator for all rooted trees of a given order.

    Uses the recursive structure: a rooted tree of order n consists
    of a root connected to a multiset of rooted trees whose orders
    sum to n-1.
    """

    def __init__(self, max_order: int = 10):
        self.max_order = max_order
        self._cache: Dict[int, List[RootedTree]] = {}
        self._generate_all()

    def _generate_all(self):
        """Pre-generate all trees up to max_order."""
        for n in range(1, self.max_order + 1):
            self._cache[n] = list(self._trees_of_order(n))

    def _trees_of_order(self, n: int) -> Iterator[RootedTree]:
        """Generate all rooted trees with exactly n nodes."""
        if n == 1:
            yield RootedTree(())
            return

        # Generate all ways to partition n-1 among subtrees
        for partition in self._partitions_to_trees(n - 1):
            yield RootedTree(tuple(partition))

    def _partitions_to_trees(self, total: int, max_part: Optional[int] = None) -> Iterator[List[RootedTree]]:
        """
        Generate all multisets of trees whose orders sum to total.

        Args:
            total: Target sum of tree orders
            max_part: Maximum order of any tree (for canonical ordering)
        """
        if total == 0:
            yield []
            return

        if max_part is None:
            max_part = total

        for part_size in range(min(max_part, total), 0, -1):
            # Get all trees of this order
            trees_of_size = self.trees(part_size)

            for tree in trees_of_size:
                # Recursively generate remaining parts
                for rest in self._partitions_to_trees(total - part_size, part_size):
                    # Only include if tree >= all trees in rest (canonical form)
                    if not rest or tree >= rest[0]:
                        yield [tree] + rest

    def trees(self, order: int) -> List[RootedTree]:
        """Get all rooted trees of the given order."""
        if order not in self._cache:
            self._cache[order] = list(self._trees_of_order(order))
        return self._cache[order]

    def count(self, order: int) -> int:
        """Return a(n) - the number of rooted trees of order n (OEIS A000081)."""
        return len(self.trees(order))

    def a000081_sequence(self, n: int) -> List[int]:
        """Return first n terms of OEIS A000081."""
        return [self.count(i) for i in range(1, n + 1)]


# Pre-instantiate for convenience
_generator = RootedTreeGenerator(max_order=12)


def get_trees(order: int) -> List[RootedTree]:
    """Get all rooted trees of the given order."""
    global _generator
    if order > _generator.max_order:
        _generator = RootedTreeGenerator(max_order=order)
    return _generator.trees(order)


def a000081(n: int) -> int:
    """Return a(n) from OEIS A000081 - number of rooted trees with n nodes."""
    return len(get_trees(n))


def a000081_sequence(length: int) -> List[int]:
    """Return first 'length' terms of OEIS A000081 starting from n=1."""
    return [a000081(i) for i in range(1, length + 1)]


# Canonical tree instances for reference
TAU_1 = RootedTree(())  # Single node: •
TAU_2 = RootedTree((TAU_1,))  # Linear 2: ─•
TAU_3A = RootedTree((TAU_2,))  # Linear 3: ─•─•
TAU_3B = RootedTree((TAU_1, TAU_1))  # Branching 3: <•
TAU_4A = RootedTree((TAU_3A,))  # Linear 4
TAU_4B = RootedTree((TAU_3B,))  # Cherry on stick
TAU_4C = RootedTree((TAU_2, TAU_1))  # Y with tail
TAU_4D = RootedTree((TAU_1, TAU_1, TAU_1))  # 3-star


if __name__ == "__main__":
    print("OEIS A000081: Number of rooted trees with n unlabeled nodes")
    print("=" * 60)

    seq = a000081_sequence(10)
    print(f"a(1..10) = {seq}")
    print()

    for n in range(1, 6):
        trees = get_trees(n)
        print(f"\nOrder {n}: {len(trees)} tree(s)")
        print("-" * 40)
        for i, tree in enumerate(trees, 1):
            print(f"\n  τ_{n},{i}: {tree.to_bracket_notation()}")
            print(f"  σ={tree.symmetry()}, γ={tree.density()}, α={tree.alpha()}")
            for line in tree.to_ascii().split('\n'):
                print(f"    {line}")
