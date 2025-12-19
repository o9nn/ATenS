"""
Hyper-Chatbot Attention: Tree-Structured Multi-Head Attention

This module implements the mighty hyper-chatbot architecture where each
attention head corresponds to a rooted tree from OEIS A000081.

The key insight is that rooted trees encode:
1. Hierarchical attention patterns (tree structure → attention mask)
2. Parameter sharing schemes (symmetry σ → weight tying)
3. Computation order (density γ → evaluation order)

Architecture Overview:
    ┌─────────────────────────────────────────────────────────┐
    │  HyperChatbotTransformer                                │
    │  ┌─────────────────────────────────────────────────────┐│
    │  │ ATenLayer(1): 1 head  (τ₁)                         ││
    │  ├─────────────────────────────────────────────────────┤│
    │  │ ATenLayer(2): 1 head  (τ₂)                         ││
    │  ├─────────────────────────────────────────────────────┤│
    │  │ ATenLayer(3): 2 heads (τ₃ₐ, τ₃ᵦ)                   ││
    │  ├─────────────────────────────────────────────────────┤│
    │  │ ATenLayer(4): 4 heads (τ₄ₐ, τ₄ᵦ, τ₄ᶜ, τ₄ᵈ)        ││
    │  ├─────────────────────────────────────────────────────┤│
    │  │ ATenLayer(5): 9 heads (...)                        ││
    │  └─────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────┘

Each ATenLayer(n) has a(n) attention heads from OEIS A000081.
Total heads for n layers: Σᵢ₌₁ⁿ a(i) = 1 + 1 + 2 + 4 + 9 + ...
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import math
from functools import reduce

from .rooted_trees import RootedTree, get_trees, a000081, a000081_sequence
from .attention_heads import TreeHeadSpec, TreeDimensionMapper, ATenTreeLayer
from .elementary_differentials import ElementaryDifferentialShape

# Optional PyTorch integration
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Stub classes for documentation
    class nn:
        class Module:
            pass


@dataclass
class ATenConfig:
    """Configuration for the ATen Tree attention architecture."""
    num_layers: int = 5
    model_dim: int = 512
    max_seq_len: int = 2048
    dropout: float = 0.1
    use_tree_masks: bool = True
    share_symmetric_weights: bool = True


class TreeAttentionMask:
    """
    Generates attention masks based on tree structure.

    Different tree structures induce different attention patterns:
    - Linear trees (τ₂, τ₃ₐ, τ₄ₐ): Causal/sequential attention
    - Star trees (τ₃ᵦ, τ₄ᵈ): Parallel/broadcast attention
    - Mixed trees: Hierarchical attention patterns
    """

    @staticmethod
    def mask_from_tree(tree: RootedTree, seq_len: int) -> List[List[int]]:
        """
        Generate an attention mask from tree structure.

        The mask encodes the tree's information flow:
        - Nodes can attend to ancestors and descendants
        - Siblings in the tree attend in parallel
        """
        depth = TreeAttentionMask._tree_depth(tree)
        branching = len(tree.children)

        # Base causal mask
        mask = [[0] * seq_len for _ in range(seq_len)]

        for i in range(seq_len):
            for j in range(seq_len):
                if TreeAttentionMask._should_attend(tree, i, j, seq_len):
                    mask[i][j] = 1

        return mask

    @staticmethod
    def _tree_depth(tree: RootedTree) -> int:
        if not tree.children:
            return 1
        return 1 + max(TreeAttentionMask._tree_depth(c) for c in tree.children)

    @staticmethod
    def _should_attend(tree: RootedTree, i: int, j: int, seq_len: int) -> bool:
        """Determine if position i should attend to position j."""
        depth = TreeAttentionMask._tree_depth(tree)
        branching = len(tree.children) if tree.children else 1

        # Linear trees: sequential (causal) attention
        if depth == tree.order:
            return j <= i

        # Star trees: all positions attend to root + local
        if branching == tree.order - 1:
            # Attend to position 0 (root) and local window
            window = tree.order
            return j == 0 or abs(i - j) < window

        # Mixed trees: hierarchical attention
        # Positions attend within tree-structured blocks
        block_size = max(1, seq_len // tree.order)
        same_block = (i // block_size) == (j // block_size)
        parent_block = (i // block_size) == ((j // block_size) + 1)

        return same_block or parent_block or j <= i // branching


if HAS_TORCH:

    class TreeHead(nn.Module):
        """
        A single attention head derived from a rooted tree.

        The tree structure determines:
        - head_dim: from tree order and depth
        - attention pattern: from tree topology
        - weight sharing: from tree symmetry
        """

        def __init__(
            self,
            tree: RootedTree,
            model_dim: int,
            head_dim: int,
            dropout: float = 0.1
        ):
            super().__init__()
            self.tree = tree
            self.model_dim = model_dim
            self.head_dim = head_dim
            self.scale = math.sqrt(head_dim)

            # Projections
            self.q_proj = nn.Linear(model_dim, head_dim)
            self.k_proj = nn.Linear(model_dim, head_dim)
            self.v_proj = nn.Linear(model_dim, head_dim)

            self.dropout = nn.Dropout(dropout)

            # Tree-specific parameters
            self.tree_order = tree.order
            self.tree_symmetry = tree.symmetry()

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Apply tree-structured attention.

            Args:
                query: (batch, seq_len, model_dim)
                key: (batch, seq_len, model_dim)
                value: (batch, seq_len, model_dim)
                mask: Optional attention mask

            Returns:
                (batch, seq_len, head_dim)
            """
            Q = self.q_proj(query)
            K = self.k_proj(key)
            V = self.v_proj(value)

            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            # Apply tree-structured mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output = torch.matmul(attn_weights, V)
            return output


    class ATenAttention(nn.Module):
        """
        Multi-head attention where heads are indexed by rooted trees.

        ATen(n) has a(n) attention heads, each corresponding to a
        unique rooted tree of order n.
        """

        def __init__(
            self,
            layer_order: int,
            model_dim: int,
            config: ATenConfig
        ):
            super().__init__()
            self.layer_order = layer_order
            self.model_dim = model_dim
            self.config = config

            # Get trees for this layer
            self.trees = get_trees(layer_order)
            self.num_heads = len(self.trees)

            # Compute head dimensions
            dim_mapper = TreeDimensionMapper()
            self.head_dims = [dim_mapper.compute_dim(t) for t in self.trees]

            # Normalize head dims to fit model_dim
            total_dim = sum(self.head_dims)
            scale = model_dim / max(total_dim, 1)
            self.head_dims = [max(1, int(d * scale)) for d in self.head_dims]

            # Create tree-indexed heads
            self.heads = nn.ModuleList([
                TreeHead(tree, model_dim, dim, config.dropout)
                for tree, dim in zip(self.trees, self.head_dims)
            ])

            # Output projection
            self.out_proj = nn.Linear(sum(self.head_dims), model_dim)

            # Tree masks
            self.register_buffer('tree_masks', None)

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Apply ATen(n) multi-head attention.

            Returns:
                (batch, seq_len, model_dim)
            """
            head_outputs = []

            for i, head in enumerate(self.heads):
                # Each head may use its tree-specific mask
                head_mask = mask
                if self.config.use_tree_masks and mask is None:
                    # Generate mask from tree structure
                    seq_len = query.size(1)
                    tree_mask = TreeAttentionMask.mask_from_tree(
                        self.trees[i], seq_len
                    )
                    head_mask = torch.tensor(tree_mask, device=query.device)

                out = head(query, key, value, head_mask)
                head_outputs.append(out)

            # Concatenate head outputs
            concat = torch.cat(head_outputs, dim=-1)

            # Project back to model_dim
            output = self.out_proj(concat)
            return output


    class ATenTransformerLayer(nn.Module):
        """
        A transformer layer with ATen tree-structured attention.
        """

        def __init__(
            self,
            layer_order: int,
            model_dim: int,
            ff_dim: int,
            config: ATenConfig
        ):
            super().__init__()
            self.layer_order = layer_order

            # ATen attention
            self.attention = ATenAttention(layer_order, model_dim, config)

            # Feed-forward
            self.ff = nn.Sequential(
                nn.Linear(model_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(ff_dim, model_dim),
                nn.Dropout(config.dropout)
            )

            # Layer norms
            self.norm1 = nn.LayerNorm(model_dim)
            self.norm2 = nn.LayerNorm(model_dim)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Self-attention with residual
            attn_out = self.attention(x, x, x, mask)
            x = self.norm1(x + attn_out)

            # Feed-forward with residual
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)

            return x


    class HyperChatbotTransformer(nn.Module):
        """
        The mighty hyper-chatbot transformer!

        Architecture with ATen[a(n)] attention heads per layer:
        - Layer 1: 1 head (single-node tree)
        - Layer 2: 1 head (two-node tree)
        - Layer 3: 2 heads (two 3-node trees)
        - Layer 4: 4 heads (four 4-node trees)
        - Layer 5: 9 heads (nine 5-node trees)
        - ...

        Total heads: Σ a(n) for n=1..N
        """

        def __init__(self, config: ATenConfig):
            super().__init__()
            self.config = config

            # Compute architecture stats
            self.head_counts = a000081_sequence(config.num_layers)
            self.total_heads = sum(self.head_counts)

            # Token embedding
            self.embedding = nn.Embedding(50000, config.model_dim)
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.model_dim)

            # ATen transformer layers
            ff_dim = config.model_dim * 4
            self.layers = nn.ModuleList([
                ATenTransformerLayer(n + 1, config.model_dim, ff_dim, config)
                for n in range(config.num_layers)
            ])

            # Output head
            self.ln_f = nn.LayerNorm(config.model_dim)
            self.lm_head = nn.Linear(config.model_dim, 50000, bias=False)

        def forward(
            self,
            input_ids: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass through the hyper-chatbot.

            Args:
                input_ids: (batch, seq_len) token indices
                mask: Optional attention mask

            Returns:
                (batch, seq_len, vocab_size) logits
            """
            batch_size, seq_len = input_ids.shape

            # Embeddings
            tok_emb = self.embedding(input_ids)
            pos = torch.arange(seq_len, device=input_ids.device)
            pos_emb = self.pos_embedding(pos)
            x = tok_emb + pos_emb

            # Pass through ATen layers
            for layer in self.layers:
                x = layer(x, mask)

            # Output
            x = self.ln_f(x)
            logits = self.lm_head(x)

            return logits

        def print_architecture(self):
            """Display the hyper-chatbot architecture."""
            print("=" * 60)
            print("HYPER-CHATBOT TRANSFORMER")
            print("=" * 60)
            print(f"Layers: {self.config.num_layers}")
            print(f"Model dim: {self.config.model_dim}")
            print(f"Total attention heads: {self.total_heads}")
            print(f"Head distribution (OEIS A000081): {self.head_counts}")
            print()

            for n, (count, layer) in enumerate(zip(self.head_counts, self.layers), 1):
                trees = get_trees(n)
                tree_strs = [t.to_bracket_notation() for t in trees]
                print(f"ATen({n}): {count} head(s)")
                print(f"  Trees: {', '.join(tree_strs)}")
            print("=" * 60)


# Pure Python implementation (no PyTorch dependency)
class PureATenAttention:
    """
    Pure Python implementation of ATen attention for demonstration.

    This shows the conceptual structure without requiring PyTorch.
    """

    def __init__(self, layer_order: int, model_dim: int = 512):
        self.layer_order = layer_order
        self.model_dim = model_dim
        self.trees = get_trees(layer_order)
        self.num_heads = len(self.trees)

        dim_mapper = TreeDimensionMapper()
        self.head_dims = [dim_mapper.compute_dim(t) for t in self.trees]

    def describe(self) -> str:
        """Return a description of this attention layer."""
        lines = [f"ATen({self.layer_order}) Attention Layer:"]
        lines.append(f"  Number of heads: {self.num_heads}")
        lines.append(f"  Trees and dimensions:")

        for tree, dim in zip(self.trees, self.head_dims):
            lines.append(
                f"    • {tree.to_bracket_notation():15s} "
                f"dim={dim}, σ={tree.symmetry()}, γ={tree.density()}"
            )

        return "\n".join(lines)


def create_hyper_chatbot(
    num_layers: int = 5,
    model_dim: int = 512
) -> 'HyperChatbotTransformer':
    """Create a hyper-chatbot with ATen tree attention."""
    if not HAS_TORCH:
        raise ImportError("PyTorch required for HyperChatbotTransformer")

    config = ATenConfig(num_layers=num_layers, model_dim=model_dim)
    return HyperChatbotTransformer(config)


def demonstrate_architecture(num_layers: int = 5):
    """Demonstrate the ATen attention architecture."""
    print("\n" + "=" * 70)
    print("ATen[a(n)] HYPER-CHATBOT ATTENTION ARCHITECTURE")
    print("=" * 70)

    print("\nLayer structure based on OEIS A000081:")
    print("-" * 50)

    total_heads = 0
    for n in range(1, num_layers + 1):
        layer = PureATenAttention(n)
        print(f"\n{layer.describe()}")
        total_heads += layer.num_heads

    print("\n" + "-" * 50)
    print(f"Total attention heads across all layers: {total_heads}")
    print(f"Head counts: {a000081_sequence(num_layers)}")


if __name__ == "__main__":
    demonstrate_architecture(6)

    if HAS_TORCH:
        print("\n\nCreating PyTorch HyperChatbotTransformer...")
        model = create_hyper_chatbot(num_layers=5, model_dim=256)
        model.print_architecture()

        # Test forward pass
        print("\nTest forward pass:")
        x = torch.randint(0, 1000, (2, 32))  # batch=2, seq_len=32
        with torch.no_grad():
            out = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
