"""
ATen Trees: Rooted Tree Attention Heads

A novel attention mechanism where each head corresponds to a rooted tree
from OEIS A000081, implementing elementary differential attention patterns.

The hyper-chatbot architecture:
    ATen(1) -> [a(1)] = 1 head  (single node tree)
    ATen(2) -> [a(2)] = 1 head  (two-node tree)
    ATen(3) -> [a(3)] = 2 heads (two 3-node trees)
    ATen(4) -> [a(4)] = 4 heads (four 4-node trees)
    ATen(5) -> [a(5)] = 9 heads (nine 5-node trees)
    ...

Where a(n) is the n-th term of OEIS A000081 (rooted trees with n nodes).

Example usage:
    >>> from aten_trees import get_trees, a000081
    >>> print(a000081(5))  # Number of 5-node rooted trees
    9
    >>> trees = get_trees(4)  # Get all 4-node rooted trees
    >>> for t in trees:
    ...     print(t.to_bracket_notation())

    >>> from aten_trees import HyperChatbotAttention
    >>> chatbot = HyperChatbotAttention(num_layers=5)
    >>> chatbot.print_architecture()
"""

__version__ = "0.1.0"
__author__ = "ATen Trees Project"

from .rooted_trees import (
    RootedTree,
    RootedTreeGenerator,
    get_trees,
    a000081,
    a000081_sequence,
    TAU_1, TAU_2, TAU_3A, TAU_3B,
    TAU_4A, TAU_4B, TAU_4C, TAU_4D,
)

from .attention_heads import (
    TreeHeadSpec,
    TreeDimensionMapper,
    ATenTreeLayer,
    HyperChatbotAttention,
    create_aten_specification,
)

from .elementary_differentials import (
    TensorContraction,
    ElementaryDifferentialShape,
    TreeTensorFactory,
    compute_aten_dimensions,
    format_aten_specification,
)

from .hyper_attention import (
    ATenConfig,
    TreeAttentionMask,
    PureATenAttention,
    demonstrate_architecture,
)

# Optional PyTorch components
try:
    from .hyper_attention import (
        TreeHead,
        ATenAttention,
        ATenTransformerLayer,
        HyperChatbotTransformer,
        create_hyper_chatbot,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


__all__ = [
    # Core tree structures
    'RootedTree',
    'RootedTreeGenerator',
    'get_trees',
    'a000081',
    'a000081_sequence',

    # Canonical trees
    'TAU_1', 'TAU_2', 'TAU_3A', 'TAU_3B',
    'TAU_4A', 'TAU_4B', 'TAU_4C', 'TAU_4D',

    # Attention head mapping
    'TreeHeadSpec',
    'TreeDimensionMapper',
    'ATenTreeLayer',
    'HyperChatbotAttention',
    'create_aten_specification',

    # Elementary differentials
    'TensorContraction',
    'ElementaryDifferentialShape',
    'TreeTensorFactory',
    'compute_aten_dimensions',
    'format_aten_specification',

    # Attention implementation
    'ATenConfig',
    'TreeAttentionMask',
    'PureATenAttention',
    'demonstrate_architecture',
]

if HAS_TORCH:
    __all__.extend([
        'TreeHead',
        'ATenAttention',
        'ATenTransformerLayer',
        'HyperChatbotTransformer',
        'create_hyper_chatbot',
    ])
