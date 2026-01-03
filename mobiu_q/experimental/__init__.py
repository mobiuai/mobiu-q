"""
Mobiu-Q Experimental Module
===========================
🧪 EXPERIMENTAL: APIs in this module are subject to change.

This module contains cutting-edge features that are functional but
still being refined based on user feedback.

Current experimental features:
- MobiuAttention: O(N) linear attention using Soft Algebra
- MobiuAttentionFast: Optimized version with parallel scan
- MobiuBlock / MobiuBlockFast: Complete transformer blocks

Performance vs Standard Transformer:
    | Seq Length | Speedup | Memory    |
    |------------|---------|-----------|
    | 2,048      | 2.3x    | ~50%      |
    | 4,096      | 3.9x    | ~25%      |
    | 8,192      | 6.0x    | ~12%      |
    | 16,384     | ∞       | Works! 🏆 |
    
    (Transformer OOMs at 16K on typical GPU)

Quality benchmarks (vs Transformer):
    | Benchmark            | Result      |
    |----------------------|-------------|
    | Shakespeare PPL      | Equal       |
    | ListOps Accuracy     | Equal (~80%)|
    | Needle-in-Haystack   | Equal (100%)|

Usage:
    from mobiu_q.experimental import MobiuAttention, MobiuBlock
    
    # Option 1: Replace attention layer
    self.attn = MobiuAttention(d_model=512, num_heads=8)
    
    # Option 2: Use complete block
    self.block = MobiuBlock(d_model=512, num_heads=8)

Recommended: Use MobiuAttentionFast / MobiuBlockFast for production.

⚠️  Please report issues: https://github.com/mobiu-ai/mobiu-q/issues
"""

import warnings

# Issue warning on import to make experimental status clear
warnings.warn(
    "mobiu_q.experimental contains experimental features. "
    "APIs may change in future versions.",
    FutureWarning,
    stacklevel=2
)

# Import from layers (reference implementation - sequential, slower)
from .layers import (
    MobiuAttention as MobiuAttentionRef,
    MobiuBlock as MobiuBlockRef,
)

# Import from layers_fast (recommended - parallel scan, faster)
from .layers_fast import (
    MobiuAttentionFast,
    MobiuBlockFast,
    parallel_scan_simple,
)

# Default exports point to fast versions (recommended)
MobiuAttention = MobiuAttentionFast
MobiuBlock = MobiuBlockFast

__all__ = [
    # Recommended (fast implementations)
    "MobiuAttention",      # Alias for MobiuAttentionFast
    "MobiuBlock",          # Alias for MobiuBlockFast
    "MobiuAttentionFast",
    "MobiuBlockFast",
    # Reference implementations (for understanding/debugging)
    "MobiuAttentionRef",
    "MobiuBlockRef",
    # Utilities
    "parallel_scan_simple",
]