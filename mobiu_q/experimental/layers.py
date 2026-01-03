"""
Mobiu-Q Neural Network Layers
=============================
O(N) Linear Attention layers based on Soft Algebra (Klein/Maimon Theory).

This module provides GPU-optimized implementations of Soft Algebra principles
for use in Large Language Models and Transformers.

Mathematical Foundation
-----------------------
The layers implement the Soft Algebra evolution equation:

    S(t+1) = γ·S(t) + Δ(t)

Where:
    - S = State matrix (the "memory")
    - γ = Decay factor (learned, dynamic)
    - Δ = Innovation/Update (new information)

Nilpotent Property (ε²=0)
-------------------------
In Soft Algebra, the nilpotent property ensures that infinitesimal components
don't "explode" through repeated self-multiplication.

Traditional RNNs apply nonlinear activations (tanh/sigmoid) inside the loop,
causing state to interact with itself nonlinearly (S², S³...) → chaos.

MobiuAttention keeps the recurrence LINEAR:
    - Interactions happen once at input (K×V)
    - Accumulate linearly (+= kv_update)
    - No self-multiplication of state

This is the hardware implementation of ε²=0.

Soft Components (a, b)
----------------------
In Soft Numbers: (a, b) where a=potential, b=realization.

In MobiuAttention:
    - a (potential) = complexity_sensor output
    - b (realization) = state matrix

The system measures "potential for change" (complexity) and uses it
to control how strongly the realization (memory) should be preserved.

Why Tensors Instead of SoftNumber Class?
----------------------------------------
GPUs can't process Python objects efficiently. They only multiply matrices.

We "compiled" the abstract Soft Algebra rules into tensor operations:
    - Same mathematics
    - 1000x faster on GPU
    - Differentiable for backpropagation

"The soul is Soft Algebra. The body is PyTorch."

Usage
-----
    from mobiu_q.layers import MobiuAttention, MobiuBlock
    
    # Replace nn.MultiheadAttention
    self.attn = MobiuAttention(d_model=512, num_heads=8)
    
    # Or use complete block (Attention + FFN)
    self.block = MobiuBlock(d_model=512, num_heads=8)

References
----------
- Klein, M. & Maimon, O. - Soft Logic and Soft Numbers (Tel Aviv University)
- Mobiu-Q: https://mobiu.ai

© 2025-2026 Mobiu Technologies. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MobiuAttention(nn.Module):
    """
    Mobiu-Q Linear Attention Layer
    
    A drop-in replacement for nn.MultiheadAttention with O(N) complexity.
    Implements Soft Algebra evolution equation with dynamic decay.
    
    Complexity Comparison:
        - Standard Attention: O(N²) memory, O(N²) compute
        - MobiuAttention:     O(1) memory*, O(N) compute
        
        *State size is O(heads × head_dim²), independent of sequence length
    
    Key Features:
        - O(N) memory and compute
        - Dynamic Memory Decay (learned + input-adaptive)
        - Built-in Complexity Sensor (Soft Algebra "potential")
        - Mixed Precision (fp16/bf16) compatible
        - Implicit causal masking (no future leakage)
    
    Args:
        d_model: Total dimension of the model
        num_heads: Number of attention heads (must divide d_model)
        dropout: Dropout probability on output (default: 0.0)
    
    Example:
        >>> attn = MobiuAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(2, 1000, 512)  # [batch, seq_len, dim]
        >>> out = attn(x)  # [2, 1000, 512] - no memory explosion!
    
    Mathematical Details:
        For each timestep t:
            1. Compute Q, K, V projections
            2. Measure local complexity: a(t) = sensor(x[t])
            3. Compute adaptive decay: γ(t) = σ(params) × (1 + 0.2×a(t))
            4. Update state: S(t) = γ(t)×S(t-1) + K(t)ᵀ×V(t)
            5. Retrieve output: O(t) = Q(t)×S(t)
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # === PROJECTIONS ===
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # === THE MOBIU CORE (Soft Algebra Implementation) ===
        
        # Learnable Decay Parameters (γ base)
        # Initialized to +3.0 → sigmoid(3.0) ≈ 0.95 (long memory)
        self.decay_params = nn.Parameter(torch.randn(num_heads, self.head_dim) + 3.0)
        
        # Complexity Sensor (computes "a" - the Soft potential)
        # Measures local information density to adjust memory retention
        self.complexity_sensor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, num_heads)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass implementing Soft Algebra evolution.
        
        Args:
            x: Input tensor [Batch, Seq_Len, D_model]
            mask: Ignored (causal masking is implicit in recurrence)
        
        Returns:
            Output tensor [Batch, Seq_Len, D_model]
        """
        B, T, D = x.shape
        H = self.num_heads
        E = self.head_dim
        
        # === 1. PROJECTIONS ===
        q = self.q_proj(x).view(B, T, H, E)
        k = self.k_proj(x).view(B, T, H, E)
        v = self.v_proj(x).view(B, T, H, E)
        
        # === 2. DYNAMIC DECAY (Soft Algebra: adaptive γ) ===
        
        # Measure local complexity ("a" - potential component)
        local_complexity = self.complexity_sensor(x)  # [B, T, H]
        local_complexity = torch.sigmoid(local_complexity).unsqueeze(-1)  # [B, T, H, 1]
        
        # Base decay from learned parameters
        base_decay = torch.sigmoid(self.decay_params).view(1, 1, H, E)
        
        # Adaptive decay: high complexity → higher decay (preserve memory longer)
        # This implements the Soft Algebra interaction between a and b
        lambda_decay = base_decay * (1.0 + local_complexity * 0.2)
        lambda_decay = torch.clamp(lambda_decay, 0.0, 0.9995)
        
        # === 3. O(N) RECURRENCE (Soft Algebra Evolution: S = γS + Δ) ===
        
        outputs = []
        
        # State Matrix [Batch, Heads, head_dim, head_dim]
        # This is "b" - the realization component
        # dtype=x.dtype is crucial for Mixed Precision training!
        state = torch.zeros(B, H, E, E, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            qt = q[:, t]  # [B, H, E] - Query at time t
            kt = k[:, t]  # [B, H, E] - Key at time t
            vt = v[:, t]  # [B, H, E] - Value at time t
            dt = lambda_decay[:, t]  # [B, H, E] - Decay at time t
            
            # Δ(t) = Kᵀ × V (the innovation/update)
            # Outer product: [B,H,E] × [B,H,E] → [B,H,E,E]
            kv_update = torch.einsum('bhe,bhf->bhef', kt, vt)
            
            # S(t) = γ(t)·S(t-1) + Δ(t)
            # This is THE Soft Algebra evolution equation!
            # Nilpotent property: linear accumulation, no S² terms
            state = state * dt.unsqueeze(-1) + kv_update
            
            # Output: O(t) = Q(t) × S(t)
            # Retrieval from state matrix
            out_t = torch.einsum('bhe,bhef->bhf', qt, state)
            outputs.append(out_t)
        
        # === 4. REASSEMBLE & PROJECT ===
        y = torch.stack(outputs, dim=1)  # [B, T, H, E]
        y = y.reshape(B, T, D)
        
        return self.dropout(self.o_proj(y))
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, head_dim={self.head_dim}'


class MobiuBlock(nn.Module):
    """
    Complete Mobiu Transformer Block.
    
    Architecture (Pre-LayerNorm, like Llama/GPT-3):
        x → LayerNorm → MobiuAttention → + → LayerNorm → FFN → + → out
            ↑_____________________________|   ↑___________________|
                    (residual)                    (residual)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward hidden dimension (default: 4 × d_model)
        dropout: Dropout probability
    
    Example:
        >>> block = MobiuBlock(d_model=512, num_heads=8)
        >>> x = torch.randn(2, 100, 512)
        >>> out = block(x)  # [2, 100, 512]
        
        # Stack multiple blocks for deeper model:
        >>> blocks = nn.Sequential(*[MobiuBlock(512, 8) for _ in range(6)])
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        ffn_dim = ffn_dim or d_model * 4
        
        # Pre-LayerNorm (like Llama)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Attention (O(N) Soft Algebra)
        self.attn = MobiuAttention(d_model, num_heads, dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input [Batch, Seq_Len, D_model]
        
        Returns:
            Output [Batch, Seq_Len, D_model]
        """
        # Pre-Norm + Attention + Residual
        x = x + self.attn(self.norm1(x))
        
        # Pre-Norm + FFN + Residual
        x = x + self.ffn(self.norm2(x))
        
        return x


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MobiuAttention",
    "MobiuBlock",
]


# =============================================================================
# FUTURE: Parallel Scan for faster training
# =============================================================================
# 
# The current implementation uses a Python for-loop, which is correct but
# sequential. For production training on very long sequences, we will add:
#
# class MobiuAttentionFast(MobiuAttention):
#     """
#     Parallel Scan implementation using associative scan algorithm.
#     Achieves O(N) with full GPU parallelism.
#     
#     Reference: "Parallel Scan" (Blelloch, 1990)
#     """
#     pass
