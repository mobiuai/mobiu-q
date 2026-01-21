"""
Mobiu-Q — Soft Algebra for Optimization & Attention
====================================================
Version: 3.8.6

A framework built on Soft Algebra (nilpotent ε²=0) enabling:
1. Stable optimization in noisy environments
2. Efficient linear-time attention for long sequences

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STABLE API (Production Ready)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classes:
    | Class          | Use Case                                   |
    |----------------|---------------------------------------------|
    | MobiuOptimizer | PyTorch (RL, LLM, Deep Learning)           |
    | MobiuQCore     | Quantum (VQE, QAOA) & NumPy optimization   |

Methods:
    | Method   | Use Case                                    |
    |----------|---------------------------------------------|
    | standard | Smooth landscapes, chemistry, physics       |
    | deep     | Deep circuits, noisy hardware, complex opt  |
    | adaptive | RL, LLM fine-tuning, high-variance problems |

Quick Start (PyTorch):
    from mobiu_q import MobiuOptimizer
    
    base_opt = torch.optim.Adam(model.parameters(), lr=0.0003)
    opt = MobiuOptimizer(base_opt, method="adaptive", use_soft_algebra=True)
    
    for batch in data:
        loss = criterion(model(batch))
        loss.backward()
        opt.step(loss.item())
    
    opt.end()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧪 EXPERIMENTAL API (Subject to Change)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MobiuAttention - O(N) linear attention using Soft Algebra state tracking

Benefits over standard Transformer attention:
    - O(N) vs O(N²) complexity
    - 2-6x faster for seq > 2K
    - Works with 16K+ context (where Transformer OOMs)
    - Same quality on benchmarks (ListOps, Needle-in-Haystack)

Quick Start:
    from mobiu_q.experimental import MobiuAttention, MobiuBlock
    
    # Drop-in replacement for nn.MultiheadAttention
    self.attn = MobiuAttention(d_model=512, num_heads=8)
    output = self.attn(x)  # x: [batch, seq, dim]

⚠️  EXPERIMENTAL: API may change in future versions.
    Please report issues at https://github.com/mobiu-ai/mobiu-q

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
License:
    Free tier: 20 API calls/month (optimizer only)
    Pro tier: Unlimited - https://app.mobiu.ai
    
    Note: MobiuAttention runs locally, no API calls required.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

__version__ = "3.8.6"
__author__ = "Mobiu Technologies"

# ============================================================================
# STABLE API - Production Ready
# ============================================================================

from .core import (
    # Main optimizers
    MobiuOptimizer,
    MobiuQCore,
    # Frustration Engine
    UniversalFrustrationEngine,
    # Gradient estimation
    Demeasurement, 
    # Utilities
    get_default_lr,
    get_license_key,
    save_license_key,
    activate_license,
    check_status,
    # Constants
    AVAILABLE_OPTIMIZERS,
    DEFAULT_OPTIMIZER,
    METHOD_ALIASES,
    VALID_METHODS,
    API_ENDPOINT,
)

# ============================================================================
# EXPERIMENTAL API - Lazy loaded to avoid torch dependency for quantum users
# ============================================================================

# Don't import experimental at top level - let users import explicitly
# This avoids requiring torch for quantum-only users

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # === Stable API ===
    # Optimizers
    "MobiuOptimizer",
    "MobiuQCore",
    "UniversalFrustrationEngine",
    "Demeasurement",
    # Utilities
    "get_default_lr",
    "get_license_key",
    "save_license_key",
    "activate_license",
    "check_status",
    # Constants
    "AVAILABLE_OPTIMIZERS",
    "DEFAULT_OPTIMIZER",
    "METHOD_ALIASES",
    "VALID_METHODS",
    "API_ENDPOINT",
]