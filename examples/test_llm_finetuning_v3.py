#!/usr/bin/env python3
"""
================================================================================
MOBIU-Q LLM FINE-TUNING BENCHMARK v3
================================================================================
Testing Momentum + Soft Algebra via Cloud API

Changes from v2:
- SA runs on server (no local SoftNumber / soft_algebra_core)
- Momentum coefficient controlled by client (not hardcoded)
- Uses existing mobiu_q_step endpoint with return_adjustments=True
- License key from environment variable MOBIU_LICENSE_KEY
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from dataclasses import dataclass
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

from mobiu_q import MobiuOptimizer

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Training
    num_epochs: int = 30
    batch_size: int = 8
    base_lr: float = 5e-3
    momentum: float = 0.9          # Client-controlled momentum
    num_seeds: int = 10
    
    # Model
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    vocab_size: int = 1000
    max_seq_len: int = 64
    
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    
    # Data
    num_train_samples: int = 500
    num_eval_samples: int = 100
    
    # Noise
    gradient_noise: float = 0.01

config = Config()

LICENSE_KEY = os.environ.get('MOBIU_LICENSE_KEY', 'YOUR_KEY')
if not LICENSE_KEY:
    print("⚠️  MOBIU_LICENSE_KEY not set. Set it with:")
    print("   export MOBIU_LICENSE_KEY='your-key-here'")
    print("   Continuing without SA (Mobiu runs will use plain Momentum fallback)")
    print()

print("=" * 70)
print("🤖 MOBIU-Q LLM FINE-TUNING BENCHMARK v3 (Cloud API)")
print("=" * 70)
print(f"Epochs: {config.num_epochs} | Seeds: {config.num_seeds}")
print(f"Optimizer: Momentum (LR={config.base_lr}, μ={config.momentum})")
print(f"Model: {config.num_layers}L, {config.model_dim}d, {config.num_heads}H")
print(f"Gradient noise: {config.gradient_noise}")
print(f"License: {'✓' if LICENSE_KEY else '✗ (fallback mode)'}")
print("=" * 70)

# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"\n📱 Device: {device}")

# ============================================================================
# MINI TRANSFORMER MODEL
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        
    def forward(self, x, mask=None):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        
        x = self.token_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_final(x)
        return self.lm_head(x)


# ============================================================================
# LoRA LAYER
# ============================================================================

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank, alpha):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        for param in self.original.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def apply_lora(model, rank, alpha, target_modules=['q_proj', 'v_proj']):
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, LoRALayer(module, rank, alpha))
    return model


def get_lora_params(model):
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    return lora_params


# ============================================================================
# SYNTHETIC DATASET
# ============================================================================

class SyntheticTextDataset:
    def __init__(self, num_samples, seq_len, vocab_size, task='next_token', seed=42):
        self.rng = np.random.RandomState(seed)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.task = task
        self.data = self._generate_data()
        
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            base = self.rng.randint(1, self.vocab_size // 10, size=self.seq_len // 4)
            seq = np.tile(base, 4) + self.rng.randint(0, 3, size=self.seq_len)
            seq = np.clip(seq, 0, self.vocab_size - 1)
            data.append(seq)
        return np.array(data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


# ============================================================================
# OPTIMIZER SETUP (same pattern as SB3 benchmark)
# ============================================================================
#
# Baseline:  torch.optim.SGD(params, lr=..., momentum=...)
# Mobiu:     MobiuOptimizer(base_sgd, ...) wrapping the same SGD
#
# No custom optimizer classes needed.
#


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def add_gradient_noise(model, noise_scale):
    """Add gradient noise to simulate real-world variance."""
    if noise_scale > 0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data += torch.randn_like(p.grad) * noise_scale * p.grad.abs().mean()


def train_epoch(model, dataloader, optimizer, device, mobiu=None):
    """
    Standard training loop.
    
    Args:
        mobiu: If provided, a MobiuOptimizer wrapper. Loss is fed via set_metric().
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        mask = create_causal_mask(x.size(1), device)
        
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        add_gradient_noise(model, config.gradient_noise)
        
        optimizer.step()
        
        # Feed loss to Mobiu (SA adjustments happen server-side)
        if mobiu is not None:
            mobiu.set_metric(-loss.item())  # Negative because minimizing
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            mask = create_causal_mask(x.size(1), device)
            logits = model(x, mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def run_lora_finetune(seed, use_mobiu=False):
    """Run LoRA fine-tuning with SGD+Momentum, optionally wrapped by Mobiu."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    model = MiniLLM(
        vocab_size=config.vocab_size,
        d_model=config.model_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len
    )
    model = apply_lora(model, config.lora_rank, config.lora_alpha)
    model = model.to(device)
    
    lora_params = get_lora_params(model)
    
    train_dataset = SyntheticTextDataset(
        config.num_train_samples, config.max_seq_len, config.vocab_size, seed=seed
    )
    eval_dataset = SyntheticTextDataset(
        config.num_eval_samples, config.max_seq_len, config.vocab_size, seed=seed+1000
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size)
    
    # Base optimizer: standard PyTorch SGD with momentum
    base_opt = torch.optim.SGD(lora_params, lr=config.base_lr, momentum=config.momentum)
    
    mobiu = None
    if use_mobiu:
        mobiu = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method='adaptive',
            use_soft_algebra=True,
            sync_interval=50,
            verbose=False
        )
        optimizer = mobiu
    else:
        optimizer = base_opt
    
    eval_losses = []
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, mobiu=mobiu)
        eval_loss = evaluate(model, eval_loader, device)
        eval_losses.append(eval_loss)
    
    if mobiu is not None:
        mobiu.end()
    
    return {
        'final_eval_loss': eval_losses[-1],
        'best_eval_loss': min(eval_losses),
        'eval_history': eval_losses,
    }


def run_full_finetune(seed, use_mobiu=False):
    """Run full fine-tuning with SGD+Momentum, optionally wrapped by Mobiu."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    model = MiniLLM(
        vocab_size=config.vocab_size,
        d_model=config.model_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    train_dataset = SyntheticTextDataset(
        config.num_train_samples, config.max_seq_len, config.vocab_size, seed=seed
    )
    eval_dataset = SyntheticTextDataset(
        config.num_eval_samples, config.max_seq_len, config.vocab_size, seed=seed+1000
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size)
    
    lr = config.base_lr / 10
    
    base_opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.momentum)
    
    mobiu = None
    if use_mobiu:
        mobiu = MobiuOptimizer(
            base_opt,
            license_key=LICENSE_KEY,
            method='adaptive',
            use_soft_algebra=True,
            sync_interval=50,
            verbose=False
        )
        optimizer = mobiu
    else:
        optimizer = base_opt
    
    eval_losses = []
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, mobiu=mobiu)
        eval_loss = evaluate(model, eval_loader, device)
        eval_losses.append(eval_loss)
    
    if mobiu is not None:
        mobiu.end()
    
    return {
        'final_eval_loss': eval_losses[-1],
        'best_eval_loss': min(eval_losses),
        'eval_history': eval_losses,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    # LoRA Fine-tuning
    print("\n" + "=" * 70)
    print("📊 BENCHMARK 1: LoRA FINE-TUNING (Momentum)")
    print("=" * 70)
    
    momentum_lora = []
    mobiu_lora = []
    
    for seed in range(config.num_seeds):
        print(f"\nSeed {seed + 1}/{config.num_seeds}", end="", flush=True)
        
        torch_state = torch.get_rng_state()
        np_state    = np.random.get_state()

        print(" [Momentum", end="", flush=True)
        m_result = run_lora_finetune(seed, use_mobiu=False)
        print(f"={m_result['final_eval_loss']:.3f}]", end="", flush=True)
        momentum_lora.append(m_result)

        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)

        print(" [Mobiu", end="", flush=True)
        mobiu_result = run_lora_finetune(seed, use_mobiu=True)
        print(f"={mobiu_result['final_eval_loss']:.3f}]", end="", flush=True)
        mobiu_lora.append(mobiu_result)
        
        winner = "✓" if mobiu_result['final_eval_loss'] < m_result['final_eval_loss'] else ""
        print(f" {winner}")
    
    # Results
    m_losses = [r['final_eval_loss'] for r in momentum_lora]
    mobiu_losses = [r['final_eval_loss'] for r in mobiu_lora]
    
    print("\n" + "-" * 70)
    print("LoRA RESULTS (Momentum + SA vs Momentum):")
    print("-" * 70)
    print(f"Momentum: {np.mean(m_losses):.4f} ± {np.std(m_losses):.4f}")
    print(f"Mobiu:    {np.mean(mobiu_losses):.4f} ± {np.std(mobiu_losses):.4f}")
    
    improvement = (np.mean(m_losses) - np.mean(mobiu_losses)) / np.mean(m_losses) * 100
    print(f"Improvement: {improvement:+.1f}%")
    
    wins = sum(1 for m, mo in zip(m_losses, mobiu_losses) if mo < m)
    print(f"Win rate: {wins}/{config.num_seeds} ({wins/config.num_seeds*100:.0f}%)")
    
    try:
        stat, p_val = wilcoxon([m - mo for m, mo in zip(m_losses, mobiu_losses)])
        print(f"p-value: {p_val:.6f}")
    except:
        p_val = 1.0
    
    # Full Fine-tuning
    print("\n" + "=" * 70)
    print("📊 BENCHMARK 2: FULL FINE-TUNING (Momentum)")
    print("=" * 70)
    
    momentum_full = []
    mobiu_full = []
    
    for seed in range(config.num_seeds):
        print(f"\nSeed {seed + 1}/{config.num_seeds}", end="", flush=True)
        
        torch_state = torch.get_rng_state()
        np_state    = np.random.get_state()

        print(" [Momentum", end="", flush=True)
        m_result = run_full_finetune(seed, use_mobiu=False)
        print(f"={m_result['final_eval_loss']:.3f}]", end="", flush=True)
        momentum_full.append(m_result)

        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)

        print(" [Mobiu", end="", flush=True)
        mobiu_result = run_full_finetune(seed, use_mobiu=True)
        print(f"={mobiu_result['final_eval_loss']:.3f}]", end="", flush=True)
        mobiu_full.append(mobiu_result)
        
        winner = "✓" if mobiu_result['final_eval_loss'] < m_result['final_eval_loss'] else ""
        print(f" {winner}")
    
    m_losses_full = [r['final_eval_loss'] for r in momentum_full]
    mobiu_losses_full = [r['final_eval_loss'] for r in mobiu_full]
    
    print("\n" + "-" * 70)
    print("FULL FINE-TUNING RESULTS:")
    print("-" * 70)
    print(f"Momentum: {np.mean(m_losses_full):.4f} ± {np.std(m_losses_full):.4f}")
    print(f"Mobiu:    {np.mean(mobiu_losses_full):.4f} ± {np.std(mobiu_losses_full):.4f}")
    
    improvement_full = (np.mean(m_losses_full) - np.mean(mobiu_losses_full)) / np.mean(m_losses_full) * 100
    print(f"Improvement: {improvement_full:+.1f}%")
    
    wins_full = sum(1 for m, mo in zip(m_losses_full, mobiu_losses_full) if mo < m)
    print(f"Win rate: {wins_full}/{config.num_seeds} ({wins_full/config.num_seeds*100:.0f}%)")
    
    try:
        stat, p_val_full = wilcoxon([m - mo for m, mo in zip(m_losses_full, mobiu_losses_full)])
        print(f"p-value: {p_val_full:.6f}")
    except:
        p_val_full = 1.0
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 SUMMARY: Momentum + SA vs Momentum")
    print("=" * 70)
    print(f"""
┌────────────────────────────────────────────────────────────────────┐
│  LoRA Fine-tuning:                                                 │
│    Improvement: {improvement:+.1f}%                                           │
│    Win Rate: {wins}/{config.num_seeds} ({wins/config.num_seeds*100:.0f}%)                                               │
│    p-value: {p_val:.4f}                                                │
├────────────────────────────────────────────────────────────────────┤
│  Full Fine-tuning:                                                 │
│    Improvement: {improvement_full:+.1f}%                                           │
│    Win Rate: {wins_full}/{config.num_seeds} ({wins_full/config.num_seeds*100:.0f}%)                                               │
│    p-value: {p_val_full:.4f}                                                │
├────────────────────────────────────────────────────────────────────┤
│  Previous results:                                                 │
│    Soft Prompt Tuning: +18.1%                                      │
│    RL (LunarLander): +129.7%                                       │
│    RL (MuJoCo): +118.6%                                            │
└────────────────────────────────────────────────────────────────────┘
""")
    
    if wins >= 7 or wins_full >= 7:
        print("🎉 SIGNIFICANT IMPROVEMENT DETECTED!")
    elif wins >= 5 or wins_full >= 5:
        print("✅ Positive trend, may need more seeds")
    else:
        print("⚠️ Results inconclusive")


if __name__ == "__main__":
    main()
