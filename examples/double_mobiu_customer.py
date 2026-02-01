#!/usr/bin/env python3
"""
================================================================================
🔥 DOUBLE MOBIU BENCHMARK - CUSTOMER VIEW TEST
================================================================================
The Ultimate Test: MobiuAttention + MobiuOptimizer Together

This test shows what a CUSTOMER would experience with different combinations:
1. Pure Adam + Standard Attention (baseline - what customer has)
2. Pure Adam + Mobiu Attention (attention enhancement only)
3. Adam + Mobiu-Q + Standard Attention (optimizer enhancement only)
4. Adam + Mobiu-Q + Mobiu Attention (DOUBLE MOBIU - full stack!)

Task: Shakespeare Character-Level Language Modeling
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import urllib.request

from mobiu_q import MobiuOptimizer

# ============================================================
# CONFIGURATION
# ============================================================

LICENSE_KEY = "YOUR_KEY"

SEQ_LEN = 256
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4
BATCH_SIZE = 32
EPOCHS = 5
STEPS_PER_EPOCH = 100
LR = 3e-4
METHOD = "adaptive"

# ============================================================
# DEVICE SELECTION
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

DEVICE = get_device()

# ============================================================
# IMPORTS
# ============================================================

MOBIU_ATTENTION_AVAILABLE = False

try:
    from mobiu_q.experimental import MobiuBlockFast
    MOBIU_ATTENTION_AVAILABLE = True
except ImportError:
    print("⚠️  MobiuAttention not available (experimental feature)")


# ============================================================
# STANDARD TRANSFORMER BLOCK
# ============================================================

class StandardAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class StandardBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attn = StandardAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ============================================================
# LANGUAGE MODEL
# ============================================================

class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, max_seq: int, use_mobiu_attention: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        
        if use_mobiu_attention and MOBIU_ATTENTION_AVAILABLE:
            self.blocks = nn.ModuleList([
                MobiuBlockFast(d_model, num_heads) for _ in range(num_layers)
            ])
            self.attention_type = "MobiuAttention"
        else:
            self.blocks = nn.ModuleList([
                StandardBlock(d_model, num_heads) for _ in range(num_layers)
            ])
            self.attention_type = "Standard"
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        
        for block in self.blocks:
            h = block(h)
        
        return self.head(self.norm(h))


# ============================================================
# DATA
# ============================================================

def load_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache_path = "/tmp/shakespeare.txt"
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return f.read()
    
    print("📥 Downloading Shakespeare...")
    urllib.request.urlretrieve(url, cache_path)
    with open(cache_path, 'r') as f:
        return f.read()


# ============================================================
# TRAINING - CUSTOMER VIEW
# ============================================================

def train_config(name: str, use_mobiu_attention: bool, use_mobiu_optimizer: bool,
                 data: torch.Tensor, vocab_size: int, seed: int = 42):
    """Train a configuration - Customer View approach."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SimpleLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq=SEQ_LEN,
        use_mobiu_attention=use_mobiu_attention
    ).to(DEVICE)
    
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    if use_mobiu_optimizer:
        # Customer adds Mobiu-Q like this:
        optimizer = MobiuOptimizer(
            base_optimizer,
            license_key=LICENSE_KEY,
            method=METHOD,
            verbose=False
        )
        optimizer_type = "Adam+Mobiu"
    else:
        # Baseline: Pure Adam (no Mobiu at all!)
        optimizer = base_optimizer
        optimizer_type = "Pure Adam"
    
    n_train = int(len(data) * 0.9)
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - SEQ_LEN - 1, (BATCH_SIZE,))
        x = torch.stack([d[i:i+SEQ_LEN] for i in ix]).to(DEVICE)
        y = torch.stack([d[i+1:i+SEQ_LEN+1] for i in ix]).to(DEVICE)
        return x, y
    
    print(f"\n{'='*60}")
    print(f"🔬 {name}")
    print(f"   Attention: {model.attention_type} | Optimizer: {optimizer_type}")
    print(f"{'='*60}")
    
    best_val_ppl = float('inf')
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for step in range(STEPS_PER_EPOCH):
            x, y = get_batch('train')
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            if use_mobiu_optimizer:
                optimizer.step(loss.item())
            else:
                optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / STEPS_PER_EPOCH
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _ in range(20):
                x, y = get_batch('val')
                logits = model(x)
                val_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        
        val_ppl = np.exp(val_loss / 20)
        best_val_ppl = min(best_val_ppl, val_ppl)
        
        print(f"   Epoch {epoch+1}: Train Loss={avg_train_loss:.3f}, Val PPL={val_ppl:.1f}")
    
    elapsed = time.time() - start_time
    
    if use_mobiu_optimizer:
        optimizer.end()
    
    print(f"   ✅ Best PPL: {best_val_ppl:.2f} | Time: {elapsed:.1f}s")
    
    return {
        'name': name,
        'attention': model.attention_type,
        'optimizer': optimizer_type,
        'best_ppl': best_val_ppl,
        'time': elapsed
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🔥 DOUBLE MOBIU BENCHMARK - CUSTOMER VIEW TEST")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seq: {SEQ_LEN} | d_model: {D_MODEL} | Heads: {NUM_HEADS} | Layers: {NUM_LAYERS}")
    print()
    print("This test shows what a CUSTOMER would experience:")
    print("  1. Pure Adam + Standard Attention (baseline)")
    print("  2. Pure Adam + Mobiu Attention (attention only)")
    print("  3. Adam + Mobiu-Q + Standard Attention (optimizer only)")
    print("  4. Adam + Mobiu-Q + Mobiu Attention (DOUBLE MOBIU!)")
    print("=" * 70)
    
    # Load data
    text = load_shakespeare()
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    print(f"Data: {len(data):,} chars, vocab={vocab_size}")
    
    results = []
    
    # 1. Pure Adam + Standard Attention (baseline)
    result = train_config(
        "Baseline (Pure Adam + Standard)",
        use_mobiu_attention=False,
        use_mobiu_optimizer=False,
        data=data,
        vocab_size=vocab_size
    )
    results.append(result)
    baseline_ppl = result['best_ppl']
    
    # 2. Pure Adam + Mobiu Attention
    if MOBIU_ATTENTION_AVAILABLE:
        result = train_config(
            "Mobiu Attention Only",
            use_mobiu_attention=True,
            use_mobiu_optimizer=False,
            data=data,
            vocab_size=vocab_size
        )
        results.append(result)
    
    # 3. Adam + Mobiu-Q + Standard Attention
    result = train_config(
        "Mobiu Optimizer Only",
        use_mobiu_attention=False,
        use_mobiu_optimizer=True,
        data=data,
        vocab_size=vocab_size
    )
    results.append(result)
    
    # 4. DOUBLE MOBIU
    if MOBIU_ATTENTION_AVAILABLE:
        result = train_config(
            "🔥 DOUBLE MOBIU",
            use_mobiu_attention=True,
            use_mobiu_optimizer=True,
            data=data,
            vocab_size=vocab_size
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Best PPL':<12} {'vs Baseline':<12}")
    print("-" * 70)
    
    for r in results:
        improvement = (baseline_ppl - r['best_ppl']) / baseline_ppl * 100
        imp_str = f"{improvement:+.1f}%" if r['best_ppl'] != baseline_ppl else "-"
        print(f"{r['name']:<35} {r['best_ppl']:<12.2f} {imp_str:<12}")
    
    print("-" * 70)
    
    # Overall winner
    best = min(results, key=lambda x: x['best_ppl'])
    worst = max(results, key=lambda x: x['best_ppl'])
    total_improvement = (worst['best_ppl'] - best['best_ppl']) / worst['best_ppl'] * 100
    
    print(f"\n🏆 WINNER: {best['name']}")
    print(f"   Best PPL: {best['best_ppl']:.2f}")
    print(f"   Total improvement over baseline: {(baseline_ppl - best['best_ppl']) / baseline_ppl * 100:+.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
