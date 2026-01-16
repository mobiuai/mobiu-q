#!/usr/bin/env python3
"""
================================================================================
🔥 DOUBLE MOBIU BENCHMARK
================================================================================
The Ultimate Test: MobiuAttention + MobiuOptimizer Together

Configurations tested:
1. Standard Attention + Adam (Baseline)
2. Mobiu Attention + Adam (Attention only)
3. Standard Attention + MobiuOptimizer (Optimizer only)
4. Mobiu Attention + MobiuOptimizer (DOUBLE MOBIU!)

Task: Shakespeare Language Modeling
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import urllib.request

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Get from environment or use default
LICENSE_KEY = os.environ.get('MOBIU_LICENSE_KEY', 'YOUR_KEY')

# Hyperparameters
SEQ_LEN = 256
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4
BATCH_SIZE = 32
EPOCHS = 5
STEPS_PER_EPOCH = 100
LR = 3e-4
METHODS = ["adaptive", "deep"]  # Test both methods

# ==============================================================================
# DEVICE SELECTION
# ==============================================================================

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

DEVICE = get_device()

# ==============================================================================
# IMPORTS
# ==============================================================================

MOBIU_ATTENTION_AVAILABLE = False
MOBIU_OPTIMIZER_AVAILABLE = False

try:
    from mobiu_q.experimental import MobiuBlockFast
    MOBIU_ATTENTION_AVAILABLE = True
except ImportError:
    print("⚠️  MobiuAttention not available")

try:
    from mobiu_q import MobiuOptimizer
    MOBIU_OPTIMIZER_AVAILABLE = True
except ImportError:
    print("⚠️  MobiuOptimizer not available")


# ==============================================================================
# STANDARD TRANSFORMER BLOCK
# ==============================================================================

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


# ==============================================================================
# LANGUAGE MODEL
# ==============================================================================

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
            self.attention_type = "Mobiu"
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


# ==============================================================================
# DATA
# ==============================================================================

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


# ==============================================================================
# TRAINING
# ==============================================================================

def train_config(name: str, use_mobiu_attention: bool, use_mobiu_optimizer: bool,
                 data: torch.Tensor, vocab_size: int, seed: int = 42, method: str = "adaptive"):
    """Train a single configuration and return results."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = SimpleLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq=SEQ_LEN,
        use_mobiu_attention=use_mobiu_attention
    ).to(DEVICE)
    
    # Create optimizer
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    if use_mobiu_optimizer and MOBIU_OPTIMIZER_AVAILABLE:
        optimizer = MobiuOptimizer(
            base_optimizer,
            license_key=LICENSE_KEY,
            method=method,
            use_soft_algebra=True,
            verbose=False
        )
        optimizer_type = f"MobiuOpt({method})"
    else:
        optimizer = base_optimizer
        optimizer_type = "Adam"
    
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
    
    history = {'train_loss': [], 'val_ppl': []}
    best_val_ppl = float('inf')
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        for step in range(STEPS_PER_EPOCH):
            x, y = get_batch('train')
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # MobiuOptimizer needs the loss value
            if use_mobiu_optimizer and MOBIU_OPTIMIZER_AVAILABLE:
                optimizer.step(loss.item())
            else:
                optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / STEPS_PER_EPOCH
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _ in range(20):
                x, y = get_batch('val')
                logits = model(x)
                val_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        
        val_ppl = np.exp(val_loss / 20)
        history['val_ppl'].append(val_ppl)
        best_val_ppl = min(best_val_ppl, val_ppl)
        
        print(f"   Epoch {epoch+1}: Train Loss={avg_train_loss:.3f}, Val PPL={val_ppl:.1f}")
    
    elapsed = time.time() - start_time
    
    # Cleanup MobiuOptimizer
    if use_mobiu_optimizer and MOBIU_OPTIMIZER_AVAILABLE and hasattr(optimizer, 'end'):
        optimizer.end()
    
    print(f"   ✅ Best PPL: {best_val_ppl:.2f} | Time: {elapsed:.1f}s")
    
    return {
        'name': name,
        'attention': model.attention_type,
        'optimizer': optimizer_type,
        'best_ppl': best_val_ppl,
        'final_ppl': history['val_ppl'][-1],
        'time': elapsed,
        'history': history
    }


# ==============================================================================
# MAIN BENCHMARK
# ==============================================================================

def run_double_mobiu_benchmark():
    print("=" * 70)
    print("🔥 DOUBLE MOBIU - FAIR A/B TEST (Soft Algebra ON vs OFF)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"MobiuAttention: {'✅' if MOBIU_ATTENTION_AVAILABLE else '❌'}")
    print(f"MobiuOptimizer: {'✅' if MOBIU_OPTIMIZER_AVAILABLE else '❌'}")
    print(f"Config: seq={SEQ_LEN}, d={D_MODEL}, heads={NUM_HEADS}, layers={NUM_LAYERS}")
    print(f"Training: {EPOCHS} epochs × {STEPS_PER_EPOCH} steps, LR={LR}")
    print(f"Method: adaptive")
    print("=" * 70)
    
    # Load data
    text = load_shakespeare()
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    print(f"Data: {len(data):,} chars, vocab={vocab_size}")
    
    results = []
    
    # 1. Local Adam (no API) - for reference
    print(f"\n{'='*60}")
    print("📋 PART 1: LOCAL ADAM (no API)")
    print(f"{'='*60}")
    result = train_config_fair("Local Adam (no API)", False, False, False, data, vocab_size)
    results.append(result)
    baseline_ppl = result['best_ppl']
    
    # 2. MobiuOptimizer with use_soft_algebra=False (API baseline)
    print(f"\n{'='*60}")
    print("📋 PART 2: API BASELINE (use_soft_algebra=False)")
    print(f"{'='*60}")
    if MOBIU_OPTIMIZER_AVAILABLE:
        result = train_config_fair("API Adam (SA=off)", False, True, False, data, vocab_size)
        results.append(result)
        api_baseline_ppl = result['best_ppl']
    
    # 3. MobiuOptimizer with use_soft_algebra=True
    print(f"\n{'='*60}")
    print("📋 PART 3: MOBIU OPTIMIZER (use_soft_algebra=True)")
    print(f"{'='*60}")
    if MOBIU_OPTIMIZER_AVAILABLE:
        result = train_config_fair("Mobiu (SA=on)", False, True, True, data, vocab_size)
        results.append(result)
    
    # 4. MobiuAttention + API Adam (SA=off)
    print(f"\n{'='*60}")
    print("📋 PART 4: MOBIU ATTENTION + API (SA=off)")
    print(f"{'='*60}")
    if MOBIU_ATTENTION_AVAILABLE and MOBIU_OPTIMIZER_AVAILABLE:
        result = train_config_fair("MobiuAttn + API (SA=off)", True, True, False, data, vocab_size)
        results.append(result)
    
    # 5. Double Mobiu (MobiuAttention + SA=on)
    print(f"\n{'='*60}")
    print("📋 PART 5: DOUBLE MOBIU (Attention + SA=on)")
    print(f"{'='*60}")
    if MOBIU_ATTENTION_AVAILABLE and MOBIU_OPTIMIZER_AVAILABLE:
        result = train_config_fair("DOUBLE MOBIU (SA=on)", True, True, True, data, vocab_size)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Best PPL':<12} {'vs Local':<12} {'vs API':<12}")
    print("-" * 70)
    
    for r in results:
        vs_local = (baseline_ppl - r['best_ppl']) / baseline_ppl * 100 if r['best_ppl'] != baseline_ppl else 0
        vs_api = ""
        if 'api_baseline_ppl' in dir() and api_baseline_ppl and r['best_ppl'] != api_baseline_ppl:
            vs_api = f"{(api_baseline_ppl - r['best_ppl']) / api_baseline_ppl * 100:+.1f}%"
        
        vs_local_str = f"{vs_local:+.1f}%" if vs_local != 0 else "-"
        print(f"{r['name']:<35} {r['best_ppl']:<12.2f} {vs_local_str:<12} {vs_api:<12}")
    
    print("-" * 70)
    
    # Fair comparison: SA=on vs SA=off
    print("\n" + "=" * 70)
    print("🔬 FAIR A/B TEST: Soft Algebra ON vs OFF")
    print("=" * 70)
    
    sa_off = [r for r in results if 'SA=off' in r['name'] and 'Attn' not in r['name']]
    sa_on = [r for r in results if 'SA=on' in r['name'] and 'DOUBLE' not in r['name']]
    
    if sa_off and sa_on:
        off_ppl = sa_off[0]['best_ppl']
        on_ppl = sa_on[0]['best_ppl']
        improvement = (off_ppl - on_ppl) / off_ppl * 100
        
        print(f"Standard Attention:")
        print(f"  API Adam (SA=off): {off_ppl:.2f}")
        print(f"  Mobiu (SA=on):     {on_ppl:.2f}")
        print(f"  Improvement:       {improvement:+.1f}%")
        
        if improvement > 5:
            print(f"\n✅ SOFT ALGEBRA PROVIDES REAL VALUE!")
        elif improvement > 0:
            print(f"\n🤝 Small improvement from Soft Algebra")
        else:
            print(f"\n⚠️ No improvement from Soft Algebra")
    
    # Fair comparison for MobiuAttention
    attn_off = [r for r in results if 'MobiuAttn + API' in r['name']]
    attn_on = [r for r in results if 'DOUBLE MOBIU' in r['name']]
    
    if attn_off and attn_on:
        off_ppl = attn_off[0]['best_ppl']
        on_ppl = attn_on[0]['best_ppl']
        improvement = (off_ppl - on_ppl) / off_ppl * 100
        
        print(f"\nMobiu Attention:")
        print(f"  MobiuAttn + API (SA=off): {off_ppl:.2f}")
        print(f"  DOUBLE MOBIU (SA=on):     {on_ppl:.2f}")
        print(f"  Improvement:              {improvement:+.1f}%")
    
    # Overall winner
    best = min(results, key=lambda x: x['best_ppl'])
    print(f"\n🏆 OVERALL WINNER: {best['name']}")
    print(f"   Best PPL: {best['best_ppl']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 70)
    
    return results


def train_config_fair(name: str, use_mobiu_attention: bool, use_mobiu_optimizer: bool,
                      use_soft_algebra: bool, data: torch.Tensor, vocab_size: int, 
                      seed: int = 42, method: str = "adaptive"):
    """Train a single configuration with explicit soft_algebra control."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = SimpleLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq=SEQ_LEN,
        use_mobiu_attention=use_mobiu_attention
    ).to(DEVICE)
    
    # Create optimizer
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    if use_mobiu_optimizer and MOBIU_OPTIMIZER_AVAILABLE:
        optimizer = MobiuOptimizer(
            base_optimizer,
            license_key=LICENSE_KEY,
            method=method,
            use_soft_algebra=use_soft_algebra,
            verbose=False
        )
        optimizer_type = f"MobiuOpt(SA={'on' if use_soft_algebra else 'off'})"
    else:
        optimizer = base_optimizer
        optimizer_type = "Adam (local)"
    
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
    
    history = {'train_loss': [], 'val_ppl': []}
    best_val_ppl = float('inf')
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        for step in range(STEPS_PER_EPOCH):
            x, y = get_batch('train')
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # MobiuOptimizer needs the loss value
            if use_mobiu_optimizer and MOBIU_OPTIMIZER_AVAILABLE:
                optimizer.step(loss.item())
            else:
                optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / STEPS_PER_EPOCH
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _ in range(20):
                x, y = get_batch('val')
                logits = model(x)
                val_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        
        val_ppl = np.exp(val_loss / 20)
        history['val_ppl'].append(val_ppl)
        best_val_ppl = min(best_val_ppl, val_ppl)
        
        print(f"   Epoch {epoch+1}: Train Loss={avg_train_loss:.3f}, Val PPL={val_ppl:.1f}")
    
    elapsed = time.time() - start_time
    
    # Cleanup MobiuOptimizer
    if use_mobiu_optimizer and MOBIU_OPTIMIZER_AVAILABLE and hasattr(optimizer, 'end'):
        optimizer.end()
    
    print(f"   ✅ Best PPL: {best_val_ppl:.2f} | Time: {elapsed:.1f}s")
    
    return {
        'name': name,
        'attention': model.attention_type,
        'optimizer': optimizer_type,
        'best_ppl': best_val_ppl,
        'final_ppl': history['val_ppl'][-1],
        'time': elapsed,
        'history': history
    }


if __name__ == "__main__":
    run_double_mobiu_benchmark()
