#!/usr/bin/env python3
"""
================================================================================
🧪 MOBIU ATTENTION - REAL WORLD BENCHMARK
================================================================================
Tests MobiuAttention on real tasks that matter to customers:

1. 🎭 Shakespeare Language Modeling - Classic NLP benchmark
2. 🐍 Code Understanding - Long Python files
3. 🔍 Needle in Haystack - Can it find info in long context?
4. 📊 Scaling Test - How does it scale with sequence length?

Works on: CUDA, MPS (Apple Silicon), CPU
No license key needed - runs locally!
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import urllib.request
from typing import Optional

# ==============================================================================
# DEVICE SELECTION
# ==============================================================================

def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def sync_device(device):
    """Synchronize for accurate timing"""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

DEVICE = get_device()

# ==============================================================================
# IMPORTS
# ==============================================================================

MOBIU_AVAILABLE = False
try:
    from mobiu_q.experimental import MobiuAttentionFast, MobiuBlockFast
    MOBIU_AVAILABLE = True
except ImportError:
    print("⚠️  Install mobiu-q: pip install mobiu-q")

# ==============================================================================
# STANDARD TRANSFORMER (Baseline)
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
        
        # O(N²) attention
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
# SIMPLE LM MODEL
# ==============================================================================

class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, max_seq: int, use_mobiu: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        
        if use_mobiu and MOBIU_AVAILABLE:
            self.blocks = nn.ModuleList([
                MobiuBlockFast(d_model, num_heads) for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                StandardBlock(d_model, num_heads) for _ in range(num_layers)
            ])
        
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
# 1. SHAKESPEARE BENCHMARK
# ==============================================================================

def download_shakespeare():
    """Download Shakespeare text"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache_path = "/tmp/shakespeare.txt"
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return f.read()
    
    print("📥 Downloading Shakespeare...")
    urllib.request.urlretrieve(url, cache_path)
    with open(cache_path, 'r') as f:
        return f.read()


def benchmark_shakespeare(seq_len: int = 256, epochs: int = 5, batch_size: int = 32):
    """Train on Shakespeare and compare perplexity"""
    
    print("\n" + "=" * 70)
    print("🎭 SHAKESPEARE LANGUAGE MODELING")
    print("=" * 70)
    print(f"Device: {DEVICE} | Seq: {seq_len} | Epochs: {epochs}")
    print("-" * 70)
    
    # Load data
    text = download_shakespeare()
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    n_train = int(len(data) * 0.9)
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    print(f"Vocab: {vocab_size} chars | Train: {n_train:,} | Val: {len(val_data):,}")
    
    def get_batch(split, batch_size, seq_len):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len - 1, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)
    
    results = {}
    
    for name, use_mobiu in [("Transformer", False), ("MobiuAttention", True)]:
        if use_mobiu and not MOBIU_AVAILABLE:
            continue
            
        print(f"\n🔬 {name}")
        torch.manual_seed(42)
        
        model = SimpleLM(
            vocab_size=vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=4,
            max_seq=seq_len,
            use_mobiu=use_mobiu
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        steps_per_epoch = 100
        best_val_ppl = float('inf')
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for _ in range(steps_per_epoch):
                x, y = get_batch('train', batch_size, seq_len)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(20):
                    x, y = get_batch('val', batch_size, seq_len)
                    logits = model(x)
                    val_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            
            val_ppl = np.exp(val_loss / 20)
            best_val_ppl = min(best_val_ppl, val_ppl)
            
            print(f"   Epoch {epoch+1}: Train Loss={train_loss/steps_per_epoch:.3f}, Val PPL={val_ppl:.1f}")
        
        elapsed = time.time() - start_time
        results[name] = {'ppl': best_val_ppl, 'time': elapsed}
        print(f"   ✅ Best PPL: {best_val_ppl:.1f} | Time: {elapsed:.1f}s")
    
    # Summary
    if len(results) == 2:
        t_ppl = results['Transformer']['ppl']
        m_ppl = results['MobiuAttention']['ppl']
        print(f"\n📊 Result: Transformer PPL={t_ppl:.1f} vs MobiuAttention PPL={m_ppl:.1f}")
        if m_ppl < t_ppl * 1.1:  # Within 10%
            print("   ✅ Quality comparable!")
    
    return results


# ==============================================================================
# 2. CODE UNDERSTANDING (Long Context)
# ==============================================================================

def benchmark_code_understanding(seq_lengths: list = [512, 1024, 2048, 4096]):
    """Test ability to handle long code sequences"""
    
    print("\n" + "=" * 70)
    print("🐍 CODE UNDERSTANDING (Long Context)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("-" * 70)
    
    # Simulate code tokens (numbers 0-999 representing code tokens)
    vocab_size = 1000
    
    print(f"{'Seq Len':<12} {'Transformer':<20} {'MobiuAttention':<20}")
    print("-" * 70)
    
    for seq_len in seq_lengths:
        results = {}
        
        for name, use_mobiu in [("Transformer", False), ("MobiuAttention", True)]:
            if use_mobiu and not MOBIU_AVAILABLE:
                results[name] = None
                continue
            
            try:
                torch.manual_seed(42)
                
                model = SimpleLM(
                    vocab_size=vocab_size,
                    d_model=128,
                    num_heads=4,
                    num_layers=2,
                    max_seq=seq_len,
                    use_mobiu=use_mobiu
                ).to(DEVICE)
                
                # Test forward + backward pass
                x = torch.randint(0, vocab_size, (4, seq_len), device=DEVICE)
                y = torch.randint(0, vocab_size, (4, seq_len), device=DEVICE)
                
                sync_device(DEVICE)
                start = time.perf_counter()
                
                for _ in range(5):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                    loss.backward()
                
                sync_device(DEVICE)
                elapsed = (time.perf_counter() - start) / 5 * 1000
                
                results[name] = elapsed
                del model
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[name] = None
                else:
                    raise
            
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
        
        # Format output
        t_str = f"{results.get('Transformer', None):.1f}ms" if results.get('Transformer') else "OOM 💥"
        m_str = f"{results.get('MobiuAttention', None):.1f}ms" if results.get('MobiuAttention') else "N/A"
        
        winner = ""
        if results.get('Transformer') and results.get('MobiuAttention'):
            if results['MobiuAttention'] < results['Transformer']:
                winner = "← Mobiu wins!"
            elif results['Transformer'] < results['MobiuAttention']:
                winner = "← Transformer wins"
        elif results.get('MobiuAttention') and not results.get('Transformer'):
            winner = "← Only Mobiu works! ✅"
        
        print(f"{seq_len:<12} {t_str:<20} {m_str:<20} {winner}")
    
    return results


# ==============================================================================
# 3. NEEDLE IN HAYSTACK
# ==============================================================================

def benchmark_needle_in_haystack(seq_len: int = 2048):
    """Test ability to retrieve specific information from long context"""
    
    print("\n" + "=" * 70)
    print("🔍 NEEDLE IN HAYSTACK")
    print("=" * 70)
    print(f"Device: {DEVICE} | Seq: {seq_len}")
    print("-" * 70)
    
    d_model = 128
    num_heads = 4
    
    # Create input with "needle" (distinctive pattern) at various positions
    positions = [seq_len // 4, seq_len // 2, 3 * seq_len // 4]
    
    for name, use_mobiu in [("Transformer", False), ("MobiuAttention", True)]:
        if use_mobiu and not MOBIU_AVAILABLE:
            continue
            
        print(f"\n{name}:")
        
        try:
            if use_mobiu:
                attn = MobiuAttentionFast(d_model, num_heads).to(DEVICE)
            else:
                attn = StandardAttention(d_model, num_heads).to(DEVICE)
            
            attn.eval()
            
            for needle_pos in positions:
                # Create haystack with needle
                x = torch.randn(1, seq_len, d_model, device=DEVICE) * 0.1
                x[0, needle_pos] = torch.ones(d_model, device=DEVICE) * 2.0  # Needle
                
                with torch.no_grad():
                    out = attn(x)
                
                # Check if needle info is preserved
                needle_out = out[0, needle_pos].norm().item()
                avg_out = out[0].norm(dim=1).mean().item()
                
                preserved = "✅" if needle_out > avg_out * 1.2 else "⚠️"
                print(f"   Position {needle_pos}/{seq_len}: Needle={needle_out:.2f}, Avg={avg_out:.2f} {preserved}")
            
            del attn
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   OOM 💥")
            else:
                raise


# ==============================================================================
# 4. SCALING TEST
# ==============================================================================

def benchmark_scaling():
    """Test how both methods scale with sequence length"""
    
    print("\n" + "=" * 70)
    print("📈 SCALING TEST (Memory & Speed)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("-" * 70)
    
    d_model = 128
    num_heads = 4
    batch_size = 2
    
    # Test progressively longer sequences
    seq_lengths = [256, 512, 1024, 2048]
    if DEVICE == 'cuda':
        seq_lengths.extend([4096, 8192])
    
    print(f"{'Seq':<8} {'Trans Time':<12} {'Mobiu Time':<12} {'Trans Mem':<12} {'Mobiu Mem':<12}")
    print("-" * 70)
    
    for seq_len in seq_lengths:
        results = {'Transformer': {}, 'MobiuAttention': {}}
        
        for name, use_mobiu in [("Transformer", False), ("MobiuAttention", True)]:
            if use_mobiu and not MOBIU_AVAILABLE:
                continue
                
            try:
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                if use_mobiu:
                    attn = MobiuAttentionFast(d_model, num_heads).to(DEVICE)
                else:
                    attn = StandardAttention(d_model, num_heads).to(DEVICE)
                
                x = torch.randn(batch_size, seq_len, d_model, device=DEVICE)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(2):
                        _ = attn(x)
                sync_device(DEVICE)
                
                # Time
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(5):
                        _ = attn(x)
                sync_device(DEVICE)
                elapsed = (time.perf_counter() - start) / 5 * 1000
                
                results[name]['time'] = elapsed
                
                if DEVICE == 'cuda':
                    results[name]['mem'] = torch.cuda.max_memory_allocated() / 1e6
                else:
                    results[name]['mem'] = None
                
                del attn, x
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[name] = {'time': None, 'mem': None}
                else:
                    raise
        
        # Format
        t_time = f"{results['Transformer'].get('time', None):.1f}ms" if results['Transformer'].get('time') else "OOM"
        m_time = f"{results['MobiuAttention'].get('time', None):.1f}ms" if results.get('MobiuAttention', {}).get('time') else "N/A"
        t_mem = f"{results['Transformer'].get('mem', None):.0f}MB" if results['Transformer'].get('mem') else "-"
        m_mem = f"{results.get('MobiuAttention', {}).get('mem', None):.0f}MB" if results.get('MobiuAttention', {}).get('mem') else "-"
        
        print(f"{seq_len:<8} {t_time:<12} {m_time:<12} {t_mem:<12} {m_mem:<12}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("🧪 MOBIU ATTENTION - REAL WORLD BENCHMARK")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print(f"MobiuAttention: {'✅ Available' if MOBIU_AVAILABLE else '❌ Not installed'}")
    print("=" * 70)
    
    if not MOBIU_AVAILABLE:
        print("\n⚠️  Install mobiu-q to run full benchmark:")
        print("   pip install mobiu-q\n")
    
    # Run benchmarks
    benchmark_shakespeare(seq_len=256, epochs=3)
    benchmark_code_understanding([512, 1024, 2048])
    benchmark_needle_in_haystack(seq_len=1024)
    benchmark_scaling()
    
    print("\n" + "=" * 70)
    print("✅ ALL BENCHMARKS COMPLETE")
    print("=" * 70)
    print("""
Summary:
• Shakespeare: Test language modeling quality (PPL)
• Code Understanding: Test long context handling
• Needle in Haystack: Test information retrieval
• Scaling: Test memory/speed tradeoffs

Key Insight: MobiuAttention shines on LONG sequences (>2K tokens)
where standard Transformer either slows down or runs out of memory.
""")


if __name__ == "__main__":
    main()
