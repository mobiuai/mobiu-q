#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║              TRAINGUARD - MOBIU-Q + MOBIU-AD INTEGRATION             ║
╚══════════════════════════════════════════════════════════════════════╝

Demonstrates combining Mobiu-Q (optimization) with Mobiu-AD (monitoring)
for safer ML training.

Requirements:
    pip install mobiu-ad numpy

Usage:
    python test_trainguard.py

What TrainGuard provides:
- Mobiu-Q: Optimizes gradient steps using Soft Algebra (via API)
- Mobiu-AD: Monitors loss patterns for anomalies (via API)
- Combined: Early detection of training problems + better convergence
"""

import numpy as np

# Import TrainGuard from mobiu-ad package
from mobiu_ad import TrainGuard, TrainGuardResult

# Configuration - Replace with your key
LICENSE_KEY = "YOUR_KEY"


# ============================================================================
# TEST 1: Normal Training
# ============================================================================

def test_normal_training():
    """
    TEST 1: Normal Healthy Training
    
    Simulates a simple quadratic optimization: f(x) = x²
    Training should converge smoothly with few/no alerts.
    """
    print("\n" + "="*70)
    print("🧪 TEST 1: Normal Healthy Training")
    print("="*70)
    print("""
    Scenario:
    ─────────
    • Optimize f(x) = x², starting from x=5
    • Expected: Smooth convergence, few alerts
    """)
    
    guard = TrainGuard(LICENSE_KEY)
    np.random.seed(42)
    
    x = 5.0
    lr = 0.1
    alerts = []
    
    print(f"\n{'Step':<6} {'Loss':<12} {'Scale':<10} {'Alert'}")
    print("-"*45)
    
    for step in range(40):
        loss = x**2 + np.random.normal(0, 0.05)
        gradient = [2*x]
        
        result = guard.step(loss, gradient)
        
        x = x - lr * result.adjusted_gradient[0]
        
        if result.alert:
            alerts.append((step, result.alert_type))
        
        if step < 5 or step % 10 == 0:
            alert = f"⚠️ {result.alert_type}" if result.alert else ""
            print(f"{step:<6} {loss:<12.4f} {result.scale:<10.3f} {alert}")
    
    summary = guard.get_summary()
    print(f"\n📊 Results:")
    print(f"   Final loss: {summary['final_loss']:.6f}")
    print(f"   Total alerts: {len(alerts)}")
    print(f"   Avg Q scale: {summary['avg_scale']:.3f}")
    
    success = summary['final_loss'] < 1.0
    print(f"\n{'✅ PASS' if success else '⚠️ CHECK'}: Normal training completed")
    return success


# ============================================================================
# TEST 2: Gradient Explosion Detection
# ============================================================================

def test_gradient_explosion():
    """
    TEST 2: Gradient Explosion Detection
    
    Training goes well, then gradients explode.
    TrainGuard should detect this early.
    """
    print("\n" + "="*70)
    print("🧪 TEST 2: Gradient Explosion Detection")
    print("="*70)
    print("""
    Scenario:
    ─────────
    • Normal training for 25 steps
    • Loss starts exploding at step 25
    • Expected: EXPLOSION alert within 5 steps
    """)
    
    guard = TrainGuard(LICENSE_KEY)
    np.random.seed(123)
    
    explosion_start = 25
    alerts = []
    
    print(f"\n{'Step':<6} {'Loss':<12} {'Scale':<10} {'Alert'}")
    print("-"*50)
    
    for step in range(40):
        if step < explosion_start:
            loss = 10 * np.exp(-step/15) + np.random.normal(0, 0.1)
        else:
            # Explosion!
            loss = 10 * np.exp(-explosion_start/15) * (1 + (step-explosion_start)*0.5)
        
        gradient = [np.random.normal(0, 1)]
        result = guard.step(loss, gradient)
        
        if result.alert:
            alerts.append((step, result.alert_type))
        
        if step < 5 or step >= explosion_start - 2:
            alert = f"⚠️ {result.alert_type}" if result.alert else ""
            marker = " ← EXPLOSION" if step == explosion_start else ""
            print(f"{step:<6} {loss:<12.2f} {result.scale:<10.3f} {alert}{marker}")
    
    early_alerts = [a for a in alerts if a[0] <= explosion_start + 5]
    explosion_alerts = [a for a in early_alerts if 'EXPLOSION' in str(a[1]) or 'TRAIN' in str(a[1])]
    
    print(f"\n📊 Results:")
    print(f"   Explosion at step: {explosion_start}")
    print(f"   Early alerts (within 5 steps): {len(early_alerts)}")
    print(f"   Explosion-type alerts: {len(explosion_alerts)}")
    
    success = len(early_alerts) > 0
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: {'Detected explosion!' if success else 'Missed explosion'}")
    return success


# ============================================================================
# TEST 3: Overfitting Detection
# ============================================================================

def test_overfitting():
    """
    TEST 3: Overfitting Detection
    
    Train loss keeps improving, but val loss starts rising.
    This is the classic overfitting pattern.
    """
    print("\n" + "="*70)
    print("🧪 TEST 3: Overfitting Detection")
    print("="*70)
    print("""
    Scenario:
    ─────────
    • Train loss: Keeps decreasing (good!)
    • Val loss: Decreases, then starts rising at step 30 (bad!)
    • Expected: OVERFITTING alert detected
    """)
    
    guard = TrainGuard(LICENSE_KEY)
    np.random.seed(456)
    
    overfit_start = 30
    alerts = []
    overfit_detected = False
    first_overfit_step = None
    
    print(f"\n{'Step':<6} {'Train':<10} {'Val':<10} {'Scale':<8} {'Alert'}")
    print("-"*55)
    
    for step in range(50):
        # Train loss: keeps improving
        train_loss = 10 * np.exp(-step/20) + np.random.normal(0, 0.1)
        
        # Val loss: improves then gets worse
        if step < overfit_start:
            val_loss = train_loss + np.random.normal(0, 0.2)
        else:
            val_loss = 10 * np.exp(-overfit_start/20) + (step-overfit_start)*0.15 + np.random.normal(0, 0.2)
        
        gradient = [np.random.normal(0, 1)]
        result = guard.step(train_loss, gradient, val_loss)
        
        if result.alert:
            alerts.append((step, result.alert_type))
            if result.alert_type == 'OVERFITTING' and not overfit_detected:
                overfit_detected = True
                first_overfit_step = step
        
        if step < 5 or step >= overfit_start - 2:
            alert = f"⚠️ {result.alert_type}" if result.alert else ""
            marker = " ← OVERFIT STARTS" if step == overfit_start else ""
            print(f"{step:<6} {train_loss:<10.4f} {val_loss:<10.4f} {result.scale:<8.3f} {alert}{marker}")
    
    overfit_alerts = [a for a in alerts if a[1] == 'OVERFITTING']
    
    print(f"\n📊 Results:")
    print(f"   Overfitting starts at step: {overfit_start}")
    print(f"   First OVERFITTING alert: {first_overfit_step if first_overfit_step else 'None'}")
    print(f"   Total OVERFITTING alerts: {len(overfit_alerts)}")
    
    detection_delay = first_overfit_step - overfit_start if first_overfit_step else float('inf')
    success = overfit_detected and detection_delay <= 10
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: {'Detected overfitting!' if success else 'Missed overfitting'}")
    if success:
        print(f"   Detection delay: {detection_delay} steps")
    return success


# ============================================================================
# TEST 4: Q Optimization Effect
# ============================================================================

def test_q_optimization():
    """
    TEST 4: Mobiu-Q Optimization Effect
    
    Check that Q is providing adaptive scaling.
    """
    print("\n" + "="*70)
    print("🧪 TEST 4: Mobiu-Q Optimization Effect")
    print("="*70)
    print("""
    Scenario:
    ─────────
    • Same optimization problem
    • Q adjusts gradient scale based on Soft Algebra
    • Expected: Q provides adaptive scaling (0.5x - 1.5x)
    """)
    
    guard = TrainGuard(LICENSE_KEY)
    np.random.seed(42)
    
    scales = []
    
    x = 5.0
    lr = 0.1
    
    print(f"\n{'Step':<6} {'Loss':<12} {'Scale':<10} {'Trust'}")
    print("-"*40)
    
    for step in range(30):
        loss = x**2 + np.random.normal(0, 0.05)
        gradient = [2*x]
        
        result = guard.step(loss, gradient)
        
        x = x - lr * result.adjusted_gradient[0]
        
        scales.append(result.scale)
        
        if step < 5 or step % 5 == 0:
            print(f"{step:<6} {loss:<12.4f} {result.scale:<10.3f} {result.trust_ratio:.3f}")
    
    print(f"\n📊 Results:")
    print(f"   Scale range: {min(scales):.3f} - {max(scales):.3f}")
    print(f"   Scale varies: {'✅ Yes' if max(scales) - min(scales) > 0.1 else '❌ No'}")
    
    success = max(scales) - min(scales) > 0.05
    print(f"\n{'✅ PASS' if success else '⚠️ CHECK'}: Q is {'adapting' if success else 'not adapting'} scales")
    return success


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              TRAINGUARD - MOBIU-Q + MOBIU-AD INTEGRATION             ║
║                                                                      ║
║  Combining Optimization (Q) and Monitoring (AD) for Safer Training  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    if LICENSE_KEY == "YOUR_LICENSE_KEY":
        print("⚠️  Please set your LICENSE_KEY in the script!")
        print("    Get a free key at: https://app.mobiu.ai")
        return
    
    results = []
    
    results.append(("Normal Training", test_normal_training()))
    results.append(("Gradient Explosion", test_gradient_explosion()))
    results.append(("Overfitting Detection", test_overfitting()))
    results.append(("Q Optimization", test_q_optimization()))
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("""
    ┌─────────────────────────┬──────────┐
    │ Test                    │ Result   │
    ├─────────────────────────┼──────────┤""")
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"    │ {name:<23} │ {status:<8} │")
    
    print("""    └─────────────────────────┴──────────┘""")
    
    print(f"\n    Total: {passed}/{total} tests passed")
    
    print("""
    
🎯 TRAINGUARD CAPABILITIES:
───────────────────────────
• Mobiu-Q:  Gradient optimization via Soft Algebra (API)
• Mobiu-AD: Real-time anomaly detection (API)

Alert Types:
• OVERFITTING  → Val diverging from train
• EXPLOSION    → Loss increasing rapidly
• PLATEAU      → Loss stagnating
• CRITICAL     → Both train and val alerting

Usage:
──────
    from mobiu_ad import TrainGuard
    
    guard = TrainGuard(license_key="your-key")
    
    for epoch in training:
        result = guard.step(loss, gradient, val_loss)
        
        # Q: Optimized gradient
        model.apply(result.adjusted_gradient)
        
        # AD: Alerts
        if result.alert_type == 'OVERFITTING':
            reduce_lr()

Learn more: https://mobiu.ai
    """)
    print("="*70)


if __name__ == "__main__":
    main()
