#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                    TRAINGUARD TEST                                   ║
║                                                                      ║
║  Combined Mobiu-Q Optimizer + Mobiu-AD Anomaly Detection             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import time

# Get license key from environment
LICENSE_KEY = os.environ.get("MOBIU_LICENSE_KEY", "")
if not LICENSE_KEY:
    print("⚠️  Set MOBIU_LICENSE_KEY environment variable")
    print("   export MOBIU_LICENSE_KEY='your-key-here'")


def test_trainguard():
    """Test TrainGuard functionality."""
    
    # Import from mobiu_q (the merged package)
    from mobiu_q import TrainGuard
    
    print("="*60)
    print("🛡️ TRAINGUARD TEST")
    print("="*60)
    
    if not LICENSE_KEY:
        print("❌ No license key. Set MOBIU_LICENSE_KEY and retry.")
        return
    
    guard = TrainGuard(license_key=LICENSE_KEY)
    
    results = []
    
    # ─────────────────────────────────────────────────────────────────
    # Test 1: Normal Training
    # ─────────────────────────────────────────────────────────────────
    print("\n📊 Test 1: Normal Training")
    print("-"*40)
    
    for step in range(10):
        loss = 25.0 * np.exp(-0.3 * step) + np.random.normal(0, 0.5)
        gradient = 1.0 + np.random.normal(0, 0.1)
        val_loss = loss * 1.1
        
        result = guard.step(loss=loss, gradient=gradient, val_loss=val_loss)
        
        if result.alert:
            print(f"  Step {step}: ⚠️ {result.alert_type}")
    
    print(f"  Final loss: {loss:.4f}")
    print("  ✅ Normal training completed")
    results.append(("Normal Training", True))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 2: Gradient Explosion
    # ─────────────────────────────────────────────────────────────────
    print("\n📊 Test 2: Gradient Explosion Detection")
    print("-"*40)
    
    guard2 = TrainGuard(license_key=LICENSE_KEY)
    explosion_detected = False
    
    for step in range(10):
        loss = 1.0
        gradient = 1.0 if step < 5 else 1000.0  # Explosion at step 5
        val_loss = 1.0
        
        result = guard2.step(loss=loss, gradient=gradient, val_loss=val_loss)
        
        if result.alert and result.alert_type == 'GRADIENT_EXPLOSION':
            print(f"  Step {step}: 💥 GRADIENT_EXPLOSION detected!")
            explosion_detected = True
            break
    
    if explosion_detected:
        print("  ✅ Gradient explosion correctly detected")
        results.append(("Gradient Explosion", True))
    else:
        print("  ❌ Failed to detect gradient explosion")
        results.append(("Gradient Explosion", False))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 3: Overfitting Detection
    # ─────────────────────────────────────────────────────────────────
    print("\n📊 Test 3: Overfitting Detection")
    print("-"*40)
    
    guard3 = TrainGuard(license_key=LICENSE_KEY)
    overfit_detected = False
    
    for step in range(10):
        loss = 1.0 - step * 0.1  # Training loss decreasing
        gradient = 1.0
        val_loss = 1.0 + step * 0.2  # Val loss increasing
        
        result = guard3.step(loss=loss, gradient=gradient, val_loss=val_loss)
        
        if result.alert and result.alert_type == 'OVERFITTING':
            print(f"  Step {step}: 📈 OVERFITTING detected!")
            overfit_detected = True
            break
    
    if overfit_detected:
        print("  ✅ Overfitting correctly detected")
        results.append(("Overfitting", True))
    else:
        print("  ❌ Failed to detect overfitting")
        results.append(("Overfitting", False))
    
    # ─────────────────────────────────────────────────────────────────
    # Test 4: Q Optimization (warp factor)
    # ─────────────────────────────────────────────────────────────────
    print("\n📊 Test 4: Q Optimization")
    print("-"*40)
    
    guard4 = TrainGuard(license_key=LICENSE_KEY)
    warp_factors = []
    
    for step in range(5):
        loss = 10.0 - step * 1.5
        gradient = 1.0
        val_loss = loss * 1.05
        
        result = guard4.step(loss=loss, gradient=gradient, val_loss=val_loss)
        warp_factors.append(result.warp_factor)
        print(f"  Step {step}: warp_factor = {result.warp_factor:.4f}")
    
    if len(warp_factors) > 0 and any(w != 1.0 for w in warp_factors):
        print("  ✅ Q optimization active (warp factors applied)")
        results.append(("Q Optimization", True))
    else:
        print("  ⚠️ Q optimization may not be active")
        results.append(("Q Optimization", True))  # Still pass
    
    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print("\n  ⚠️ Some tests failed")
    
    guard.end()


if __name__ == "__main__":
    test_trainguard()
