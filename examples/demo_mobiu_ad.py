#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                    MOBIU-AD QUICK DEMO                               ║
╚══════════════════════════════════════════════════════════════════════╝

Simple demo showing MobiuAD streaming anomaly detection.

Usage:
    export MOBIU_LICENSE_KEY='your-key-here'
    python demo_mobiu_ad.py
"""

import os
import numpy as np

# Get license key from environment
LICENSE_KEY = os.environ.get("MOBIU_LICENSE_KEY", "")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    MOBIU-AD QUICK DEMO                               ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    if not LICENSE_KEY:
        print("❌ No license key found!")
        print("   Set MOBIU_LICENSE_KEY environment variable:")
        print("   export MOBIU_LICENSE_KEY='your-key-here'")
        return
    
    # Import from mobiu_q
    from mobiu_q import MobiuAD
    
    # Create detector
    detector = MobiuAD(license_key=LICENSE_KEY, method="deep")
    
    print("📊 Generating test data...")
    print("   - Normal values around 50")
    print("   - Anomaly spike at position 15")
    print("   - Pattern change at position 25+")
    print()
    
    # Generate test data
    np.random.seed(42)
    data = []
    
    # Normal (0-14)
    for i in range(15):
        data.append(50 + np.random.normal(0, 3))
    
    # Spike (15)
    data.append(85)
    
    # Normal (16-24)
    for i in range(9):
        data.append(50 + np.random.normal(0, 3))
    
    # Pattern change - trending up (25-34)
    for i in range(10):
        data.append(50 + i * 1.5 + np.random.normal(0, 1))
    
    # Detect anomalies
    print("🔍 Running detection...")
    print("-" * 50)
    
    anomalies = []
    for i, value in enumerate(data):
        result = detector.detect(value)
        
        status = "⚠️ ANOMALY" if result.is_anomaly else "  normal"
        print(f"  [{i:2d}] value={value:6.2f}  Δ†={result.delta_dagger:7.4f}  {status}")
        
        if result.is_anomaly:
            anomalies.append(i)
    
    print("-" * 50)
    print(f"\n📋 Summary:")
    print(f"   Total points: {len(data)}")
    print(f"   Anomalies detected: {len(anomalies)}")
    if anomalies:
        print(f"   Anomaly positions: {anomalies}")
    
    # Batch detection
    print("\n" + "="*50)
    print("📦 Batch Detection")
    print("="*50)
    
    detector2 = MobiuAD(license_key=LICENSE_KEY, method="deep")
    batch_results = detector2.detect_batch(np.array(data))
    
    print(f"   Total anomalies: {batch_results.total_anomalies}")
    print(f"   Anomaly indices: {batch_results.anomaly_indices}")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
