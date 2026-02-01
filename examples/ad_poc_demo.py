#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                    MOBIU-AD POC DEMONSTRATION                        ║
╚══════════════════════════════════════════════════════════════════════╝

This script demonstrates Mobiu-AD's unique capabilities:
1. Pattern Change Detection - detects behavioral shifts
2. Precursor Detection - early warning before anomalies
3. Comparison with standard methods (Z-Score)

Run: python poc_demo.py

Requires: pip install mobiu-ad numpy requests
"""

import numpy as np
import time

# Try to import Mobiu-AD
try:
    from mobiu_ad import MobiuAD
    USE_LOCAL = False
except ImportError:
    USE_LOCAL = True
    print("⚠️ mobiu-ad not installed, using direct API calls")

import requests

# Configuration
API_URL = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_ad"
LICENSE_KEY = "YOUR_KEY"  # Replace with your key


def mobiu_detect(session_id: str, values: list, method: str = "deep") -> list:
    """Run Mobiu-AD detection via API."""
    results = []
    for value in values:
        r = requests.post(API_URL, json={
            "license_key": LICENSE_KEY,
            "session_id": session_id,
            "value": float(value),
            "method": method
        })
        result = r.json()
        results.append({
            'is_anomaly': result.get('is_anomaly', False),
            'delta_dagger': result.get('delta_dagger', 0),
            'at': result.get('at', 0),
            'bt': result.get('bt', 0),
        })
    return results


def zscore_detect(values: list, window: int = 20, threshold: float = 3.0) -> list:
    """Standard Z-Score detection for comparison."""
    results = []
    for i in range(len(values)):
        if i < window:
            results.append({'is_anomaly': False, 'score': 0})
            continue
        recent = values[i-window:i]
        mean = np.mean(recent)
        std = np.std(recent) + 1e-9
        zscore = abs(values[i] - mean) / std
        results.append({'is_anomaly': zscore > threshold, 'score': zscore})
    return results


def test_pattern_change():
    """
    TEST 1: Pattern Change Detection
    
    System randomly fluctuates, then starts trending.
    Values look "normal" but the pattern changed.
    """
    print("\n" + "="*70)
    print("🧪 TEST 1: Pattern Change Detection")
    print("="*70)
    print("""
    Scenario:
    ─────────
    • Index 0-59:  Random fluctuation around 50 (std=3)
    • Index 60-99: Consistent downward trend (-0.25 per step)
    
    The values LOOK normal, but the BEHAVIOR changed!
    """)
    
    np.random.seed(42)
    random_phase = list(np.random.normal(50, 3, 60))
    trending = [50]
    for i in range(39):
        trending.append(trending[-1] - 0.25 + np.random.normal(0, 0.3))
    data = random_phase + trending
    
    # Run detections
    session_id = f"demo_pattern_{int(time.time())}"
    mobiu_results = mobiu_detect(session_id, data)
    zscore_results = zscore_detect(data)
    
    # Count detections in trend phase
    mobiu_count = sum(1 for i, r in enumerate(mobiu_results) 
                      if r['is_anomaly'] and 60 <= i < 100)
    zscore_count = sum(1 for i, r in enumerate(zscore_results) 
                       if r['is_anomaly'] and 60 <= i < 100)
    
    print(f"    Results (detections in trend phase 60-100):")
    print(f"    ┌────────────┬────────────┐")
    print(f"    │ Z-Score    │ {zscore_count:>10} │")
    print(f"    │ Mobiu-AD   │ {mobiu_count:>10} │")
    print(f"    └────────────┴────────────┘")
    
    if mobiu_count > zscore_count:
        print(f"\n    ✅ MOBIU-AD WINS: Detected {mobiu_count} vs {zscore_count}")
    
    return mobiu_count, zscore_count


def test_precursor_detection():
    """
    TEST 2: Precursor Detection (Early Warning)
    
    Before a spike, there's a buildup period with increasing variance.
    Mobiu-AD detects this "tension" building up.
    """
    print("\n" + "="*70)
    print("🧪 TEST 2: Precursor Detection (Early Warning)")
    print("="*70)
    print("""
    Scenario:
    ─────────
    • Index 0-49:   Normal operation (std=2)
    • Index 50-69:  Buildup - variance increasing (std: 2→10)
    • Index 70:     SPIKE! (value=90)
    
    Can we detect the buildup BEFORE the spike?
    """)
    
    np.random.seed(123)
    normal = list(np.random.normal(50, 2, 50))
    buildup = [50 + np.random.normal(0, 2 + i*0.5) for i in range(20)]
    spike = [90]
    after = list(np.random.normal(50, 2, 30))
    data = normal + buildup + spike + after
    
    # Run detections
    session_id = f"demo_precursor_{int(time.time())}"
    mobiu_results = mobiu_detect(session_id, data)
    zscore_results = zscore_detect(data)
    
    # Count EARLY warnings (before spike at index 70)
    mobiu_early = sum(1 for i, r in enumerate(mobiu_results) 
                      if r['is_anomaly'] and 50 <= i < 70)
    zscore_early = sum(1 for i, r in enumerate(zscore_results) 
                       if r['is_anomaly'] and 50 <= i < 70)
    
    print(f"    Results (early warnings BEFORE spike):")
    print(f"    ┌────────────┬────────────┐")
    print(f"    │ Z-Score    │ {zscore_early:>10} │")
    print(f"    │ Mobiu-AD   │ {mobiu_early:>10} │")
    print(f"    └────────────┴────────────┘")
    
    if mobiu_early > zscore_early:
        print(f"\n    ✅ MOBIU-AD WINS: {mobiu_early} early warnings vs {zscore_early}")
        print(f"       That's {mobiu_early/max(zscore_early,1):.1f}x more advance notice!")
    
    return mobiu_early, zscore_early


def test_subtle_drift():
    """
    TEST 3: Subtle Drift Detection
    
    System drifts slowly upward. Each value looks normal,
    but the trend is anomalous.
    """
    print("\n" + "="*70)
    print("🧪 TEST 3: Subtle Drift Detection")
    print("="*70)
    print("""
    Scenario:
    ─────────
    • Index 0-39:   Stable at 50 (std=2)
    • Index 40-99:  Slow drift upward (+0.15 per step)
    
    Each individual value looks normal. The TREND is anomalous.
    """)
    
    np.random.seed(789)
    stable = list(np.random.normal(50, 2, 40))
    drift = [50 + i*0.15 + np.random.normal(0, 2) for i in range(60)]
    data = stable + drift
    
    # Run detections
    session_id = f"demo_drift_{int(time.time())}"
    mobiu_results = mobiu_detect(session_id, data)
    zscore_results = zscore_detect(data)
    
    # Count detections during drift
    mobiu_count = sum(1 for i, r in enumerate(mobiu_results) 
                      if r['is_anomaly'] and 40 <= i < 100)
    zscore_count = sum(1 for i, r in enumerate(zscore_results) 
                       if r['is_anomaly'] and 40 <= i < 100)
    
    print(f"    Results (detections during drift):")
    print(f"    ┌────────────┬────────────┐")
    print(f"    │ Z-Score    │ {zscore_count:>10} │")
    print(f"    │ Mobiu-AD   │ {mobiu_count:>10} │")
    print(f"    └────────────┴────────────┘")
    
    if mobiu_count > zscore_count:
        print(f"\n    ✅ MOBIU-AD WINS: Detected drift that Z-Score missed!")
    
    return mobiu_count, zscore_count


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    MOBIU-AD POC DEMONSTRATION                        ║
║                                                                      ║
║  "Detect BEHAVIOR changes, not just outlier VALUES"                 ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    if LICENSE_KEY == "YOUR_LICENSE_KEY":
        print("⚠️  Please set your LICENSE_KEY in the script!")
        print("    Get a free key at: https://mobiu.ai")
        return
    
    results = []
    
    # Run tests
    results.append(("Pattern Change", test_pattern_change()))
    results.append(("Precursor Detection", test_precursor_detection()))
    results.append(("Subtle Drift", test_subtle_drift()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print("""
    ┌─────────────────────┬──────────┬──────────┬──────────────┐
    │ Test                │ Z-Score  │ Mobiu-AD │ Winner       │
    ├─────────────────────┼──────────┼──────────┼──────────────┤""")
    
    mobiu_wins = 0
    for name, (mobiu, zscore) in results:
        winner = "Mobiu-AD ✅" if mobiu > zscore else "Z-Score" if zscore > mobiu else "Tie"
        if mobiu > zscore:
            mobiu_wins += 1
        print(f"    │ {name:<19} │ {zscore:>8} │ {mobiu:>8} │ {winner:<12} │")
    
    print("""    └─────────────────────┴──────────┴──────────┴──────────────┘""")
    
    print(f"""
    
    CONCLUSION:
    ───────────
    Mobiu-AD won {mobiu_wins}/3 tests by detecting BEHAVIORAL changes
    that standard Z-Score completely misses.
    
    KEY DIFFERENTIATOR:
    ───────────────────
    • Z-Score asks:  "Is this VALUE unusual?"
    • Mobiu-AD asks: "Is this BEHAVIOR unusual?"
    
    The Soft Algebra math (a,b)×(c,d) = (ad+bc, bd) tracks:
    • Trend consistency (at component)
    • Variance changes (precursor detection)  
    • Accumulated "tension" before anomalies
    
    Learn more: https://mobiu.ai
    """)


if __name__ == "__main__":
    main()
