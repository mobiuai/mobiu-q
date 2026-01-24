#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║     BEHAVIORAL ANOMALY DETECTION - MOBIU's STRENGTH                  ║
║                                                                      ║
║  Test cases where VALUES look normal but BEHAVIOR changed            ║
║  This is where Mobiu-AD should beat PyOD                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import requests
import time
from typing import List, Tuple

# Get license key from environment
LICENSE_KEY = os.environ.get("MOBIU_LICENSE_KEY", "")
if not LICENSE_KEY:
    print("⚠️  Set MOBIU_LICENSE_KEY environment variable")
    print("   export MOBIU_LICENSE_KEY='your-key-here'")

MOBIU_AD_API = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_ad"


def detect_mobiu_ad(data: np.ndarray, session_id: str) -> Tuple[List[bool], List[float]]:
    """Mobiu-AD streaming detection."""
    preds = []
    scores = []
    
    for i, v in enumerate(data):
        try:
            r = requests.post(MOBIU_AD_API, json={
                'license_key': LICENSE_KEY,
                'session_id': session_id,
                'method': 'deep',
                'value': float(v)
            }, timeout=5)
            result = r.json()
            preds.append(result.get('is_anomaly', False))
            scores.append(result.get('delta_dagger', 0.0))
        except:
            preds.append(False)
            scores.append(0.0)
    
    return preds, scores


def detect_pyod(data: np.ndarray) -> Tuple[List[bool], List[float]]:
    """PyOD Isolation Forest."""
    try:
        from pyod.models.iforest import IForest
        X = data.reshape(-1, 1)
        model = IForest(contamination=0.1, random_state=42)
        model.fit(X)
        return model.labels_.astype(bool).tolist(), model.decision_scores_.tolist()
    except:
        mean, std = np.mean(data), np.std(data)
        scores = np.abs((data - mean) / (std + 1e-9))
        return (scores > 2.5).tolist(), scores.tolist()


def detect_zscore(data: np.ndarray) -> Tuple[List[bool], List[float]]:
    """Z-Score baseline."""
    mean, std = np.mean(data), np.std(data)
    scores = np.abs((data - mean) / (std + 1e-9))
    return (scores > 2.5).tolist(), scores.tolist()


def count_detections_in_range(preds: List[bool], start: int, end: int) -> int:
    """Count detections in range."""
    return sum(1 for i in range(start, min(end, len(preds))) if preds[i])


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     BEHAVIORAL ANOMALY DETECTION - MOBIU's STRENGTH                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    if not LICENSE_KEY:
        print("❌ No license key. Set MOBIU_LICENSE_KEY and retry.")
        return
    
    results = {}
    
    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: Pattern Change (Random → Trending)
    # ═══════════════════════════════════════════════════════════════════
    print("="*60)
    print("🧪 TEST 1: Pattern Change (Random → Trending)")
    print("="*60)
    print("""
    Data:
    • Points 0-59:  Random fluctuation around 50 (std=3)
    • Points 60-99: Consistent downward trend (-0.25/step)
    
    The VALUES stay between 40-60 (look normal!)
    But the BEHAVIOR changed from random to trending.
    """)
    
    np.random.seed(42)
    n = 100
    data = np.zeros(n)
    data[:60] = 50 + np.random.normal(0, 3, 60)
    trend = 50 - np.arange(40) * 0.25
    data[60:] = trend + np.random.normal(0, 1, 40)
    
    anomaly_range = (60, 100)
    
    print("Running detectors...")
    mobiu_preds, _ = detect_mobiu_ad(data, f"pattern_{int(time.time())}")
    pyod_preds, _ = detect_pyod(data)
    zscore_preds, _ = detect_zscore(data)
    
    mobiu_det = count_detections_in_range(mobiu_preds, *anomaly_range)
    pyod_det = count_detections_in_range(pyod_preds, *anomaly_range)
    zscore_det = count_detections_in_range(zscore_preds, *anomaly_range)
    
    results['pattern_change'] = {'mobiu': mobiu_det, 'pyod': pyod_det, 'zscore': zscore_det}
    
    print(f"""
    Results (detections in anomaly region 60-100):
    ┌────────────┬────────────┐
    │ Z-Score    │ {zscore_det:>10} │
    │ PyOD       │ {pyod_det:>10} │
    │ Mobiu-AD   │ {mobiu_det:>10} │
    └────────────┴────────────┘
    """)
    
    if mobiu_det > pyod_det:
        print(f"    ✅ MOBIU WINS: {mobiu_det} vs {pyod_det}")
    elif pyod_det > mobiu_det:
        print(f"    ❌ PyOD wins: {pyod_det} vs {mobiu_det}")
    else:
        print(f"    🤝 TIE")
    
    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: Variance Change (Precursor Detection)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("🧪 TEST 2: Variance Change (Early Warning)")
    print("="*60)
    print("""
    Data:
    • Points 0-49:  Normal (std=2)
    • Points 50-69: Variance increasing (std: 2→10) - PRECURSOR
    • Point 70:     SPIKE (value=90)
    
    Can we detect the buildup BEFORE the spike?
    """)
    
    np.random.seed(123)
    n = 100
    data = np.zeros(n)
    data[:50] = 50 + np.random.normal(0, 2, 50)
    
    for i in range(20):
        std = 2 + i * 0.4
        data[50 + i] = 50 + np.random.normal(0, std)
    
    data[70] = 90
    data[71:] = 50 + np.random.normal(0, 2, 29)
    
    precursor_range = (50, 70)
    
    print("Running detectors...")
    mobiu_preds, _ = detect_mobiu_ad(data, f"precursor_{int(time.time())}")
    pyod_preds, _ = detect_pyod(data)
    zscore_preds, _ = detect_zscore(data)
    
    mobiu_early = count_detections_in_range(mobiu_preds, *precursor_range)
    pyod_early = count_detections_in_range(pyod_preds, *precursor_range)
    zscore_early = count_detections_in_range(zscore_preds, *precursor_range)
    
    results['precursor'] = {'mobiu': mobiu_early, 'pyod': pyod_early, 'zscore': zscore_early}
    
    print(f"""
    Results (early warnings BEFORE spike at 70):
    ┌────────────┬────────────┐
    │ Z-Score    │ {zscore_early:>10} │
    │ PyOD       │ {pyod_early:>10} │
    │ Mobiu-AD   │ {mobiu_early:>10} │
    └────────────┴────────────┘
    """)
    
    if mobiu_early > pyod_early:
        print(f"    ✅ MOBIU WINS: {mobiu_early} vs {pyod_early} early warnings!")
    elif pyod_early > mobiu_early:
        print(f"    ❌ PyOD wins: {pyod_early} vs {mobiu_early}")
    else:
        print(f"    🤝 TIE")
    
    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: Subtle Drift
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("🧪 TEST 3: Subtle Drift")
    print("="*60)
    print("""
    Data:
    • Points 0-39:  Stable at 50 (std=2)
    • Points 40-99: Slow drift upward (+0.1/step)
    
    Each individual value looks normal.
    The TREND is anomalous.
    """)
    
    np.random.seed(456)
    n = 100
    data = np.zeros(n)
    data[:40] = 50 + np.random.normal(0, 2, 40)
    
    for i in range(60):
        data[40 + i] = 50 + i * 0.1 + np.random.normal(0, 2)
    
    drift_range = (40, 100)
    
    print("Running detectors...")
    mobiu_preds, _ = detect_mobiu_ad(data, f"drift_{int(time.time())}")
    pyod_preds, _ = detect_pyod(data)
    zscore_preds, _ = detect_zscore(data)
    
    mobiu_drift = count_detections_in_range(mobiu_preds, *drift_range)
    pyod_drift = count_detections_in_range(pyod_preds, *drift_range)
    zscore_drift = count_detections_in_range(zscore_preds, *drift_range)
    
    results['drift'] = {'mobiu': mobiu_drift, 'pyod': pyod_drift, 'zscore': zscore_drift}
    
    print(f"""
    Results (detections during drift):
    ┌────────────┬────────────┐
    │ Z-Score    │ {zscore_drift:>10} │
    │ PyOD       │ {pyod_drift:>10} │
    │ Mobiu-AD   │ {mobiu_drift:>10} │
    └────────────┴────────────┘
    """)
    
    if mobiu_drift > pyod_drift:
        print(f"    ✅ MOBIU WINS: {mobiu_drift} vs {pyod_drift}")
    elif pyod_drift > mobiu_drift:
        print(f"    ❌ PyOD wins: {pyod_drift} vs {mobiu_drift}")
    else:
        print(f"    🤝 TIE")
    
    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    print("""
    ┌─────────────────────┬──────────┬──────────┬──────────┬──────────────┐
    │ Test                │ Z-Score  │ PyOD     │ Mobiu-AD │ Winner       │
    ├─────────────────────┼──────────┼──────────┼──────────┼──────────────┤""")
    
    mobiu_wins = 0
    pyod_wins = 0
    
    for test_name, r in results.items():
        m = r['mobiu']
        p = r['pyod']
        z = r['zscore']
        
        if m > p:
            winner = "Mobiu ✅"
            mobiu_wins += 1
        elif p > m:
            winner = "PyOD"
            pyod_wins += 1
        else:
            winner = "TIE"
        
        display_name = test_name.replace('_', ' ').title()
        print(f"    │ {display_name:<19} │ {z:>8} │ {p:>8} │ {m:>8} │ {winner:<12} │")
    
    print("""    └─────────────────────┴──────────┴──────────┴──────────┴──────────────┘""")
    
    print(f"""
    SCORE: Mobiu {mobiu_wins} vs PyOD {pyod_wins}
    """)
    
    if mobiu_wins > pyod_wins:
        print("""
    🏆 MOBIU-AD WINS on BEHAVIORAL anomalies!
    
    Mobiu-AD excels at:
    • Pattern changes (random→trending)
    • Early warning (precursor detection)
    • Subtle drift detection
    
    PyOD is better for:
    • Obvious outliers (spikes)
    • Batch analysis
    
    CONCLUSION: Different tools for different jobs.
    Use Mobiu-AD when you need to detect BEHAVIOR changes.
    Use PyOD when you need to detect VALUE outliers.
        """)
    elif pyod_wins > mobiu_wins:
        print("""
    ❌ PyOD wins even on behavioral tests.
    
    Need to investigate Mobiu-AD further.
        """)
    else:
        print("""
    🤝 TIE - Both have their strengths.
        """)


if __name__ == "__main__":
    main()
