"""
================================================================================
Mobiu-AD Enhanced - Combining PyOD with Soft Algebra
================================================================================

This module provides an enhanced anomaly detector that combines:
1. PyOD base detectors (IForest, LOF, KNN) - good at spatial outliers
2. Mobiu-AD Soft Algebra - good at temporal changes

The ensemble approach captures BOTH types of anomalies!

Usage:
    from mobiu_ad_enhanced import MobiuADEnhanced
    
    detector = MobiuADEnhanced()  # Uses LOF + Soft Algebra by default
    
    # Batch mode
    results = detector.detect_batch(data)
    
    # Streaming mode (coming soon)
    for value in stream:
        result = detector.detect(value)

================================================================================
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Optional PyOD import
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.knn import KNN
    from pyod.models.copod import COPOD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    print("Warning: PyOD not installed. Install with: pip install pyod")


# ============== SOFT ALGEBRA CORE ==============

def soft_multiply(s1: Tuple[float, float], s2: Tuple[float, float]) -> Tuple[float, float]:
    """Nilpotent multiplication: (a,b)×(c,d) = (ad+bc, bd), where ε²=0"""
    return (s1[0]*s2[1] + s1[1]*s2[0], s1[1]*s2[1])


def soft_add(s1: Tuple[float, float], s2: Tuple[float, float]) -> Tuple[float, float]:
    """SoftNumber addition"""
    return (s1[0] + s2[0], s1[1] + s2[1])


def compute_delta_dagger(soft_state: Tuple[float, float]) -> float:
    """
    Super-Equation Δ† - measures emergence/collapse pattern.
    
    From Universal Attention Field Theory:
    The 0→peak→0 profile of attention/measurement.
    """
    a, b = soft_state
    magnitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(a, b + 1e-9) / np.pi
    return magnitude * (1 - abs(phase))


# ============== RESULT DATACLASS ==============

@dataclass
class EnhancedDetectionResult:
    """Result from enhanced detection."""
    index: int
    value: float
    is_anomaly: bool
    score: float  # Combined ensemble score
    pyod_score: float  # Base detector score
    mobiu_score: float  # Soft Algebra score
    confidence: str  # "high", "medium", "low"
    detection_source: str  # "pyod", "mobiu", "both", "ensemble"


# ============== ENHANCED DETECTOR ==============

class MobiuADEnhanced:
    """
    Enhanced Anomaly Detector combining PyOD + Soft Algebra.
    
    This detector captures:
    - Spatial outliers (via PyOD): Points that are statistically unusual
    - Temporal changes (via Soft Algebra): Sudden shifts, transitions, spikes
    
    The ensemble approach means you don't need to choose - it catches both!
    """
    
    VALID_BASES = ["lof", "knn", "iforest", "copod"]
    
    def __init__(
        self,
        base_detector: str = "lof",
        contamination: float = 0.05,
        threshold_percentile: float = 95,
        ensemble_strategy: str = "max",
        gamma: float = 0.9,
        window: int = 30
    ):
        """
        Initialize enhanced detector.
        
        Args:
            base_detector: PyOD base - "lof", "knn", "iforest", or "copod"
            contamination: Expected proportion of anomalies (for PyOD)
            threshold_percentile: Percentile for anomaly threshold
            ensemble_strategy: How to combine scores - "max", "weighted", "boosted"
            gamma: Soft Algebra decay factor
            window: Window size for temporal analysis
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is required. Install with: pip install pyod")
        
        if base_detector not in self.VALID_BASES:
            raise ValueError(f"base_detector must be one of {self.VALID_BASES}")
        
        self.base_name = base_detector
        self.contamination = contamination
        self.threshold_percentile = threshold_percentile
        self.ensemble_strategy = ensemble_strategy
        self.gamma = gamma
        self.window = window
        
        # State
        self._fitted = False
        self._data = None
        self._results = None
    
    def _create_base_detector(self):
        """Create PyOD base detector."""
        if self.base_name == "lof":
            return LOF(contamination=self.contamination, n_neighbors=20)
        elif self.base_name == "knn":
            return KNN(contamination=self.contamination, n_neighbors=10)
        elif self.base_name == "iforest":
            return IForest(contamination=self.contamination, random_state=42)
        elif self.base_name == "copod":
            return COPOD(contamination=self.contamination)
    
    def _compute_mobiu_scores(self, data: np.ndarray) -> np.ndarray:
        """Compute Soft Algebra scores for data."""
        scores = []
        soft_state = (0.0, 0.0)
        history = []
        
        for i, value in enumerate(data):
            history.append(value)
            if len(history) > self.window:
                history = history[-self.window:]
            
            if i < 3:
                scores.append(0.0)
                continue
            
            # at = rate of change
            at = abs(history[-1] - history[-2])
            
            # bt = acceleration (second derivative)
            if len(history) >= 3:
                d1 = history[-1] - history[-2]
                d2 = history[-2] - history[-3]
                bt = abs(d1 - d2)
            else:
                bt = 0.0
            
            # Normalize
            std = np.std(history) + 1e-9
            at = at / std
            bt = bt / std
            
            # Evolve soft state
            delta_sn = (at, bt)
            decayed = (soft_state[0] * self.gamma, soft_state[1] * self.gamma)
            evolved = soft_multiply(decayed, delta_sn)
            soft_state = soft_add(evolved, delta_sn)
            
            # Compute Δ†
            delta_dagger = compute_delta_dagger(soft_state)
            scores.append(delta_dagger)
        
        return np.array(scores)
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
    
    def _combine_scores(
        self, 
        pyod_norm: np.ndarray, 
        mobiu_norm: np.ndarray
    ) -> np.ndarray:
        """Combine PyOD and Mobiu scores using ensemble strategy."""
        if self.ensemble_strategy == "max":
            # Catches what EITHER method catches
            return np.maximum(pyod_norm, mobiu_norm)
        
        elif self.ensemble_strategy == "weighted":
            # Weighted average, boosted by agreement
            agreement = pyod_norm * mobiu_norm
            return 0.4 * pyod_norm + 0.4 * mobiu_norm + 0.2 * agreement * 2
        
        elif self.ensemble_strategy == "boosted":
            # Additive with agreement bonus
            return pyod_norm + mobiu_norm + (pyod_norm * mobiu_norm)
        
        else:
            return np.maximum(pyod_norm, mobiu_norm)
    
    def detect_batch(self, data: np.ndarray) -> List[EnhancedDetectionResult]:
        """
        Detect anomalies in a batch of data.
        
        Args:
            data: Array of values
        
        Returns:
            List of EnhancedDetectionResult for each point
        """
        data = np.array(data).flatten()
        X = data.reshape(-1, 1)
        
        # PyOD scores
        base = self._create_base_detector()
        base.fit(X)
        pyod_scores = base.decision_scores_
        pyod_norm = self._normalize(pyod_scores)
        
        # Mobiu scores
        mobiu_scores = self._compute_mobiu_scores(data)
        mobiu_norm = self._normalize(mobiu_scores)
        
        # Ensemble
        ensemble_scores = self._combine_scores(pyod_norm, mobiu_norm)
        
        # Threshold
        threshold = np.percentile(ensemble_scores, self.threshold_percentile)
        
        # Build results
        results = []
        for i in range(len(data)):
            is_anomaly = ensemble_scores[i] > threshold
            
            # Determine source
            pyod_high = pyod_norm[i] > np.percentile(pyod_norm, 90)
            mobiu_high = mobiu_norm[i] > np.percentile(mobiu_norm, 90)
            
            if pyod_high and mobiu_high:
                source = "both"
                confidence = "high"
            elif pyod_high:
                source = "pyod"
                confidence = "medium"
            elif mobiu_high:
                source = "mobiu"
                confidence = "medium"
            else:
                source = "ensemble"
                confidence = "low"
            
            results.append(EnhancedDetectionResult(
                index=i,
                value=float(data[i]),
                is_anomaly=is_anomaly,
                score=float(ensemble_scores[i]),
                pyod_score=float(pyod_scores[i]),
                mobiu_score=float(mobiu_scores[i]),
                confidence=confidence if is_anomaly else "n/a",
                detection_source=source if is_anomaly else "n/a"
            ))
        
        self._fitted = True
        self._data = data
        self._results = results
        
        return results
    
    def get_anomalies(self) -> List[EnhancedDetectionResult]:
        """Get only the anomalous points."""
        if not self._fitted:
            raise RuntimeError("Call detect_batch first")
        return [r for r in self._results if r.is_anomaly]
    
    def summary(self) -> Dict[str, Any]:
        """Get detection summary."""
        if not self._fitted:
            raise RuntimeError("Call detect_batch first")
        
        anomalies = self.get_anomalies()
        
        return {
            "total_points": len(self._data),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": len(anomalies) / len(self._data),
            "base_detector": self.base_name,
            "ensemble_strategy": self.ensemble_strategy,
            "by_source": {
                "both": sum(1 for a in anomalies if a.detection_source == "both"),
                "pyod_only": sum(1 for a in anomalies if a.detection_source == "pyod"),
                "mobiu_only": sum(1 for a in anomalies if a.detection_source == "mobiu"),
                "ensemble": sum(1 for a in anomalies if a.detection_source == "ensemble"),
            },
            "by_confidence": {
                "high": sum(1 for a in anomalies if a.confidence == "high"),
                "medium": sum(1 for a in anomalies if a.confidence == "medium"),
                "low": sum(1 for a in anomalies if a.confidence == "low"),
            }
        }


# ============== CONVENIENCE FUNCTION ==============

def detect_anomalies_enhanced(
    data: np.ndarray,
    base_detector: str = "lof",
    contamination: float = 0.05,
    threshold_percentile: float = 95
) -> Tuple[List[int], List[EnhancedDetectionResult]]:
    """
    Detect anomalies using enhanced PyOD + Soft Algebra ensemble.
    
    Args:
        data: Array of values
        base_detector: "lof", "knn", "iforest", or "copod"
        contamination: Expected proportion of anomalies
        threshold_percentile: Percentile for threshold
    
    Returns:
        Tuple of (anomaly_indices, all_results)
    
    Example:
        indices, results = detect_anomalies_enhanced(my_data)
        print(f"Anomalies at: {indices}")
    """
    detector = MobiuADEnhanced(
        base_detector=base_detector,
        contamination=contamination,
        threshold_percentile=threshold_percentile
    )
    results = detector.detect_batch(data)
    indices = [r.index for r in results if r.is_anomaly]
    
    return indices, results


if __name__ == "__main__":
    # Quick test
    print("Testing MobiuADEnhanced...")
    
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(50, 3, 100),
        [90],  # spike
        np.random.normal(50, 3, 50),
        [15],  # low spike
        np.random.normal(50, 3, 50),
    ])
    
    detector = MobiuADEnhanced()
    results = detector.detect_batch(data)
    
    print(f"\n{detector.summary()}")
    
    print("\nAnomalies:")
    for a in detector.get_anomalies():
        print(f"  Index {a.index}: value={a.value:.1f}, source={a.detection_source}, conf={a.confidence}")
