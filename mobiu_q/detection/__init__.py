"""
Mobiu-Q Detection Module
========================

Anomaly detection using the same Soft Algebra mathematics as the optimizer.

Classes:
    MobiuAD: Streaming anomaly detector
    MobiuADLocal: Local detector (no API)
    TrainGuard: Combined Q optimizer + AD monitor

Usage:
    from mobiu_q.detection import MobiuAD, TrainGuard
    
    # Standalone detection
    detector = MobiuAD(license_key="your-key")
    for value in stream:
        result = detector.detect(value)
        if result.is_anomaly:
            print(f"Anomaly! Score: {result.delta_dagger}")
    
    # Training guard (Q + AD combined)
    guard = TrainGuard(license_key="your-key")
    for epoch in training:
        result = guard.step(loss, gradient, val_loss)
        optimized_grad = result.adjusted_gradient
        if result.alert:
            print(f"{result.alert_type}: {result.alert_message}")
"""

from .detector import (
    MobiuAD,
    MobiuADLocal,
    DetectionResult,
    BatchResult,
    detect_anomalies,
    find_transitions,
)

from .trainguard import (
    TrainGuard,
    TrainGuardResult,
)

# Enhanced module (requires PyOD - optional)
try:
    from .enhanced import (
        MobiuADEnhanced,
        EnhancedDetectionResult,
        detect_anomalies_enhanced,
    )
    _ENHANCED_AVAILABLE = True
except ImportError:
    _ENHANCED_AVAILABLE = False

__all__ = [
    # Core detection
    "MobiuAD",
    "MobiuADLocal", 
    "DetectionResult",
    "BatchResult",
    "detect_anomalies",
    "find_transitions",
    
    # TrainGuard
    "TrainGuard",
    "TrainGuardResult",
    
    # Enhanced (PyOD + Soft Algebra)
    "MobiuADEnhanced",
    "EnhancedDetectionResult",
    "detect_anomalies_enhanced",
]
