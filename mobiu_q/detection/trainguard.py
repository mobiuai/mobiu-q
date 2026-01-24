"""
TrainGuard - Mobiu-Q + Mobiu-AD Integration
============================================

Combines optimization (Q) and monitoring (AD) for safer ML training.
Both use cloud APIs directly.

Requirements:
    pip install requests numpy

Usage:
    from mobiu_ad import TrainGuard
    
    guard = TrainGuard(license_key="your-key")
    
    for epoch in training:
        loss = train_step()
        result = guard.step(loss=loss, gradient=grad, val_loss=val_loss)
        
        # Use optimized gradient from Mobiu-Q
        optimized_grad = result.adjusted_gradient
        
        # Check for alerts from Mobiu-AD
        if result.alert:
            print(f"⚠️ {result.alert_type}: {result.alert_message}")
"""

import requests
import numpy as np
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

# API endpoints
MOBIU_Q_API = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"
MOBIU_AD_API = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_ad"


@dataclass
class TrainGuardResult:
    """Result from TrainGuard step."""
    # Mobiu-Q outputs
    scale: float
    trust_ratio: float
    adjusted_gradient: List[float]
    
    # Mobiu-AD outputs
    train_alert: bool
    val_alert: bool
    train_score: float
    val_score: float
    
    # Combined alert
    alert: bool
    alert_type: Optional[str]
    alert_message: Optional[str]


class TrainGuard:
    """
    TrainGuard: Safer ML Training with Mobiu-Q + Mobiu-AD
    
    Mobiu-Q:  Optimizes gradient steps using Soft Algebra (via API)
    Mobiu-AD: Monitors loss patterns for anomalies (via API)
    
    Together they provide:
    - Better convergence (Q)
    - Early warning for problems (AD)
    - Overfitting detection (AD on val_loss)
    - Gradient explosion detection (AD on train_loss)
    """
    
    def __init__(
        self,
        license_key: str,
        q_method: str = "standard",
        q_mode: str = "simulation",
        ad_method: str = "deep",
        base_lr: float = 0.01,
    ):
        """
        Initialize TrainGuard.
        
        Args:
            license_key: Mobiu license key (works for both Q and AD)
            q_method: Mobiu-Q method (standard/deep/adaptive)
            q_mode: Mobiu-Q mode (simulation/hardware)
            ad_method: Mobiu-AD method (deep/transition/auto)
            base_lr: Base learning rate for Q optimizer
        """
        self.license_key = license_key
        self.ad_method = ad_method
        self.q_method = q_method
        self.q_mode = q_mode
        self.base_lr = base_lr
        
        # Q session management
        self.q_session_id = None
        self._q_initialized = False
        
        # AD session IDs
        self.train_ad_session = f"tg_train_{int(time.time())}"
        self.val_ad_session = f"tg_val_{int(time.time())}"
        
        # History for analysis
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'scales': [],
            'train_alerts': [],
            'val_alerts': [],
        }
    
    def _init_q_session(self) -> bool:
        """Initialize Q session with 'start' action."""
        try:
            response = requests.post(MOBIU_Q_API, json={
                'action': 'start',
                'license_key': self.license_key,
                'method': self.q_method,
                'mode': self.q_mode,
                'base_lr': self.base_lr,
                'base_optimizer': 'Adam',
                'use_soft_algebra': True,
                'maximize': False
            }, timeout=10)
            
            data = response.json()
            
            if data.get('success'):
                self.q_session_id = data['session_id']
                self._q_initialized = True
                return True
            else:
                print(f"⚠️ Q API start failed: {data.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"⚠️ Q API start error: {e}")
            return False
    
    def _call_q_api(self, energy: float, gradient: List[float]) -> dict:
        """Call Mobiu-Q API with 'step' action."""
        # Initialize session if needed
        if not self._q_initialized:
            if not self._init_q_session():
                return {
                    "scale": 1.0,
                    "trust_ratio": 0.5,
                    "adjusted_gradient": gradient
                }
        
        try:
            response = requests.post(MOBIU_Q_API, json={
                'action': 'step',
                'license_key': self.license_key,
                'session_id': self.q_session_id,
                'params': [0.0] * len(gradient),
                'gradient': [float(g) for g in gradient],
                'energy': float(energy)
            }, timeout=10)
            
            data = response.json()
            
            if data.get('success'):
                # Extract scale from response (warp_factor in Q API)
                scale = data.get('warp_factor', 1.0)
                adaptive_lr = data.get('adaptive_lr', self.base_lr)
                
                # Use warp_factor as trust indicator
                trust_ratio = min(1.0, scale) if scale > 0 else 0.5
                
                # Apply scale to gradient
                adjusted_gradient = [g * scale for g in gradient]
                
                return {
                    "scale": scale,
                    "trust_ratio": trust_ratio,
                    "adjusted_gradient": adjusted_gradient,
                    "adaptive_lr": adaptive_lr
                }
            else:
                return {
                    "scale": 1.0,
                    "trust_ratio": 0.5,
                    "adjusted_gradient": gradient
                }
        except Exception as e:
            print(f"⚠️ Q API error: {e}")
            return {
                "scale": 1.0,
                "trust_ratio": 0.5,
                "adjusted_gradient": gradient
            }
    
    def _call_ad_api(self, session_id: str, value: float) -> dict:
        """Call Mobiu-AD API."""
        try:
            response = requests.post(MOBIU_AD_API, json={
                "license_key": self.license_key,
                "session_id": session_id,
                "method": self.ad_method,
                "value": float(value),
            }, timeout=30)
            return response.json()
        except Exception as e:
            print(f"⚠️ AD API error: {e}")
            return {
                "is_anomaly": False,
                "delta_dagger": 0.0
            }
    
    def step(
        self,
        loss: float,
        gradient: List[float],
        val_loss: Optional[float] = None,
    ) -> TrainGuardResult:
        """
        Perform one TrainGuard step.
        
        Args:
            loss: Current training loss
            gradient: Current gradient
            val_loss: Optional validation loss (for overfitting detection)
            
        Returns:
            TrainGuardResult with optimized gradient and alerts
        """
        # ═══════════════════════════════════════════════════════════════
        # Mobiu-Q: Optimize gradient (via API)
        # ═══════════════════════════════════════════════════════════════
        q_result = self._call_q_api(loss, gradient)
        scale = q_result.get("scale", 1.0)
        trust_ratio = q_result.get("trust_ratio", 0.5)
        adjusted_gradient = q_result.get("adjusted_gradient", gradient)
        
        # ═══════════════════════════════════════════════════════════════
        # Mobiu-AD: Monitor training loss (via API)
        # ═══════════════════════════════════════════════════════════════
        train_result = self._call_ad_api(self.train_ad_session, loss)
        train_alert = train_result.get("is_anomaly", False)
        train_score = train_result.get("delta_dagger", 0.0)
        
        # ═══════════════════════════════════════════════════════════════
        # Mobiu-AD: Monitor validation loss (via API)
        # ═══════════════════════════════════════════════════════════════
        val_alert = False
        val_score = 0.0
        
        if val_loss is not None:
            val_result = self._call_ad_api(self.val_ad_session, val_loss)
            val_alert = val_result.get("is_anomaly", False)
            val_score = val_result.get("delta_dagger", 0.0)
        
        # ═══════════════════════════════════════════════════════════════
        # Update history
        # ═══════════════════════════════════════════════════════════════
        self.history['train_loss'].append(loss)
        self.history['scales'].append(scale)
        
        if train_alert:
            self.history['train_alerts'].append(len(self.history['train_loss']) - 1)
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
            if val_alert:
                self.history['val_alerts'].append(len(self.history['val_loss']) - 1)
        
        # ═══════════════════════════════════════════════════════════════
        # Determine alert type and message
        # ═══════════════════════════════════════════════════════════════
        alert = train_alert or val_alert
        alert_type = None
        alert_message = None
        
        if train_alert and val_alert:
            alert_type = "CRITICAL"
            alert_message = "Both train and val loss show anomalies!"
            
        elif val_alert:
            # Check for overfitting pattern
            if len(self.history['train_loss']) > 5:
                recent_train = np.mean(self.history['train_loss'][-5:])
                past_train = np.mean(self.history['train_loss'][-10:-5]) if len(self.history['train_loss']) > 10 else recent_train
                
                if recent_train < past_train:
                    alert_type = "OVERFITTING"
                    alert_message = "Val loss pattern changed while train improves - possible overfitting!"
                else:
                    alert_type = "VAL_ANOMALY"
                    alert_message = "Validation loss pattern changed"
            else:
                alert_type = "VAL_ANOMALY"
                alert_message = "Validation loss anomaly detected"
                
        elif train_alert:
            # Check for explosion or plateau
            if len(self.history['train_loss']) > 3:
                recent = self.history['train_loss'][-3:]
                
                if recent[-1] > recent[0] * 1.5:
                    alert_type = "EXPLOSION"
                    alert_message = "Training loss increasing rapidly!"
                elif abs(recent[-1] - recent[0]) < 0.001 * abs(recent[0] + 1e-9):
                    alert_type = "PLATEAU"
                    alert_message = "Training loss stagnating"
                else:
                    alert_type = "TRAIN_ANOMALY"
                    alert_message = "Training loss pattern changed"
            else:
                alert_type = "TRAIN_ANOMALY"
                alert_message = "Training loss anomaly detected"
        
        return TrainGuardResult(
            scale=scale,
            trust_ratio=trust_ratio,
            adjusted_gradient=adjusted_gradient,
            train_alert=train_alert,
            val_alert=val_alert,
            train_score=train_score,
            val_score=val_score,
            alert=alert,
            alert_type=alert_type,
            alert_message=alert_message,
        )
    
    def get_summary(self) -> Dict:
        """Get training summary."""
        return {
            'total_steps': len(self.history['train_loss']),
            'final_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'train_alerts': len(self.history['train_alerts']),
            'val_alerts': len(self.history['val_alerts']),
            'avg_scale': np.mean(self.history['scales']) if self.history['scales'] else 1.0,
        }
    
    def reset(self):
        """Reset TrainGuard state for new training run."""
        # Reset Q session
        self._q_initialized = False
        self.q_session_id = None
        
        # New AD session IDs
        self.train_ad_session = f"tg_train_{int(time.time())}"
        self.val_ad_session = f"tg_val_{int(time.time())}"
        
        # Reset history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'scales': [],
            'train_alerts': [],
            'val_alerts': [],
        }


if __name__ == "__main__":
    print("="*60)
    print("TrainGuard - Mobiu-Q + Mobiu-AD Integration")
    print("="*60)
    print("""
Usage Example:
--------------
from mobiu_ad import TrainGuard

# Initialize with your license key
guard = TrainGuard(license_key="your-key")

# Training loop
for epoch in range(100):
    loss = train_step()
    val_loss = validate()
    
    # TrainGuard step (calls both Q and AD APIs)
    result = guard.step(
        loss=loss,
        gradient=model.get_gradients(),
        val_loss=val_loss
    )
    
    # Mobiu-Q: Use optimized gradient
    model.apply_gradients(result.adjusted_gradient)
    
    # Mobiu-AD: Check alerts
    if result.alert:
        print(f"⚠️ {result.alert_type}: {result.alert_message}")
        
        if result.alert_type == 'OVERFITTING':
            reduce_learning_rate()
        elif result.alert_type == 'EXPLOSION':
            break
""")
