"""
MOBIU-Q CUSTOMER VALIDATION (v2.1)
==================================
Validated against SDK: core.py (MobiuQCore)

This script demonstrates Mobiu-Q's superiority using the OFFICIAL class structure.

Usage:
    python customer_validation.py
"""

import numpy as np
import requests
import json
import time

# ==============================================================================
# 👇 PUT YOUR LICENSE KEY HERE 👇
# ==============================================================================
LICENSE_KEY = "YOUR_LICENSE_KEY_HERE"
# ==============================================================================

API_ENDPOINT = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"

# ------------------------------------------------------------------------------
# 1. MOBIU-Q CORE (Matches core.py exactly)
# ------------------------------------------------------------------------------
class MobiuQCore:
    """
    Official Client Logic (Embedded for Validation Script)
    """
    def __init__(self, license_key, mode="standard", problem="vqe", base_lr=None):
        self.license_key = license_key
        self.mode = mode
        self.problem = problem
        self.base_lr = base_lr
        self.session_id = None
        self.api_endpoint = API_ENDPOINT
        
        # Start Session
        try:
            r = requests.post(self.api_endpoint, json={
                "license_key": self.license_key,
                "action": "start",
                "mode": self.mode,
                "problem": self.problem,
                "base_lr": self.base_lr
            }, timeout=10)
            data = r.json()
            
            if data.get("success"):
                self.session_id = data["session_id"]
                print(f"   [Mobiu] Connected. Session: {self.session_id[:8]}...")
            else:
                print(f"   [Mobiu] Init Error: {data.get('error')}")
                
        except Exception as e:
            print(f"   [Mobiu] Connection Error: {e}")

    def step(self, params, gradient, energy):
        if not self.session_id:
            return params
        
        # Convert to list for JSON
        p_list = params.tolist() if isinstance(params, np.ndarray) else params
        g_list = gradient.tolist() if isinstance(gradient, np.ndarray) else gradient
        
        try:
            r = requests.post(self.api_endpoint, json={
                "license_key": self.license_key,
                "session_id": self.session_id,
                "action": "step",
                "params": p_list,
                "gradient": g_list,
                "energy": float(energy)
            }, timeout=10)
            data = r.json()
            
            if data.get("success"):
                return np.array(data["new_params"])
            else:
                return params
        except:
            return params

    def end(self):
        if self.session_id:
            try:
                requests.post(self.api_endpoint, json={
                    "license_key": self.license_key,
                    "session_id": self.session_id,
                    "action": "end"
                }, timeout=3)
            except: pass

# ------------------------------------------------------------------------------
# 2. HELPER: DEMEASUREMENT (Gradient Estimation)
# ------------------------------------------------------------------------------
class Demeasurement:
    @staticmethod
    def finite_difference(circuit_fn, params, epsilon=1e-3):
        grad = np.zeros_like(params)
        base = circuit_fn(params)
        for i in range(len(params)):
            p_plus = params.copy(); p_plus[i] += epsilon
            grad[i] = (circuit_fn(p_plus) - base) / epsilon
        return grad

# ------------------------------------------------------------------------------
# 3. LOCAL ADAM (Baseline)
# ------------------------------------------------------------------------------
class AdamOptimizer:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.epsilon = epsilon
        self.m = None; self.v = None; self.t = 0

    def step(self, params, grads):
        if self.m is None: self.m = np.zeros_like(params); self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# ------------------------------------------------------------------------------
# 4. BENCHMARK RUNNER
# ------------------------------------------------------------------------------
def h2_energy(params):
    # Simulated H2 Landscape
    p0, p1 = params[0], params[1]
    return -1.137 + (p0 - 0.5)**2 + 2.5*(p1 + 0.2)**2 + 0.1*np.sin(5*p0)*np.cos(5*p1)

def run_test():
    print(f"\n⚡ Benchmarking: H2 Molecule (VQE)")
    print("-" * 60)
    
    np.random.seed(42)
    init_params = np.random.uniform(-1, 1, 2)
    
    # 1. Adam
    adam = AdamOptimizer(lr=0.05)
    p_adam = init_params.copy()
    loss_adam = 0
    for _ in range(60):
        loss_adam = h2_energy(p_adam)
        grad = Demeasurement.finite_difference(h2_energy, p_adam)
        p_adam = adam.step(p_adam, grad)
        
    # 2. Mobiu-Q
    opt = MobiuQCore(license_key=LICENSE_KEY, problem="vqe", base_lr=0.05)
    p_mobiu = init_params.copy()
    loss_mobiu = 0
    for _ in range(60):
        loss_mobiu = h2_energy(p_mobiu)
        grad = Demeasurement.finite_difference(h2_energy, p_mobiu)
        p_mobiu = opt.step(p_mobiu, grad, loss_mobiu)
    opt.end()
    
    print(f"   📉 Adam Final Loss:    {loss_adam:.6f}")
    print(f"   🚀 Mobiu-Q Final Loss: {loss_mobiu:.6f}")
    
    imp = ((loss_adam - loss_mobiu) / abs(loss_adam)) * 100
    color = "\033[92m" if imp > 0 else "\033[91m"
    print(f"   🏆 Improvement:        {color}{imp:+.2f}%\033[0m")

if __name__ == "__main__":
    if LICENSE_KEY == "YOUR_LICENSE_KEY_HERE":
        print("\n❌ Please set your LICENSE_KEY in the script!")
    else:
        run_test()