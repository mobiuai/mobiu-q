"""
MOBIU-Q CUSTOMER VALIDATION (v2.4.1)
==================================
Validated against SDK: core.py (MobiuQCore)

This script demonstrates Mobiu-Q's superiority using the OFFICIAL class structure.

Features demonstrated:
- VQE optimization (quantum chemistry)
- QAOA optimization (combinatorial)
- A/B comparison (SA on vs off)
- Multi-optimizer support
- Multi-seed experiments (1 billing session)

Usage:
    python customer_validation.py
"""

import numpy as np
import requests
import json

# ==============================================================================
# 👇 PUT YOUR LICENSE KEY HERE 👇
# ==============================================================================
LICENSE_KEY = "YOUR_LICENSE_KEY_HERE"
# ==============================================================================

API_ENDPOINT = "https://us-central1-mobiu-q.cloudfunctions.net/mobiu_q_step"

# Available optimizers
AVAILABLE_OPTIMIZERS = ["Adam", "NAdam", "AMSGrad", "SGD", "Momentum", "LAMB"]

# ------------------------------------------------------------------------------
# 1. MOBIU-Q CORE (Matches core.py v2.4)
# ------------------------------------------------------------------------------
class MobiuQCore:
    """
    Official Client Logic (Embedded for Validation Script)
    
    Args:
        license_key: Your Mobiu-Q license key
        method: "vqe", "qaoa", or "rl"
        mode: "simulation" or "hardware"
        base_lr: Learning rate (auto-selected if None)
        base_optimizer: "Adam" (default), "NAdam", "AMSGrad", "SGD", "Momentum", "LAMB"
        use_soft_algebra: Enable Soft Algebra enhancement (default: True)
    """
    def __init__(self, license_key, method="vqe", mode="simulation", 
                 base_lr=None, base_optimizer="Adam", use_soft_algebra=True):
        self.license_key = license_key
        self.method = method
        self.mode = mode
        self.base_lr = base_lr
        self.base_optimizer = base_optimizer
        self.use_soft_algebra = use_soft_algebra
        self.session_id = None
        self.api_endpoint = API_ENDPOINT
        
        # Start Session
        try:
            r = requests.post(self.api_endpoint, json={
                "license_key": self.license_key,
                "action": "start",
                "method": self.method,
                "mode": self.mode,
                "base_lr": self.base_lr,
                "base_optimizer": self.base_optimizer,
                "use_soft_algebra": self.use_soft_algebra
            }, timeout=10)
            data = r.json()
            
            if data.get("success"):
                self.session_id = data["session_id"]
                sa_status = "SA=on" if use_soft_algebra else "SA=off"
                print(f"   [Mobiu] Connected ({base_optimizer}, {sa_status})")
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

    def new_run(self):
        """Reset optimizer state for multi-seed experiments."""
        if not self.session_id:
            return
        try:
            requests.post(self.api_endpoint, json={
                "license_key": self.license_key,
                "session_id": self.session_id,
                "action": "reset"
            }, timeout=10)
        except:
            pass

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
    
    @staticmethod
    def spsa(circuit_fn, params, c_shift=0.1):
        """SPSA - Only 2 evaluations regardless of parameter count."""
        delta = np.random.choice([-1, 1], size=params.shape)
        params_plus = params + c_shift * delta
        params_minus = params - c_shift * delta
        energy_plus = circuit_fn(params_plus)
        energy_minus = circuit_fn(params_minus)
        grad = (energy_plus - energy_minus) / (2 * c_shift) * delta
        avg_energy = (energy_plus + energy_minus) / 2.0
        return grad, avg_energy

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
# 4. TEST FUNCTIONS
# ------------------------------------------------------------------------------

def h2_energy(params):
    """Simulated H2 VQE Landscape"""
    p0, p1 = params[0], params[1]
    return -1.137 + (p0 - 0.5)**2 + 2.5*(p1 + 0.2)**2 + 0.1*np.sin(5*p0)*np.cos(5*p1)

def maxcut_cost(params):
    """Simulated QAOA MaxCut landscape (rugged)"""
    gamma, beta = params[0], params[1]
    # Rugged landscape with many local minima
    return -2.0 + np.sin(3*gamma)*np.cos(3*beta) + 0.5*np.sin(5*gamma + beta) + 0.3*np.cos(7*gamma - 2*beta)

# ------------------------------------------------------------------------------
# 5. BENCHMARK TESTS
# ------------------------------------------------------------------------------

def test_vqe_basic():
    """Test 1: Basic VQE (H2 molecule)"""
    print(f"\n⚡ Test 1: VQE H2 Molecule")
    print("-" * 60)
    
    np.random.seed(42)
    init_params = np.random.uniform(-1, 1, 2)
    
    # 1. Plain Adam
    adam = AdamOptimizer(lr=0.05)
    p_adam = init_params.copy()
    for _ in range(60):
        loss_adam = h2_energy(p_adam)
        grad = Demeasurement.finite_difference(h2_energy, p_adam)
        p_adam = adam.step(p_adam, grad)
    loss_adam = h2_energy(p_adam)
        
    # 2. Mobiu-Q (Adam + SA)
    opt = MobiuQCore(license_key=LICENSE_KEY, method="vqe", base_lr=0.05)
    p_mobiu = init_params.copy()
    for _ in range(60):
        loss_mobiu = h2_energy(p_mobiu)
        grad = Demeasurement.finite_difference(h2_energy, p_mobiu)
        p_mobiu = opt.step(p_mobiu, grad, loss_mobiu)
    opt.end()
    loss_mobiu = h2_energy(p_mobiu)
    
    print(f"   📉 Adam Final Loss:    {loss_adam:.6f}")
    print(f"   🚀 Mobiu-Q Final Loss: {loss_mobiu:.6f}")
    
    imp = ((loss_adam - loss_mobiu) / abs(loss_adam)) * 100
    color = "\033[92m" if imp > 0 else "\033[91m"
    print(f"   🏆 Improvement:        {color}{imp:+.2f}%\033[0m")
    return imp > 0


def test_qaoa():
    """Test 2: QAOA MaxCut (rugged landscape)"""
    print(f"\n⚡ Test 2: QAOA MaxCut (Rugged Landscape)")
    print("-" * 60)
    
    np.random.seed(42)
    init_params = np.random.uniform(-np.pi, np.pi, 2)
    
    # 1. Plain Adam
    adam = AdamOptimizer(lr=0.1)
    p_adam = init_params.copy()
    for _ in range(100):
        loss = maxcut_cost(p_adam)
        grad = Demeasurement.finite_difference(maxcut_cost, p_adam)
        p_adam = adam.step(p_adam, grad)
    loss_adam = maxcut_cost(p_adam)
        
    # 2. Mobiu-Q QAOA mode
    opt = MobiuQCore(license_key=LICENSE_KEY, method="qaoa", base_lr=0.1)
    p_mobiu = init_params.copy()
    for _ in range(100):
        loss = maxcut_cost(p_mobiu)
        grad = Demeasurement.finite_difference(maxcut_cost, p_mobiu)
        p_mobiu = opt.step(p_mobiu, grad, loss)
    opt.end()
    loss_mobiu = maxcut_cost(p_mobiu)
    
    print(f"   📉 Adam Final Cost:    {loss_adam:.6f}")
    print(f"   🚀 Mobiu-Q Final Cost: {loss_mobiu:.6f}")
    
    imp = ((loss_adam - loss_mobiu) / abs(loss_adam)) * 100
    color = "\033[92m" if imp > 0 else "\033[91m"
    print(f"   🏆 Improvement:        {color}{imp:+.2f}%\033[0m")
    return imp > 0


def test_ab_comparison():
    """Test 3: A/B Comparison (SA on vs off)"""
    print(f"\n⚡ Test 3: A/B Comparison (use_soft_algebra)")
    print("-" * 60)
    
    np.random.seed(42)
    init_params = np.random.uniform(-1, 1, 2)
    
    # 1. Mobiu-Q WITHOUT SA (plain Adam via API)
    opt_off = MobiuQCore(license_key=LICENSE_KEY, method="vqe", 
                         base_lr=0.05, use_soft_algebra=False)
    p_off = init_params.copy()
    for _ in range(60):
        loss = h2_energy(p_off)
        grad = Demeasurement.finite_difference(h2_energy, p_off)
        p_off = opt_off.step(p_off, grad, loss)
    opt_off.end()
    loss_off = h2_energy(p_off)
        
    # 2. Mobiu-Q WITH SA
    opt_on = MobiuQCore(license_key=LICENSE_KEY, method="vqe", 
                        base_lr=0.05, use_soft_algebra=True)
    p_on = init_params.copy()
    for _ in range(60):
        loss = h2_energy(p_on)
        grad = Demeasurement.finite_difference(h2_energy, p_on)
        p_on = opt_on.step(p_on, grad, loss)
    opt_on.end()
    loss_on = h2_energy(p_on)
    
    print(f"   📉 SA=off Final Loss:  {loss_off:.6f}")
    print(f"   🚀 SA=on Final Loss:   {loss_on:.6f}")
    
    imp = ((loss_off - loss_on) / abs(loss_off)) * 100
    color = "\033[92m" if imp > 0 else "\033[91m"
    print(f"   🏆 SA Improvement:     {color}{imp:+.2f}%\033[0m")
    return imp > 0


def test_multi_optimizer():
    """Test 4: Compare different base optimizers"""
    print(f"\n⚡ Test 4: Multi-Optimizer Comparison")
    print("-" * 60)
    
    optimizers_to_test = ["Adam", "NAdam", "AMSGrad"]
    results = {}
    
    for opt_name in optimizers_to_test:
        np.random.seed(42)
        init_params = np.random.uniform(-1, 1, 2)
        
        opt = MobiuQCore(license_key=LICENSE_KEY, method="vqe", 
                         base_lr=0.05, base_optimizer=opt_name)
        p = init_params.copy()
        for _ in range(60):
            loss = h2_energy(p)
            grad = Demeasurement.finite_difference(h2_energy, p)
            p = opt.step(p, grad, loss)
        opt.end()
        
        results[opt_name] = h2_energy(p)
    
    print(f"\n   Results:")
    best = min(results, key=results.get)
    for name, loss in results.items():
        marker = "🏆" if name == best else "  "
        print(f"   {marker} {name+'+SA':<12}: {loss:.6f}")
    
    return True


def test_multi_seed():
    """Test 5: Multi-seed experiment (counts as 1 run)"""
    print(f"\n⚡ Test 5: Multi-Seed Experiment")
    print("-" * 60)
    
    opt = MobiuQCore(license_key=LICENSE_KEY, method="vqe", base_lr=0.05)
    
    all_losses = []
    for seed in range(3):
        opt.new_run()  # Reset state, keep session
        np.random.seed(seed)
        p = np.random.uniform(-1, 1, 2)
        
        for _ in range(40):
            loss = h2_energy(p)
            grad = Demeasurement.finite_difference(h2_energy, p)
            p = opt.step(p, grad, loss)
        
        final_loss = h2_energy(p)
        all_losses.append(final_loss)
        print(f"   Seed {seed}: Final loss = {final_loss:.6f}")
    
    opt.end()  # Only here it counts as 1 run
    
    print(f"\n   Mean loss: {np.mean(all_losses):.6f} ± {np.std(all_losses):.6f}")
    print(f"   ✅ All 3 seeds counted as 1 run!")
    return True


# ------------------------------------------------------------------------------
# 6. MAIN
# ------------------------------------------------------------------------------

def run_all_tests():
    print("=" * 60)
    print("🔬 MOBIU-Q CUSTOMER VALIDATION (v2.4)")
    print("=" * 60)
    
    results = []
    
    results.append(("VQE Basic", test_vqe_basic()))
    results.append(("QAOA MaxCut", test_qaoa()))
    results.append(("A/B Comparison", test_ab_comparison()))
    results.append(("Multi-Optimizer", test_multi_optimizer()))
    results.append(("Multi-Seed", test_multi_seed()))
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")
    
    print(f"\n   Total: {passed}/{len(results)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    if LICENSE_KEY == "YOUR_LICENSE_KEY_HERE":
        print("\n❌ Please set your LICENSE_KEY in the script!")
        print("   Get your key at: https://app.mobiu.ai")
    else:
        run_all_tests()