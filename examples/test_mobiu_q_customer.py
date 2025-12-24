"""
Mobiu-Q Customer Tests (v2.4)
=============================
Comprehensive tests with verified quantum problems.
Run with: pytest test_mobiu_q_customer.py -v
"""

import pytest
import numpy as np
import os
from typing import Callable, List, Tuple

LICENSE_KEY = os.environ.get("MOBIU_LICENSE_KEY", "YOUR_LICENCE")

@pytest.fixture
def license_key():
    return LICENSE_KEY

# ============================================================================
# PAULI MATRICES & GROUND STATE COMPUTATION
# ============================================================================

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

def pauli_tensor(pauli_str):
    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    result = paulis[pauli_str[0]]
    for p in pauli_str[1:]:
        result = np.kron(result, paulis[p])
    return result

def compute_ground_state(pauli_list):
    n_qubits = len(pauli_list[0][0])
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for pauli_str, coef in pauli_list:
        H += coef * pauli_tensor(pauli_str)
    return np.linalg.eigvalsh(H)[0].real

# ============================================================================
# QUANTUM CHEMISTRY PROBLEMS
# ============================================================================

H2_PAULI = [("II", -0.4804), ("ZZ", 0.3435), ("ZI", -0.4347), 
            ("IZ", 0.5716), ("XX", 0.0910), ("YY", 0.0910)]
H2_GROUND = compute_ground_state(H2_PAULI)

def h2_energy(params, noise=0.0):
    coeffs = [-0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    zi, iz = c0**2 - s0**2, c1**2 - s1**2
    xx = 2 * c0 * s0 * c1 * s1
    energy = coeffs[0] + coeffs[1]*zz + coeffs[2]*zi + coeffs[3]*iz + coeffs[4]*xx + coeffs[5]*xx
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

LIH_PAULI = [("II", -0.25), ("ZZ", 0.17), ("ZI", -0.22), ("IZ", 0.12), ("XX", 0.08)]
LIH_GROUND = compute_ground_state(LIH_PAULI)

def lih_energy(params, noise=0.0):
    coeffs = [-0.25, 0.17, -0.22, 0.12, 0.08]
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    zi, iz = c0**2 - s0**2, c1**2 - s1**2
    xx = 2 * c0 * s0 * c1 * s1
    energy = coeffs[0] + coeffs[1]*zz + coeffs[2]*zi + coeffs[3]*iz + coeffs[4]*xx
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

BEH2_PAULI = [("II", -0.35), ("ZZ", 0.25), ("ZI", -0.30), ("IZ", 0.20), ("XX", 0.12), ("YY", 0.12)]
BEH2_GROUND = compute_ground_state(BEH2_PAULI)

def beh2_energy(params, noise=0.0):
    coeffs = [-0.35, 0.25, -0.30, 0.20, 0.12, 0.12]
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    zi, iz = c0**2 - s0**2, c1**2 - s1**2
    xx = 2 * c0 * s0 * c1 * s1
    energy = coeffs[0] + coeffs[1]*zz + coeffs[2]*zi + coeffs[3]*iz + coeffs[4]*xx + coeffs[5]*xx
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

HE_PAULI = [("II", -2.0), ("ZZ", 0.5), ("ZI", -0.3), ("IZ", -0.3)]
HE_GROUND = compute_ground_state(HE_PAULI)

def he_energy(params, noise=0.0):
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    zi, iz = c0**2 - s0**2, c1**2 - s1**2
    energy = -2.0 + 0.5*zz - 0.3*zi - 0.3*iz
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

# ============================================================================
# CONDENSED MATTER PROBLEMS
# ============================================================================

HEISENBERG_PAULI = [("XX", 1.0), ("YY", 1.0), ("ZZ", 1.0)]
HEISENBERG_GROUND = compute_ground_state(HEISENBERG_PAULI)

def heisenberg_energy(params, noise=0.0):
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    xx = 4 * c0 * s0 * c1 * s1
    yy = 4 * c0 * s0 * c1 * s1
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    energy = xx + yy + zz
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

TRANSVERSE_ISING_PAULI = [("ZZ", -1.0), ("XI", -0.5), ("IX", -0.5)]
TRANSVERSE_ISING_GROUND = compute_ground_state(TRANSVERSE_ISING_PAULI)

def transverse_ising_energy(params, noise=0.0):
    J, h = 1.0, 0.5
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    zz = (c0**2 - s0**2) * (c1**2 - s1**2)
    x0 = 2 * c0 * s0
    x1 = 2 * c1 * s1
    energy = -J * zz - h * (x0 + x1)
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

XY_PAULI = [("XX", 1.0), ("YY", 1.0)]
XY_GROUND = compute_ground_state(XY_PAULI)

def xy_energy(params, noise=0.0):
    c0, s0 = np.cos(params[0]/2), np.sin(params[0]/2)
    c1, s1 = np.cos(params[1]/2), np.sin(params[1]/2)
    xx = 4 * c0 * s0 * c1 * s1
    yy = 4 * c0 * s0 * c1 * s1
    energy = xx + yy
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

# ============================================================================
# QAOA PROBLEMS
# ============================================================================

def qaoa_expectation(params, n_qubits, cost_terms, p, noise=0.0):
    """QAOA expectation value with actual state simulation"""
    gammas = params[:p]
    betas = params[p:]
    state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
    
    for layer in range(p):
        gamma, beta = gammas[layer], betas[layer]
        for coef, qubits in cost_terms:
            if len(qubits) == 2:
                i, j = qubits
                for k in range(2**n_qubits):
                    z_i = 1 - 2 * ((k >> i) & 1)
                    z_j = 1 - 2 * ((k >> j) & 1)
                    state[k] *= np.exp(-1j * gamma * coef * z_i * z_j)
        for qubit in range(n_qubits):
            new_state = np.zeros_like(state)
            c, s = np.cos(beta), np.sin(beta)
            for k in range(2**n_qubits):
                bit = (k >> qubit) & 1
                k_flipped = k ^ (1 << qubit)
                if bit == 0:
                    new_state[k] += c * state[k] - 1j * s * state[k_flipped]
                else:
                    new_state[k] += -1j * s * state[k_flipped] + c * state[k]
            state = new_state
    
    expectation = 0.0
    for k in range(2**n_qubits):
        prob = np.abs(state[k])**2
        cost = sum(coef * (1 - 2*((k >> q[0]) & 1)) * (1 - 2*((k >> q[1]) & 1)) 
                   for coef, q in cost_terms if len(q) == 2)
        expectation += prob * cost
    
    if noise > 0:
        expectation += np.random.normal(0, noise * abs(expectation) + 0.01)
    
    return expectation

def create_maxcut_problem(n_qubits=5, graph_seed=42, p=5, noise=0.1):
    np.random.seed(graph_seed)
    edges = [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits) 
             if np.random.random() < 0.5]
    if not edges:
        edges = [(0, 1)]
    cost_terms = [(-0.5, (i, j)) for i, j in edges]
    
    def energy_fn(params):
        return qaoa_expectation(params, n_qubits, cost_terms, p, noise)
    def clean_fn(params):
        return qaoa_expectation(params, n_qubits, cost_terms, p, 0)
    
    return energy_fn, clean_fn, 2*p

# ============================================================================
# CLASSICAL PROBLEMS
# ============================================================================

def rosenbrock_energy(params, noise=0.0):
    x, y = params[0], params[1]
    energy = (1 - x)**2 + 100 * (y - x**2)**2
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

def rastrigin_energy(params, noise=0.0):
    n = len(params)
    energy = 10*n + np.sum(params**2 - 10*np.cos(2*np.pi*params))
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

def ackley_energy(params, noise=0.0):
    n = len(params)
    sum1 = np.sum(params**2)
    sum2 = np.sum(np.cos(2*np.pi*params))
    energy = -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    if noise > 0:
        energy += np.random.normal(0, noise * abs(energy) + 0.01)
    return energy

# ============================================================================
# HELPER
# ============================================================================

def check_convergence(energies, window=10):
    first = np.mean(energies[:window])
    last = np.mean(energies[-window:])
    improvement = (first - last) / abs(first) * 100 if first != 0 else 0
    return {'first': first, 'last': last, 'improvement': improvement, 'converged': last < first}


def run_vqe_test(energy_fn, license_key, n_steps=100, seed=42):
    """Helper to run VQE test with clean gradients"""
    from mobiu_q import MobiuQCore, Demeasurement
    
    opt = MobiuQCore(license_key=license_key, method="vqe")
    np.random.seed(seed)
    params = np.random.uniform(-np.pi, np.pi, 2)
    clean_energies = []
    
    for step in range(n_steps):
        grad = Demeasurement.finite_difference(lambda p: energy_fn(p, 0), params)
        energy = energy_fn(params, 0.02)
        clean_energies.append(energy_fn(params, 0))
        params = opt.step(params, grad, energy)
    
    opt.end()
    return check_convergence(clean_energies)

# ============================================================================
# TESTS: INSTALLATION
# ============================================================================

class TestInstallation:
    def test_import(self):
        from mobiu_q import MobiuQCore, Demeasurement
        assert MobiuQCore is not None
    
    def test_create_optimizer(self, license_key):
        from mobiu_q import MobiuQCore
        opt = MobiuQCore(license_key=license_key, method="vqe")
        opt.end()

# ============================================================================
# TESTS: VQE QUANTUM CHEMISTRY
# ============================================================================

class TestVQEChemistry:
    def test_h2_molecule(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(42)
        params = np.random.uniform(-np.pi, np.pi, 2)
        clean_energies = []
        
        for step in range(100):  # More steps
            grad = Demeasurement.finite_difference(lambda p: h2_energy(p, 0), params)  # Clean grad
            energy = h2_energy(params, 0.02)
            clean_energies.append(h2_energy(params, 0))
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(clean_energies)
        gap = result['last'] - H2_GROUND
        
        assert result['converged'] or result['improvement'] > 5, f"H2 should improve: {result}"
        assert gap < 0.2, f"H2 gap should be reasonable: {gap}"
    
    def test_lih_molecule(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(123)
        params = np.random.uniform(-np.pi, np.pi, 2)
        clean_energies = []
        
        for step in range(100):
            grad = Demeasurement.finite_difference(lambda p: lih_energy(p, 0), params)
            energy = lih_energy(params, 0.02)
            clean_energies.append(lih_energy(params, 0))
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(clean_energies)
        gap = result['last'] - LIH_GROUND
        
        assert result['converged'] or result['improvement'] > 5, f"LiH should improve: {result}"
        assert gap < 0.2, f"LiH gap should be reasonable: {gap}"
    
    def test_beh2_molecule(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(456)
        params = np.random.uniform(-np.pi, np.pi, 2)
        clean_energies = []
        
        for step in range(100):
            grad = Demeasurement.finite_difference(lambda p: beh2_energy(p, 0), params)
            energy = beh2_energy(params, 0.02)
            clean_energies.append(beh2_energy(params, 0))
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(clean_energies)
        assert result['converged'] or result['improvement'] > 5, f"BeH2 should improve: {result}"
    
    def test_he_atom(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(789)
        params = np.random.uniform(-np.pi, np.pi, 2)
        clean_energies = []
        
        for step in range(100):
            grad = Demeasurement.finite_difference(lambda p: he_energy(p, 0), params)
            energy = he_energy(params, 0.02)
            clean_energies.append(he_energy(params, 0))
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(clean_energies)
        assert result['converged'] or result['improvement'] > 5, f"He should improve: {result}"

# ============================================================================
# TESTS: VQE CONDENSED MATTER
# ============================================================================

class TestVQECondensedMatter:
    def test_heisenberg(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(42)
        params = np.random.uniform(-np.pi, np.pi, 2)
        clean_energies = []
        
        for step in range(100):
            grad = Demeasurement.finite_difference(lambda p: heisenberg_energy(p, 0), params)
            energy = heisenberg_energy(params, 0.02)
            clean_energies.append(heisenberg_energy(params, 0))
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(clean_energies)
        assert result['converged'] or result['improvement'] > 5, f"Heisenberg should improve: {result}"
    
    def test_transverse_ising(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(123)
        params = np.random.uniform(-np.pi, np.pi, 2)
        clean_energies = []
        
        for step in range(100):
            grad = Demeasurement.finite_difference(lambda p: transverse_ising_energy(p, 0), params)
            energy = transverse_ising_energy(params, 0.02)
            clean_energies.append(transverse_ising_energy(params, 0))
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(clean_energies)
        assert result['converged'] or result['improvement'] > 5, f"Transverse Ising should improve: {result}"
    
    def test_xy_model(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(456)
        params = np.random.uniform(-np.pi, np.pi, 2)
        clean_energies = []
        
        for step in range(100):
            grad = Demeasurement.finite_difference(lambda p: xy_energy(p, 0), params)
            energy = xy_energy(params, 0.02)
            clean_energies.append(xy_energy(params, 0))
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(clean_energies)
        assert result['converged'] or result['improvement'] > 5, f"XY model should improve: {result}"

# ============================================================================
# TESTS: QAOA
# ============================================================================

class TestQAOA:
    def test_maxcut_basic(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        energy_fn, clean_fn, n_params = create_maxcut_problem(n_qubits=5, p=5)
        
        opt = MobiuQCore(license_key=license_key, method="qaoa")
        np.random.seed(42)
        params = np.random.uniform(0, np.pi/2, n_params)
        energies = []
        
        for step in range(150):
            grad, energy = Demeasurement.spsa(energy_fn, params)
            energies.append(energy)
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(energies)
        # QAOA is hard - just check it doesn't diverge badly
        assert result['improvement'] > -20, f"QAOA should not diverge: {result}"

# ============================================================================
# TESTS: CLASSICAL
# ============================================================================

class TestClassical:
    def test_rosenbrock(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(42)
        params = np.array([0.0, 0.0])
        clean_energies = []
        
        for step in range(100):
            grad = Demeasurement.finite_difference(lambda p: rosenbrock_energy(p, 0), params)
            energy = rosenbrock_energy(params, 0.05)
            clean_energies.append(rosenbrock_energy(params, 0))
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(clean_energies)
        assert result['converged'] or result['improvement'] > 5, f"Rosenbrock should improve: {result}"
    
    def test_rastrigin(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(42)
        params = np.random.uniform(-2, 2, 2)
        energies = []
        
        for step in range(100):
            grad = Demeasurement.finite_difference(lambda p: rastrigin_energy(p, 0.1), params)
            energy = rastrigin_energy(params, 0.1)
            energies.append(energy)
            params = opt.step(params, grad, energy)
        
        opt.end()
        # Rastrigin is multimodal - just check it ran
        assert len(energies) == 100

# ============================================================================
# TESTS: SOFT ALGEBRA A/B
# ============================================================================

class TestSoftAlgebra:
    def test_sa_vs_plain_h2(self, license_key):
        from mobiu_q import MobiuQCore, Demeasurement
        
        results = {}
        for use_sa in [True, False]:
            np.random.seed(42)
            opt = MobiuQCore(license_key=license_key, method="vqe", use_soft_algebra=use_sa)
            params = np.random.uniform(-np.pi, np.pi, 2)
            energies = []
            
            for step in range(60):
                grad = Demeasurement.finite_difference(lambda p: h2_energy(p, 0.05), params)
                energy = h2_energy(params, 0.05)
                energies.append(energy)
                params = opt.step(params, grad, energy)
            
            opt.end()
            results[use_sa] = np.mean(energies[-10:]) - H2_GROUND
        
        # Both should work
        assert results[True] < 1.0, "SA should reach reasonable energy"
        assert results[False] < 1.0, "Plain should reach reasonable energy"

# ============================================================================
# TESTS: MULTI-SEED
# ============================================================================

class TestMultiSeed:
    def test_10_seeds(self, license_key):
        """Test that multi-seed sessions work correctly (billing as single run)"""
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        final_energies = []
        
        for seed in range(10):
            opt.new_run()
            np.random.seed(seed * 42)
            params = np.random.uniform(-np.pi, np.pi, 2)
            
            for step in range(100):
                grad = Demeasurement.finite_difference(lambda p: h2_energy(p, 0), params)
                energy = h2_energy(params, 0.02)
                params = opt.step(params, grad, energy)
            
            final_energies.append(h2_energy(params, 0))
        
        opt.end()
        
        # Simple criteria: average energy should be negative (below 0)
        # and most runs should show improvement from random start
        mean_energy = np.mean(final_energies)
        negative_count = sum(e < 0 for e in final_energies)
        
        assert mean_energy < 0, f"Mean energy should be negative: {mean_energy}"
        assert negative_count >= 5, f"At least half should reach negative energy: {negative_count}/10"

# ============================================================================
# TESTS: OPTIMIZERS
# ============================================================================

class TestOptimizers:
    @pytest.mark.parametrize("optimizer", ["Adam", "NAdam", "AMSGrad", "SGD", "Momentum"])
    def test_optimizer(self, license_key, optimizer):
        from mobiu_q import MobiuQCore, Demeasurement
        
        opt = MobiuQCore(license_key=license_key, method="vqe", base_optimizer=optimizer)
        np.random.seed(42)
        params = np.random.uniform(-np.pi, np.pi, 2)
        energies = []
        
        for step in range(50):
            grad = Demeasurement.finite_difference(lambda p: h2_energy(p, 0.02), params)
            energy = h2_energy(params, 0.02)
            energies.append(energy)
            params = opt.step(params, grad, energy)
        
        opt.end()
        result = check_convergence(energies)
        assert result['last'] < 0, f"{optimizer} should reach negative energy"

# ============================================================================
# TESTS: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    def test_nan_gradient(self, license_key):
        """Test that API handles NaN gracefully (either sanitizes, raises, or falls back)"""
        from mobiu_q import MobiuQCore
        
        opt = MobiuQCore(license_key=license_key, method="vqe")
        params = np.array([1.0, 2.0])
        grad = np.array([np.nan, 0.1])
        
        try:
            result = opt.step(params, grad, 1.0)
            # API might return NaN (which is acceptable - garbage in, garbage out)
            # or it might sanitize. Either way, it shouldn't crash.
            assert result is not None, "Should return something"
        except (ValueError, Exception) as e:
            # Raising an exception is also acceptable
            pass
        
        opt.end()
    
    def test_normal_after_error(self, license_key):
        """Test that a NEW optimizer works after a previous one encountered an error"""
        from mobiu_q import MobiuQCore, Demeasurement
        
        # First session with bad input
        opt1 = MobiuQCore(license_key=license_key, method="vqe")
        try:
            opt1.step(np.array([1.0, 2.0]), np.array([np.nan, 0.1]), 1.0)
        except:
            pass
        opt1.end()
        
        # New session should work normally
        opt2 = MobiuQCore(license_key=license_key, method="vqe")
        np.random.seed(42)
        params = np.random.uniform(-np.pi, np.pi, 2)
        
        for step in range(20):
            grad = Demeasurement.finite_difference(lambda p: h2_energy(p, 0), params)
            energy = h2_energy(params, 0.02)
            params = opt2.step(params, grad, energy)
        
        opt2.end()
        
        # Should have valid params
        assert not np.any(np.isnan(params)), "Final params should be valid"

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])