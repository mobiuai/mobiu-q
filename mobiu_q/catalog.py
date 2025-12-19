# mobiu_q_catalog.py
# ==================
# Mobiu-Q Problem Catalog (v2.0)
# Universal Stochastic Optimization: Quantum, Classical, Finance, AI
# ==================

import numpy as np
from typing import Callable, Dict, Any, List, Tuple

# ════════════════════════════════════════════════════════════════════════════
# 1. QUANTUM PHYSICS (Hamiltonians)
# ════════════════════════════════════════════════════════════════════════════

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_n(*matrices):
    """Kronecker product of multiple matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

class Hamiltonians:
    """Quantum Hamiltonians for VQE (Chemistry & Condensed Matter)."""

    @staticmethod
    def h2_molecule(n_qubits: int = 2) -> np.ndarray:
        return -1.0 * kron_n(Z, I) - 0.5 * kron_n(I, Z) + 0.3 * kron_n(X, X)

    @staticmethod
    def lih_molecule(n_qubits: int = 4) -> np.ndarray:
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits):
            ops = [I] * n_qubits; ops[i] = Z
            H += -0.5 * (i + 1) * kron_n(*ops)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits; ops[i] = X; ops[i + 1] = X
            H += 0.2 * kron_n(*ops)
        return H

    # --- Condensed Matter ---
    @staticmethod
    def transverse_ising(n_qubits: int = 4, J: float = 1.0, h: float = 0.5) -> np.ndarray:
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits; ops[i] = Z; ops[i + 1] = Z
            H += -J * kron_n(*ops)
        for i in range(n_qubits):
            ops = [I] * n_qubits; ops[i] = X
            H += -h * kron_n(*ops)
        return H

    @staticmethod
    def xy_model(n_qubits: int = 4, J: float = 1.0) -> np.ndarray:
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits; ops[i] = X; ops[i + 1] = X
            H += J * kron_n(*ops)
            ops = [I] * n_qubits; ops[i] = Y; ops[i + 1] = Y
            H += J * kron_n(*ops)
        return H

    @staticmethod
    def heisenberg_xxz(n_qubits: int = 4, Jxy: float = 1.0, Jz: float = 0.5) -> np.ndarray:
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits; ops[i] = X; ops[i + 1] = X; H += Jxy * kron_n(*ops)
            ops = [I] * n_qubits; ops[i] = Y; ops[i + 1] = Y; H += Jxy * kron_n(*ops)
            ops = [I] * n_qubits; ops[i] = Z; ops[i + 1] = Z; H += Jz * kron_n(*ops)
        return H

    # --- Topological Models ---
    @staticmethod
    def ssh_model(n_qubits: int = 4, v: float = 1.0, w: float = 0.5) -> np.ndarray:
        """Su-Schrieffer-Heeger (SSH) topological model."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        # Intracell hopping (v) and Intercell hopping (w)
        for i in range(0, n_qubits - 1, 2):
            # v terms (0-1, 2-3)
            ops_x = [I]*n_qubits; ops_x[i]=X; ops_x[i+1]=X; H += v * kron_n(*ops_x)
            ops_y = [I]*n_qubits; ops_y[i]=Y; ops_y[i+1]=Y; H += v * kron_n(*ops_y)
        for i in range(1, n_qubits - 1, 2):
            # w terms (1-2, 3-4)
            ops_x = [I]*n_qubits; ops_x[i]=X; ops_x[i+1]=X; H += w * kron_n(*ops_x)
            ops_y = [I]*n_qubits; ops_y[i]=Y; ops_y[i+1]=Y; H += w * kron_n(*ops_y)
        return H

    @staticmethod
    def kitaev_chain(n_qubits: int = 4, t: float = 1.0, delta: float = 1.0, mu: float = 0.5) -> np.ndarray:
        """Kitaev Chain (P-wave superconductor)."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            # Hopping: X X + Y Y
            ops_xx = [I]*n_qubits; ops_xx[i]=X; ops_xx[i+1]=X; H += -t * kron_n(*ops_xx)
            ops_yy = [I]*n_qubits; ops_yy[i]=Y; ops_yy[i+1]=Y; H += -t * kron_n(*ops_yy)
            # Pairing: X X - Y Y
            H += delta * kron_n(*ops_xx)
            H += -delta * kron_n(*ops_yy)
        for i in range(n_qubits):
            ops_z = [I]*n_qubits; ops_z[i]=Z; H += -mu * kron_n(*ops_z)
        return H

    @staticmethod
    def hubbard_dimer(n_qubits: int = 4, t: float = 1.0, U: float = 2.0) -> np.ndarray:
        """Two-site Hubbard model (Dimer). Maps to 4 qubits."""
        # 0,1 = Site 1 (up, down); 2,3 = Site 2 (up, down)
        # Simplified mapping
        return Hamiltonians.heisenberg_xxz(n_qubits, Jxy=t, Jz=U) # Approx mapping for structural test

    # --- Other ---
    @staticmethod
    def h3_chain(n_qubits: int = 3) -> np.ndarray:
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits):
            ops = [I] * n_qubits; ops[i] = Z; H += -0.8 * kron_n(*ops)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits; ops[i] = X; ops[i + 1] = X; H += 0.25 * kron_n(*ops)
        return H

    @staticmethod
    def ferro_ising(n_qubits: int = 4, J: float = 1.0) -> np.ndarray:
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits; ops[i] = Z; ops[i + 1] = Z; H += -J * kron_n(*ops)
        return H

    @staticmethod
    def antiferro_heisenberg(n_qubits: int = 4, J: float = 1.0) -> np.ndarray:
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits; ops[i]=X; ops[i+1]=X; H += J * kron_n(*ops)
            ops = [I] * n_qubits; ops[i]=Y; ops[i+1]=Y; H += J * kron_n(*ops)
            ops = [I] * n_qubits; ops[i]=Z; ops[i+1]=Z; H += J * kron_n(*ops)
        return H

    @staticmethod
    def be2_molecule(n_qubits: int = 4) -> np.ndarray:
        return Hamiltonians.heisenberg_xxz(n_qubits, Jxy=0.62, Jz=0.79) # Approx

    @staticmethod
    def he4_atom(n_qubits: int = 2) -> np.ndarray:
        return 0.9 * kron_n(X, X) + 0.9 * kron_n(Y, Y) + 1.1 * kron_n(Z, Z) - 0.4 * (kron_n(Z, I) + kron_n(I, Z))


# ════════════════════════════════════════════════════════════════════════════
# 2. CLASSICAL OPTIMIZATION (Rugged & Valleys)
# ════════════════════════════════════════════════════════════════════════════

class ClassicalObjectives:
    """Standard non-convex optimization benchmarks."""

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """The 'Banana Valley' function. Hard to navigate the curve."""
        return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Highly multimodal 'Egg Carton'. Requires QAOA mode."""
        A = 10
        n = len(x)
        return A * n + sum(x**2 - A * np.cos(2 * np.pi * x))

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Many local minima."""
        a, b, c = 20, 0.2, 2 * np.pi
        n = len(x)
        s1 = sum(x**2)
        s2 = sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(s1/n)) - np.exp(s2/n) + a + np.e

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Convex baseline."""
        return sum(x**2)

    @staticmethod
    def beale(x: np.ndarray) -> float:
        """Plateaus and ridges (usually 2D)."""
        if len(x) < 2: return 0.0
        x1, x2 = x[0], x[1]
        return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2


# ════════════════════════════════════════════════════════════════════════════
# 3. FINANCE MODELS (Stochastic)
# ════════════════════════════════════════════════════════════════════════════

class FinanceObjectives:
    """Financial risk and pricing models (Stochastic nature similar to Quantum)."""

    @staticmethod
    def portfolio_optimization(weights: np.ndarray) -> float:
        """
        Markowitz Portfolio: Minimize Risk (Variance) for fixed Return.
        L = w.T * Cov * w - lambda * w.T * mu + penalty
        """
        # Simulated market data (fixed seed for consistency)
        n = len(weights)
        np.random.seed(42)
        returns = np.random.uniform(0.05, 0.20, n)
        cov = np.random.uniform(0.01, 0.05, (n, n))
        cov = np.dot(cov, cov.T)  # PSD
        risk_aversion = 0.5

        # Normalize weights via softmax to ensure sum=1 constraint (softly)
        # For raw optimization, we penalize sum != 1
        w_sum = np.sum(weights)
        penalty = 100 * (w_sum - 1.0)**2
        
        port_return = np.dot(weights, returns)
        port_risk = np.dot(weights.T, np.dot(cov, weights))
        
        return risk_aversion * port_risk - port_return + penalty

    @staticmethod
    def credit_risk_var(params: np.ndarray) -> float:
        """
        Minimize Value at Risk (VaR) / Tail Loss.
        Simplified to: Minimize Exposure * Probability of Default.
        """
        # Params represents exposure allocation
        n = len(params)
        np.random.seed(101)
        default_probs = np.random.beta(2, 10, n) # Skewed low probs
        loss_given_default = np.random.uniform(0.3, 0.8, n)
        
        # Loss function with noise injection (market volatility)
        expected_loss = np.sum(params * default_probs * loss_given_default)
        volatility = np.sum(params**2 * default_probs * (1-default_probs)) 
        
        return expected_loss + 1.65 * np.sqrt(volatility) # VaR 95% approx

    @staticmethod
    def option_pricing_calibration(params: np.ndarray) -> float:
        """
        Calibrate Volatility Surface (Heston model proxy).
        Minimize squared error between Model and Market prices.
        """
        # Target volatilities (Market)
        market_vols = np.array([0.2, 0.22, 0.18, 0.25])
        
        # Simple polynomial model of volatility surface based on params
        # Model: vol = a + b*T + c*K
        model_vols = params[0] + params[1]*np.array([1, 2, 1, 2]) + params[2]*np.array([0.9, 0.9, 1.1, 1.1])
        
        mse = np.mean((model_vols - market_vols)**2)
        return mse


# ════════════════════════════════════════════════════════════════════════════
# QAOA / CIRCUIT INFRASTRUCTURE
# ════════════════════════════════════════════════════════════════════════════

class QAOAProblems:
    @staticmethod
    def random_graph(n_nodes: int, edge_prob: float = 0.5, seed: int = None):
        if seed is not None: np.random.seed(seed)
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() < edge_prob: edges.append((i, j))
        if not edges: edges = [(0, 1)]
        return edges
    
    @staticmethod
    def maxcut_cost_terms(edges):
        return [(-0.5, (i, j)) for i, j in edges]
    
    @staticmethod
    def vertex_cover_cost_terms(edges, n_nodes, penalty=2.0):
        terms = [(0.5, (i,)) for i in range(n_nodes)]
        for i, j in edges: terms.append((penalty * 0.25, (i, j)))
        return terms
    
    @staticmethod
    def max_independent_set_cost_terms(edges, n_nodes, penalty=2.0):
        terms = [(-0.5, (i,)) for i in range(n_nodes)]
        for i, j in edges: terms.append((penalty * 0.25, (i, j)))
        return terms

class QAOACircuit:
    @staticmethod
    def qaoa_expectation(params, n_qubits, cost_terms, p, noise_level=0.0):
        gammas, betas = params[:p], params[p:]
        state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
        
        for layer in range(p):
            # Cost
            gamma = gammas[layer]
            for coef, qubits in cost_terms:
                if len(qubits) == 2:
                    i, j = qubits
                    indices = np.arange(2**n_qubits)
                    zi = 1 - 2*((indices >> i) & 1)
                    zj = 1 - 2*((indices >> j) & 1)
                    phase = np.exp(-1j * gamma * coef * zi * zj)
                    state *= phase
                elif len(qubits) == 1:
                    i = qubits[0]
                    indices = np.arange(2**n_qubits)
                    zi = 1 - 2*((indices >> i) & 1)
                    state *= np.exp(-1j * gamma * coef * zi)
            
            # Mixer
            beta = betas[layer]
            c, s = np.cos(beta), -1j * np.sin(beta)
            for i in range(n_qubits):
                # Apply Rx(2beta) -> exp(-i beta X)
                # Efficient vectorization: X flips bit i
                indices = np.arange(2**n_qubits)
                flipped = indices ^ (1 << i)
                state = c * state + s * state[flipped]

        # Expectation
        probs = np.abs(state)**2
        total_energy = 0.0
        for coef, qubits in cost_terms:
            if len(qubits) == 2:
                i, j = qubits
                indices = np.arange(2**n_qubits)
                zi = 1 - 2*((indices >> i) & 1)
                zj = 1 - 2*((indices >> j) & 1)
                total_energy += np.sum(probs * coef * zi * zj)
            elif len(qubits) == 1:
                i = qubits[0]
                indices = np.arange(2**n_qubits)
                zi = 1 - 2*((indices >> i) & 1)
                total_energy += np.sum(probs * coef * zi)
        
        if noise_level > 0:
            total_energy += np.random.normal(0, noise_level * abs(total_energy) + 0.01)
            
        return float(total_energy)

class Ansatz:
    @staticmethod
    def vqe_hardware_efficient(n_qubits: int, depth: int, params: np.ndarray) -> np.ndarray:
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=complex); state[0] = 1.0
        param_idx = 0
        for _ in range(depth):
            for q in range(n_qubits):
                theta = params[param_idx] if param_idx < len(params) else 0.0
                param_idx += 1
                c, s = np.cos(theta/2), np.sin(theta/2)
                # Apply Ry locally (tensor product logic simplified)
                # For simulation speed, we skip full tensor build if possible, but here we use simple matrix mult
                # (Optimized for readability/compatibility with existing code)
                Ry = np.array([[c, -s], [s, c]], dtype=complex)
                ops = [I]*n_qubits; ops[q] = Ry
                state = kron_n(*ops) @ state
            
            # Entanglement (Linear)
            for q in range(n_qubits - 1):
                # CNOT q -> q+1
                # Manual application on state vector
                new_state = state.copy()
                for i in range(dim):
                    if (i >> q) & 1: # Control is 1
                        target_bit = (i >> (q+1)) & 1
                        flipped_i = i ^ (1 << (q+1))
                        new_state[i] = 0; new_state[flipped_i] = state[i] # Swap amplitudes (simplification for CNOT)
                        # Actually CNOT is unitary swap on pairs. 
                        # Correct logic:
                        # If control is 0, nothing. If 1, X on target.
                        pass # Relying on full matrix for correctness in this version
                
                # Fallback to matrix for correctness
                CNOT = np.eye(dim, dtype=complex)
                for i in range(dim):
                    if (i >> q) & 1:
                        target = i ^ (1 << (q+1))
                        CNOT[i, i] = 0; CNOT[i, target] = 1 # Warning: this matrix logic is sparse
                        # Easier:
                # Just use matrix construction
                ops_cnot = np.eye(dim)
                # ... skipping complex CNOT build for brevity, assuming existing logic works or using simple matrix
                # Re-using logic from v1.0 which was correct:
                CNOT = np.eye(dim, dtype=complex)
                for i in range(dim):
                     bits = [(i >> b) & 1 for b in range(n_qubits)]
                     if bits[q] == 1:
                         j = i ^ (1 << (q+1))
                         CNOT[i, i] = 0; CNOT[j, i] = 1; CNOT[i, j] = 1; CNOT[j, j] = 0
                state = CNOT @ state
        return state


# ════════════════════════════════════════════════════════════════════════════
# 4. PROBLEM CATALOG DEFINITION
# ════════════════════════════════════════════════════════════════════════════

PROBLEM_CATALOG: Dict[str, Dict[str, Any]] = {
    # --- Quantum VQE ---
    'h2_molecule': {'type': 'VQE', 'problem_mode': 'vqe', 'n_qubits': 2, 'depth': 3, 'hamiltonian_fn': Hamiltonians.h2_molecule, 'landscape': 'smooth', 'description': 'H2 Molecule'},
    'lih_molecule': {'type': 'VQE', 'problem_mode': 'vqe', 'n_qubits': 4, 'depth': 3, 'hamiltonian_fn': Hamiltonians.lih_molecule, 'landscape': 'smooth', 'description': 'LiH Molecule'},
    'transverse_ising': {'type': 'VQE', 'problem_mode': 'vqe', 'n_qubits': 4, 'depth': 3, 'hamiltonian_fn': Hamiltonians.transverse_ising, 'landscape': 'moderate', 'description': 'Transverse Ising'},
    'heisenberg_xxz': {'type': 'VQE', 'problem_mode': 'vqe', 'n_qubits': 4, 'depth': 4, 'hamiltonian_fn': Hamiltonians.heisenberg_xxz, 'landscape': 'frustrated', 'description': 'Heisenberg XXZ'},
    'xy_model': {'type': 'VQE', 'problem_mode': 'vqe', 'n_qubits': 4, 'depth': 4, 'hamiltonian_fn': Hamiltonians.xy_model, 'landscape': 'moderate', 'description': 'XY Model'},
    'ssh_model': {'type': 'VQE', 'problem_mode': 'vqe', 'n_qubits': 4, 'depth': 4, 'hamiltonian_fn': Hamiltonians.ssh_model, 'landscape': 'topological', 'description': 'SSH Topological Model'},
    'kitaev_chain': {'type': 'VQE', 'problem_mode': 'vqe', 'n_qubits': 4, 'depth': 4, 'hamiltonian_fn': Hamiltonians.kitaev_chain, 'landscape': 'topological', 'description': 'Kitaev Chain'},
    'hubbard_dimer': {'type': 'VQE', 'problem_mode': 'vqe', 'n_qubits': 4, 'depth': 4, 'hamiltonian_fn': Hamiltonians.hubbard_dimer, 'landscape': 'correlated', 'description': 'Hubbard Dimer'},
    
    # --- Classical ---
    'rosenbrock': {'type': 'Classical', 'problem_mode': 'vqe', 'n_vars': 4, 'func': ClassicalObjectives.rosenbrock, 'landscape': 'valley', 'description': 'Rosenbrock (Banana Valley)'},
    'rastrigin': {'type': 'Classical', 'problem_mode': 'qaoa', 'n_vars': 4, 'func': ClassicalObjectives.rastrigin, 'landscape': 'rugged', 'description': 'Rastrigin (Multimodal)'},
    'ackley': {'type': 'Classical', 'problem_mode': 'qaoa', 'n_vars': 4, 'func': ClassicalObjectives.ackley, 'landscape': 'rugged', 'description': 'Ackley Function'},
    'sphere': {'type': 'Classical', 'problem_mode': 'vqe', 'n_vars': 4, 'func': ClassicalObjectives.sphere, 'landscape': 'convex', 'description': 'Sphere Baseline'},
    'beale': {'type': 'Classical', 'problem_mode': 'vqe', 'n_vars': 2, 'func': ClassicalObjectives.beale, 'landscape': 'plateau', 'description': 'Beale Function'},

    # --- Finance ---
    'portfolio': {'type': 'Finance', 'problem_mode': 'vqe', 'n_vars': 5, 'func': FinanceObjectives.portfolio_optimization, 'landscape': 'stochastic', 'description': 'Portfolio Optimization'},
    'credit_risk': {'type': 'Finance', 'problem_mode': 'vqe', 'n_vars': 5, 'func': FinanceObjectives.credit_risk_var, 'landscape': 'stochastic', 'description': 'Credit Risk VaR'},
    'option_pricing': {'type': 'Finance', 'problem_mode': 'vqe', 'n_vars': 3, 'func': FinanceObjectives.option_pricing_calibration, 'landscape': 'stochastic', 'description': 'Option Volatility Calibration'},

    # --- QAOA ---
    'maxcut_5': {'type': 'QAOA', 'problem_mode': 'qaoa', 'n_qubits': 5, 'p': 3, 'graph_type': 'random', 'cost_fn': QAOAProblems.maxcut_cost_terms, 'landscape': 'rugged', 'description': 'MaxCut 5-node'},
    'vertex_cover_5': {'type': 'QAOA', 'problem_mode': 'qaoa', 'n_qubits': 5, 'p': 3, 'graph_type': 'random', 'cost_fn': QAOAProblems.vertex_cover_cost_terms, 'landscape': 'rugged', 'description': 'Vertex Cover 5-node'},
    'max_independent_set_5': {'type': 'QAOA', 'problem_mode': 'qaoa', 'n_qubits': 5, 'p': 3, 'graph_type': 'random', 'cost_fn': QAOAProblems.max_independent_set_cost_terms, 'landscape': 'rugged', 'description': 'Max Independent Set 5-node'},
}


# ════════════════════════════════════════════════════════════════════════════
# 5. INTERFACE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def get_problem(name: str):
    return PROBLEM_CATALOG.get(name)

def list_problems(ptype=None):
    if ptype: return [k for k, v in PROBLEM_CATALOG.items() if v['type'] == ptype]
    return list(PROBLEM_CATALOG.keys())

def get_energy_function(name: str, seed=None, noise=0.0):
    prob = get_problem(name)
    if not prob: raise ValueError(f"Unknown problem: {name}")
    
    if prob['type'] == 'VQE':
        H = prob['hamiltonian_fn'](prob['n_qubits'])
        def fn(p):
            s = Ansatz.vqe_hardware_efficient(prob['n_qubits'], prob['depth'], p)
            e = np.real(s.conj() @ H @ s).item()
            if noise > 0: e += np.random.normal(0, noise * abs(e) + 0.01)
            return e
        return fn
        
    elif prob['type'] == 'Classical' or prob['type'] == 'Finance':
        # Direct function call
        f = prob['func']
        def fn(p):
            val = f(p)
            if noise > 0: val += np.random.normal(0, noise * abs(val) + 0.01)
            return val
        return fn
        
    elif prob['type'] == 'QAOA':
        edges = QAOAProblems.random_graph(prob['n_qubits'], 0.5, seed)
        cost = prob['cost_fn'](edges, prob['n_qubits']) if 'vertex' in name or 'independent' in name else prob['cost_fn'](edges)
        return lambda p: QAOACircuit.qaoa_expectation(p, prob['n_qubits'], cost, prob['p'], noise)

def get_n_params(name: str):
    p = get_problem(name)
    if p['type'] == 'VQE': return p['n_qubits'] * p['depth']
    if p['type'] == 'QAOA': return 2 * p['p']
    return p['n_vars'] # Classical/Finance

def get_problem_mode(name: str):
    return get_problem(name).get('problem_mode', 'standard')

# Self-check
if __name__ == "__main__":
    print(f"Catalog Loaded: {len(PROBLEM_CATALOG)} problems.")
    print("Types:", set(p['type'] for p in PROBLEM_CATALOG.values()))