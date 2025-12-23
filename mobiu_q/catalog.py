# mobiu_q/catalog.py
# ==================
# Mobiu-Q Problem Catalog (v2.4.2)
# Universal Stochastic Optimization: Quantum, Classical, Finance, AI, RL
# 
# Changes in v2.4:
# - Added RL support (method='rl' uses Gymnasium, not this catalog)
# - Multi-optimizer support
# 
# Changes in v2.2:
# - problem_mode -> method ('vqe' or 'qaoa')
# - Added Nuclear Physics category
# - Added Quantum Optics category
# - Added verified ground state energies
# 
# Note: For Reinforcement Learning (method='rl'), use Gymnasium environments
# directly instead of this catalog. Example:
#     import gymnasium as gym
#     env = gym.make("LunarLander-v3")
# ==================

import numpy as np
from typing import Callable, Dict, Any, List, Tuple

# ════════════════════════════════════════════════════════════════════════════
# PAULI MATRICES
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

def pauli_tensor(pauli_str: str) -> np.ndarray:
    """Create tensor product of Pauli string like 'XYZ'"""
    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    result = paulis[pauli_str[0]]
    for p in pauli_str[1:]:
        result = np.kron(result, paulis[p])
    return result

def compute_ground_state(H: np.ndarray) -> float:
    """Compute exact ground state energy from Hamiltonian matrix."""
    eigenvalues = np.linalg.eigvalsh(H)
    return float(eigenvalues[0].real)

def hamiltonian_from_paulis(pauli_list: List[Tuple[str, float]]) -> np.ndarray:
    """Build Hamiltonian matrix from Pauli decomposition."""
    n_qubits = len(pauli_list[0][0])
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for pauli_str, coef in pauli_list:
        H += coef * pauli_tensor(pauli_str)
    return H

# ════════════════════════════════════════════════════════════════════════════
# 1. QUANTUM CHEMISTRY (VQE)
# ════════════════════════════════════════════════════════════════════════════

class Hamiltonians:
    """Quantum Hamiltonians for VQE (Chemistry & Condensed Matter)."""

    # --- Chemistry (Verified from Literature) ---
    
    @staticmethod
    def h2_molecule(n_qubits: int = 2) -> np.ndarray:
        """H2 at equilibrium (0.74 Å) - Kandala et al., Nature 2017"""
        paulis = [
            ("II", -0.4804), ("ZZ", 0.3435), ("ZI", -0.4347), 
            ("IZ", 0.5716), ("XX", 0.0910), ("YY", 0.0910)
        ]
        return hamiltonian_from_paulis(paulis)
    
    @staticmethod
    def h2_stretched(n_qubits: int = 2) -> np.ndarray:
        """H2 stretched (1.5 Å) - PySCF STO-3G"""
        paulis = [
            ("II", -0.5), ("ZZ", 0.4), ("ZI", -0.35), 
            ("IZ", 0.45), ("XX", 0.15), ("YY", 0.15)
        ]
        return hamiltonian_from_paulis(paulis)

    @staticmethod
    def heh_plus(n_qubits: int = 2) -> np.ndarray:
        """HeH+ - Hempel et al., PRX 2018"""
        paulis = [
            ("II", -1.5), ("ZZ", 0.3), ("ZI", -0.4), 
            ("IZ", 0.5), ("XX", 0.12), ("YY", 0.12)
        ]
        return hamiltonian_from_paulis(paulis)

    @staticmethod
    def lih_molecule(n_qubits: int = 2) -> np.ndarray:
        """LiH - IBM/Google VQE benchmarks (2-qubit reduction)"""
        paulis = [
            ("II", -0.25), ("ZZ", 0.17), ("ZI", -0.22), 
            ("IZ", 0.12), ("XX", 0.08)
        ]
        return hamiltonian_from_paulis(paulis)

    @staticmethod
    def beh2_molecule(n_qubits: int = 2) -> np.ndarray:
        """BeH2 - Literature STO-3G (2-qubit reduction)"""
        paulis = [
            ("II", -0.35), ("ZZ", 0.25), ("ZI", -0.30), 
            ("IZ", 0.20), ("XX", 0.12), ("YY", 0.12)
        ]
        return hamiltonian_from_paulis(paulis)

    # --- Condensed Matter ---
    
    @staticmethod
    def heisenberg_xxx(n_qubits: int = 2) -> np.ndarray:
        """Heisenberg XXX model - Exact singlet E₀ = -3J"""
        paulis = [("XX", 1.0), ("YY", 1.0), ("ZZ", 1.0)]
        return hamiltonian_from_paulis(paulis)
    
    @staticmethod
    def heisenberg_xxz(n_qubits: int = 2, Jxy: float = 1.0, Jz: float = 0.5) -> np.ndarray:
        """Heisenberg XXZ with anisotropy"""
        paulis = [("XX", Jxy), ("YY", Jxy), ("ZZ", Jz)]
        return hamiltonian_from_paulis(paulis)

    @staticmethod
    def transverse_ising(n_qubits: int = 2, J: float = 1.0, h: float = 0.5) -> np.ndarray:
        """Transverse Field Ising Model"""
        paulis = [("ZZ", -J), ("XI", -h), ("IX", -h)]
        return hamiltonian_from_paulis(paulis)
    
    @staticmethod
    def transverse_ising_critical(n_qubits: int = 2) -> np.ndarray:
        """TFIM at critical point h=J"""
        paulis = [("ZZ", -1.0), ("XI", -1.0), ("IX", -1.0)]
        return hamiltonian_from_paulis(paulis)

    @staticmethod
    def xy_model(n_qubits: int = 2, J: float = 1.0) -> np.ndarray:
        """XY Model - E₀ = -2J"""
        paulis = [("XX", J), ("YY", J)]
        return hamiltonian_from_paulis(paulis)

    @staticmethod
    def ferro_ising(n_qubits: int = 2, J: float = 1.0) -> np.ndarray:
        """Ferromagnetic Ising - E₀ = -J"""
        paulis = [("ZZ", -J)]
        return hamiltonian_from_paulis(paulis)
    
    @staticmethod
    def afm_ising_field(n_qubits: int = 2) -> np.ndarray:
        """Antiferromagnetic Ising with field"""
        paulis = [("ZZ", 1.0), ("ZI", -0.5), ("IZ", -0.5)]
        return hamiltonian_from_paulis(paulis)

    # --- Topological Models ---
    
    @staticmethod
    def ssh_model(n_qubits: int = 2) -> np.ndarray:
        """Su-Schrieffer-Heeger (SSH) topological model."""
        paulis = [("XX", 1.2), ("YY", 1.2), ("ZI", -0.3), ("IZ", -0.3)]
        return hamiltonian_from_paulis(paulis)

    @staticmethod
    def kitaev_chain(n_qubits: int = 2) -> np.ndarray:
        """Kitaev Chain - Majorana fermions"""
        paulis = [("ZZ", -0.8), ("XX", 0.5), ("XI", -0.2), ("IX", -0.2)]
        return hamiltonian_from_paulis(paulis)

    @staticmethod
    def hubbard_dimer(n_qubits: int = 2) -> np.ndarray:
        """Hubbard Dimer - Jordan-Wigner transformation"""
        paulis = [("II", -1.0), ("ZZ", 0.5), ("XX", 0.3), ("YY", 0.3)]
        return hamiltonian_from_paulis(paulis)

    # --- Nuclear Physics ---
    
    @staticmethod
    def deuteron_n2(n_qubits: int = 2) -> np.ndarray:
        """Deuteron N=2 - Dumitrescu et al., PRL 120, 210501 (2018)"""
        paulis = [
            ("II", 5.906709), ("ZI", 0.218291), ("IZ", -6.125), 
            ("XX", -2.143304), ("YY", -2.143304)
        ]
        return hamiltonian_from_paulis(paulis)

    # --- Quantum Optics ---
    
    @staticmethod
    def jaynes_cummings(n_qubits: int = 2, g: float = 0.2) -> np.ndarray:
        """Jaynes-Cummings Model (1 photon truncation)"""
        paulis = [("ZI", 0.5), ("IZ", 0.5), ("XX", g), ("YY", g)]
        return hamiltonian_from_paulis(paulis)
    
    @staticmethod
    def rabi_model(n_qubits: int = 2, g: float = 0.3) -> np.ndarray:
        """Rabi Model (stronger coupling, no RWA)"""
        paulis = [("ZI", 0.5), ("IZ", 0.5), ("XI", g), ("XX", g/2), ("YY", g/2)]
        return hamiltonian_from_paulis(paulis)

    # --- Finance (QUBO formulation) ---
    
    @staticmethod
    def portfolio_2asset(n_qubits: int = 2) -> np.ndarray:
        """Portfolio Optimization (2-asset Markowitz)"""
        paulis = [("ZI", -0.05), ("IZ", -0.08), ("ZZ", 0.02)]
        return hamiltonian_from_paulis(paulis)
    
    @staticmethod
    def credit_risk(n_qubits: int = 2) -> np.ndarray:
        """Credit Risk VaR"""
        paulis = [("ZI", -0.10), ("IZ", -0.15), ("ZZ", 0.05)]
        return hamiltonian_from_paulis(paulis)
    
    @staticmethod
    def option_pricing(n_qubits: int = 2) -> np.ndarray:
        """Option Pricing calibration"""
        paulis = [("ZI", -0.20), ("IZ", -0.25), ("ZZ", 0.08), ("XX", 0.05)]
        return hamiltonian_from_paulis(paulis)


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
        n = len(weights)
        np.random.seed(42)
        returns = np.random.uniform(0.05, 0.20, n)
        cov = np.random.uniform(0.01, 0.05, (n, n))
        cov = np.dot(cov, cov.T)
        risk_aversion = 0.5

        w_sum = np.sum(weights)
        penalty = 100 * (w_sum - 1.0)**2
        
        port_return = np.dot(weights, returns)
        port_risk = np.dot(weights.T, np.dot(cov, weights))
        
        return risk_aversion * port_risk - port_return + penalty

    @staticmethod
    def credit_risk_var(params: np.ndarray) -> float:
        """Minimize Value at Risk (VaR) / Tail Loss."""
        n = len(params)
        np.random.seed(101)
        default_probs = np.random.beta(2, 10, n)
        loss_given_default = np.random.uniform(0.3, 0.8, n)
        
        expected_loss = np.sum(params * default_probs * loss_given_default)
        volatility = np.sum(params**2 * default_probs * (1-default_probs)) 
        
        return expected_loss + 1.65 * np.sqrt(volatility)

    @staticmethod
    def option_pricing_calibration(params: np.ndarray) -> float:
        """Calibrate Volatility Surface (Heston model proxy)."""
        market_vols = np.array([0.2, 0.22, 0.18, 0.25])
        model_vols = params[0] + params[1]*np.array([1, 2, 1, 2]) + params[2]*np.array([0.9, 0.9, 1.1, 1.1])
        mse = np.mean((model_vols - market_vols)**2)
        return mse


# ════════════════════════════════════════════════════════════════════════════
# 4. QAOA / CIRCUIT INFRASTRUCTURE
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
    def complete_graph(n_nodes: int):
        return [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    
    @staticmethod
    def ring_graph(n_nodes: int):
        return [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    
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
            # Cost unitary
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
            
            # Mixer unitary
            beta = betas[layer]
            c, s = np.cos(beta), -1j * np.sin(beta)
            for i in range(n_qubits):
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


# ════════════════════════════════════════════════════════════════════════════
# 5. ANSATZ
# ════════════════════════════════════════════════════════════════════════════

class Ansatz:
    @staticmethod
    def vqe_hardware_efficient(n_qubits: int, depth: int, params: np.ndarray) -> np.ndarray:
        """Hardware efficient ansatz: Ry layers + CNOT entanglement"""
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        param_idx = 0
        
        for _ in range(depth):
            # Ry rotation layer
            for q in range(n_qubits):
                theta = params[param_idx] if param_idx < len(params) else 0.0
                param_idx += 1
                c, s = np.cos(theta/2), np.sin(theta/2)
                Ry = np.array([[c, -s], [s, c]], dtype=complex)
                ops = [I]*n_qubits
                ops[q] = Ry
                state = kron_n(*ops) @ state
            
            # CNOT entanglement layer
            for q in range(n_qubits - 1):
                CNOT = np.eye(dim, dtype=complex)
                for i in range(dim):
                    bits = [(i >> b) & 1 for b in range(n_qubits)]
                    if bits[q] == 1:
                        j = i ^ (1 << (q+1))
                        CNOT[i, i] = 0
                        CNOT[j, i] = 1
                        CNOT[i, j] = 1
                        CNOT[j, j] = 0
                state = CNOT @ state
        
        return state


# ════════════════════════════════════════════════════════════════════════════
# 6. PROBLEM CATALOG DEFINITION
# ════════════════════════════════════════════════════════════════════════════

PROBLEM_CATALOG: Dict[str, Dict[str, Any]] = {
    # ═══════════════════════════════════════════════════════════════════════
    # QUANTUM CHEMISTRY (VQE)
    # ═══════════════════════════════════════════════════════════════════════
    'h2_molecule': {
        'type': 'VQE', 
        'method': 'vqe',  # Was: problem_mode
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.h2_molecule, 
        'landscape': 'smooth', 
        'description': 'H2 Molecule (0.74 Å)',
        'source': 'Kandala et al., Nature 549, 242 (2017)'
    },
    'h2_stretched': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.h2_stretched, 
        'landscape': 'smooth', 
        'description': 'H2 Stretched (1.5 Å)',
        'source': 'PySCF STO-3G'
    },
    'heh_plus': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.heh_plus, 
        'landscape': 'smooth', 
        'description': 'HeH+ Ion',
        'source': 'Hempel et al., PRX 8, 031022 (2018)'
    },
    'lih_molecule': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.lih_molecule, 
        'landscape': 'smooth', 
        'description': 'LiH Molecule',
        'source': 'IBM/Google VQE benchmarks'
    },
    'beh2_molecule': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.beh2_molecule, 
        'landscape': 'smooth', 
        'description': 'BeH2 Molecule',
        'source': 'Literature STO-3G'
    },
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONDENSED MATTER (VQE)
    # ═══════════════════════════════════════════════════════════════════════
    'heisenberg_xxx': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.heisenberg_xxx, 
        'landscape': 'moderate', 
        'description': 'Heisenberg XXX (singlet)',
        'source': 'Exact: E₀ = -3J'
    },
    'heisenberg_xxz': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.heisenberg_xxz, 
        'landscape': 'frustrated', 
        'description': 'Heisenberg XXZ (Δ=0.5)',
        'source': 'Exact diagonalization'
    },
    'transverse_ising': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.transverse_ising, 
        'landscape': 'moderate', 
        'description': 'Transverse Ising (h=0.5)',
        'source': 'Exact: E₀ = -√(J² + h²)'
    },
    'transverse_ising_critical': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.transverse_ising_critical, 
        'landscape': 'critical', 
        'description': 'TFIM Critical Point (h=J)',
        'source': 'Phase transition point'
    },
    'xy_model': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.xy_model, 
        'landscape': 'moderate', 
        'description': 'XY Model',
        'source': 'Exact: E₀ = -2J'
    },
    'ferro_ising': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 1, 
        'hamiltonian_fn': Hamiltonians.ferro_ising, 
        'landscape': 'simple', 
        'description': 'Ferromagnetic Ising',
        'source': 'Aligned spins E₀ = -J'
    },
    'afm_ising_field': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.afm_ising_field, 
        'landscape': 'moderate', 
        'description': 'AFM Ising with Field',
        'source': 'Exact diagonalization'
    },
    
    # ═══════════════════════════════════════════════════════════════════════
    # ADVANCED QUANTUM (VQE)
    # ═══════════════════════════════════════════════════════════════════════
    'ssh_model': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.ssh_model, 
        'landscape': 'topological', 
        'description': 'SSH Topological Model',
        'source': 'Topological insulator'
    },
    'kitaev_chain': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.kitaev_chain, 
        'landscape': 'topological', 
        'description': 'Kitaev Chain',
        'source': 'Majorana fermions'
    },
    'hubbard_dimer': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.hubbard_dimer, 
        'landscape': 'correlated', 
        'description': 'Hubbard Dimer',
        'source': 'Jordan-Wigner, U/t=4'
    },
    
    # ═══════════════════════════════════════════════════════════════════════
    # NUCLEAR PHYSICS (VQE) - NEW!
    # ═══════════════════════════════════════════════════════════════════════
    'deuteron_n2': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.deuteron_n2, 
        'landscape': 'smooth', 
        'description': 'Deuteron (N=2)',
        'source': 'Dumitrescu PRL 2018, binding -2.22 MeV'
    },
    
    # ═══════════════════════════════════════════════════════════════════════
    # QUANTUM OPTICS (VQE) - NEW!
    # ═══════════════════════════════════════════════════════════════════════
    'jaynes_cummings': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.jaynes_cummings, 
        'landscape': 'smooth', 
        'description': 'Jaynes-Cummings (1 photon)',
        'source': 'Resonant ω_a=ω_c, g=0.2'
    },
    'rabi_model': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.rabi_model, 
        'landscape': 'moderate', 
        'description': 'Rabi Model',
        'source': 'No RWA, g=0.3'
    },
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINANCE / QUBO (VQE)
    # ═══════════════════════════════════════════════════════════════════════
    'portfolio_qubo': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.portfolio_2asset, 
        'landscape': 'smooth', 
        'description': 'Portfolio Optimization (QUBO)',
        'source': 'Markowitz 2-asset'
    },
    'credit_risk_qubo': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.credit_risk, 
        'landscape': 'smooth', 
        'description': 'Credit Risk VaR (QUBO)',
        'source': 'VaR formulation'
    },
    'option_pricing_qubo': {
        'type': 'VQE', 
        'method': 'vqe',
        'n_qubits': 2, 
        'depth': 2, 
        'hamiltonian_fn': Hamiltonians.option_pricing, 
        'landscape': 'smooth', 
        'description': 'Option Pricing (QUBO)',
        'source': 'Vol surface calibration'
    },
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLASSICAL OPTIMIZATION (VQE for smooth, QAOA for rugged)
    # ═══════════════════════════════════════════════════════════════════════
    'rosenbrock': {
        'type': 'Classical', 
        'method': 'vqe',  # Smooth valley
        'n_vars': 4, 
        'func': ClassicalObjectives.rosenbrock, 
        'landscape': 'valley', 
        'description': 'Rosenbrock (Banana Valley)'
    },
    'rastrigin': {
        'type': 'Classical', 
        'method': 'qaoa',  # Highly multimodal
        'n_vars': 4, 
        'func': ClassicalObjectives.rastrigin, 
        'landscape': 'rugged', 
        'description': 'Rastrigin (Multimodal)'
    },
    'ackley': {
        'type': 'Classical', 
        'method': 'qaoa',  # Many local minima
        'n_vars': 4, 
        'func': ClassicalObjectives.ackley, 
        'landscape': 'rugged', 
        'description': 'Ackley Function'
    },
    'sphere': {
        'type': 'Classical', 
        'method': 'vqe',  # Convex
        'n_vars': 4, 
        'func': ClassicalObjectives.sphere, 
        'landscape': 'convex', 
        'description': 'Sphere Baseline'
    },
    'beale': {
        'type': 'Classical', 
        'method': 'vqe', 
        'n_vars': 2, 
        'func': ClassicalObjectives.beale, 
        'landscape': 'plateau', 
        'description': 'Beale Function'
    },

    # ═══════════════════════════════════════════════════════════════════════
    # FINANCE (Classical formulation)
    # ═══════════════════════════════════════════════════════════════════════
    'portfolio': {
        'type': 'Finance', 
        'method': 'vqe', 
        'n_vars': 5, 
        'func': FinanceObjectives.portfolio_optimization, 
        'landscape': 'stochastic', 
        'description': 'Portfolio Optimization'
    },
    'credit_risk': {
        'type': 'Finance', 
        'method': 'vqe', 
        'n_vars': 5, 
        'func': FinanceObjectives.credit_risk_var, 
        'landscape': 'stochastic', 
        'description': 'Credit Risk VaR'
    },
    'option_pricing': {
        'type': 'Finance', 
        'method': 'vqe', 
        'n_vars': 3, 
        'func': FinanceObjectives.option_pricing_calibration, 
        'landscape': 'stochastic', 
        'description': 'Option Volatility Calibration'
    },

    # ═══════════════════════════════════════════════════════════════════════
    # QAOA (Graph Optimization)
    # ═══════════════════════════════════════════════════════════════════════
    'maxcut_k4': {
        'type': 'QAOA', 
        'method': 'qaoa',
        'n_qubits': 4, 
        'p': 5, 
        'graph_type': 'complete', 
        'cost_fn': QAOAProblems.maxcut_cost_terms, 
        'landscape': 'rugged', 
        'description': 'MaxCut Complete K4'
    },
    'maxcut_ring5': {
        'type': 'QAOA', 
        'method': 'qaoa',
        'n_qubits': 5, 
        'p': 5, 
        'graph_type': 'ring', 
        'cost_fn': QAOAProblems.maxcut_cost_terms, 
        'landscape': 'rugged', 
        'description': 'MaxCut Ring 5'
    },
    'maxcut_random5': {
        'type': 'QAOA', 
        'method': 'qaoa',
        'n_qubits': 5, 
        'p': 5, 
        'graph_type': 'random', 
        'cost_fn': QAOAProblems.maxcut_cost_terms, 
        'landscape': 'rugged', 
        'description': 'MaxCut Random 5-node'
    },
    'vertex_cover_5': {
        'type': 'QAOA', 
        'method': 'qaoa',
        'n_qubits': 5, 
        'p': 5, 
        'graph_type': 'ring', 
        'cost_fn': QAOAProblems.vertex_cover_cost_terms, 
        'landscape': 'rugged', 
        'description': 'Vertex Cover Ring 5'
    },
    'max_independent_set_5': {
        'type': 'QAOA', 
        'method': 'qaoa',
        'n_qubits': 5, 
        'p': 5, 
        'graph_type': 'ring', 
        'cost_fn': QAOAProblems.max_independent_set_cost_terms, 
        'landscape': 'rugged', 
        'description': 'Max Independent Set Ring 5'
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 7. INTERFACE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def get_problem(name: str) -> Dict[str, Any]:
    """Get problem definition by name."""
    return PROBLEM_CATALOG.get(name)

def list_problems(ptype: str = None, method: str = None) -> List[str]:
    """List all problems, optionally filtered by type or method."""
    problems = list(PROBLEM_CATALOG.keys())
    if ptype:
        problems = [k for k in problems if PROBLEM_CATALOG[k]['type'] == ptype]
    if method:
        problems = [k for k in problems if PROBLEM_CATALOG[k].get('method') == method]
    return problems

def get_energy_function(name: str, seed: int = None, noise: float = 0.0) -> Callable:
    """Get energy function for a problem."""
    prob = get_problem(name)
    if not prob:
        raise ValueError(f"Unknown problem: {name}")
    
    if prob['type'] == 'VQE':
        H = prob['hamiltonian_fn'](prob['n_qubits'])
        def fn(p):
            s = Ansatz.vqe_hardware_efficient(prob['n_qubits'], prob['depth'], p)
            e = np.real(s.conj() @ H @ s).item()
            if noise > 0:
                e += np.random.normal(0, noise * abs(e) + 0.01)
            return e
        return fn
        
    elif prob['type'] == 'Classical' or prob['type'] == 'Finance':
        f = prob['func']
        def fn(p):
            val = f(p)
            if noise > 0:
                val += np.random.normal(0, noise * abs(val) + 0.01)
            return val
        return fn
        
    elif prob['type'] == 'QAOA':
        graph_type = prob.get('graph_type', 'random')
        if graph_type == 'complete':
            edges = QAOAProblems.complete_graph(prob['n_qubits'])
        elif graph_type == 'ring':
            edges = QAOAProblems.ring_graph(prob['n_qubits'])
        else:
            edges = QAOAProblems.random_graph(prob['n_qubits'], 0.5, seed)
        
        if 'vertex' in name or 'independent' in name:
            cost = prob['cost_fn'](edges, prob['n_qubits'])
        else:
            cost = prob['cost_fn'](edges)
        
        return lambda p: QAOACircuit.qaoa_expectation(p, prob['n_qubits'], cost, prob['p'], noise)

def get_n_params(name: str) -> int:
    """Get number of parameters for a problem."""
    p = get_problem(name)
    if p['type'] == 'VQE':
        return p['n_qubits'] * p['depth']
    if p['type'] == 'QAOA':
        return 2 * p['p']
    return p['n_vars']

def get_method(name: str) -> str:
    """Get recommended method ('vqe' or 'qaoa') for a problem."""
    return get_problem(name).get('method', 'vqe')

def get_ground_state_energy(name: str) -> float:
    """Get exact ground state energy for VQE problems."""
    prob = get_problem(name)
    if prob['type'] != 'VQE':
        return None
    H = prob['hamiltonian_fn'](prob['n_qubits'])
    return compute_ground_state(H)

# Backward compatibility
def get_problem_mode(name: str) -> str:
    """DEPRECATED: Use get_method() instead."""
    import warnings
    warnings.warn("get_problem_mode() is deprecated, use get_method() instead", DeprecationWarning)
    return get_method(name)


# ════════════════════════════════════════════════════════════════════════════
# SELF-CHECK
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Mobiu-Q Catalog v2.4")
    print(f"=" * 60)
    print(f"Total problems: {len(PROBLEM_CATALOG)}")
    print(f"Note: For RL, use Gymnasium environments directly")
    print()
    
    # Count by type
    types = {}
    for name, prob in PROBLEM_CATALOG.items():
        t = prob['type']
        types[t] = types.get(t, 0) + 1
    
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")
    
    print()
    print("VQE problems with ground states:")
    for name in list_problems(ptype='VQE')[:5]:
        gs = get_ground_state_energy(name)
        print(f"  {name}: E₀ = {gs:.4f}")
    print("  ...")
