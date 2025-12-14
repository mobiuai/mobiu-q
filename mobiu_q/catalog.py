# mobiu_q_catalog.py
# ==================
# Mobiu-Q Problem Catalog — Hamiltonians, Ansätze, and Problem Definitions
# Version 1.1.0 - Added QAOA problems (MaxCut, Vertex Cover, Max Independent Set)

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


# ════════════════════════════════════════════════════════════════════════════
# VQE HAMILTONIANS
# ════════════════════════════════════════════════════════════════════════════

class Hamiltonians:
    """Collection of quantum Hamiltonians for VQE problems."""

    @staticmethod
    def h2_molecule(n_qubits: int = 2) -> np.ndarray:
        """H2 molecule Hamiltonian (simplified)."""
        H = -1.0 * kron_n(Z, I) - 0.5 * kron_n(I, Z) + 0.3 * kron_n(X, X)
        return H

    @staticmethod
    def lih_molecule(n_qubits: int = 4) -> np.ndarray:
        """LiH molecule Hamiltonian (simplified 4-qubit)."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits):
            ops = [I] * n_qubits
            ops[i] = Z
            H += -0.5 * (i + 1) * kron_n(*ops)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits
            ops[i] = X
            ops[i + 1] = X
            H += 0.2 * kron_n(*ops)
        return H

    @staticmethod
    def transverse_ising(n_qubits: int = 4, J: float = 1.0, h: float = 0.5) -> np.ndarray:
        """Transverse field Ising model."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits
            ops[i] = Z
            ops[i + 1] = Z
            H += -J * kron_n(*ops)
        for i in range(n_qubits):
            ops = [I] * n_qubits
            ops[i] = X
            H += -h * kron_n(*ops)
        return H

    @staticmethod
    def heisenberg_xxz(n_qubits: int = 4, Jxy: float = 1.0, Jz: float = 0.5) -> np.ndarray:
        """Heisenberg XXZ model."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits
            ops[i] = X
            ops[i + 1] = X
            H += Jxy * kron_n(*ops)
            ops = [I] * n_qubits
            ops[i] = Y
            ops[i + 1] = Y
            H += Jxy * kron_n(*ops)
            ops = [I] * n_qubits
            ops[i] = Z
            ops[i + 1] = Z
            H += Jz * kron_n(*ops)
        return H

    @staticmethod
    def xy_model(n_qubits: int = 4, J: float = 1.0) -> np.ndarray:
        """XY model Hamiltonian."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits
            ops[i] = X
            ops[i + 1] = X
            H += J * kron_n(*ops)
            ops = [I] * n_qubits
            ops[i] = Y
            ops[i + 1] = Y
            H += J * kron_n(*ops)
        return H

    @staticmethod
    def h3_chain(n_qubits: int = 3) -> np.ndarray:
        """H3 chain Hamiltonian."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits):
            ops = [I] * n_qubits
            ops[i] = Z
            H += -0.8 * kron_n(*ops)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits
            ops[i] = X
            ops[i + 1] = X
            H += 0.25 * kron_n(*ops)
        return H

    @staticmethod
    def ferro_ising(n_qubits: int = 4, J: float = 1.0) -> np.ndarray:
        """Ferromagnetic Ising model."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits
            ops[i] = Z
            ops[i + 1] = Z
            H += -J * kron_n(*ops)
        return H

    @staticmethod
    def antiferro_heisenberg(n_qubits: int = 4, J: float = 1.0) -> np.ndarray:
        """Antiferromagnetic Heisenberg model."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n_qubits - 1):
            ops = [I] * n_qubits
            ops[i] = X
            ops[i + 1] = X
            H += J * kron_n(*ops)
            ops = [I] * n_qubits
            ops[i] = Y
            ops[i + 1] = Y
            H += J * kron_n(*ops)
            ops = [I] * n_qubits
            ops[i] = Z
            ops[i + 1] = Z
            H += J * kron_n(*ops)
        return H

    @staticmethod
    def be2_molecule(n_qubits: int = 4) -> np.ndarray:
        """
        Simplified Be2 Hamiltonian approximation for 4-qubit VQE tests.
        """
        if n_qubits != 4:
            raise ValueError("Be2 Hamiltonian defined for 4 qubits only.")

        Jx, Jy, Jz = 0.62, 0.58, 0.79
        h = -0.35

        def kron(*ops):
            out = ops[0]
            for op in ops[1:]:
                out = np.kron(out, op)
            return out

        H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

        for q in range(n_qubits - 1):
            H += Jx * kron(*(X if i in [q, q+1] else I for i in range(n_qubits)))
            H += Jy * kron(*(Y if i in [q, q+1] else I for i in range(n_qubits)))
            H += Jz * kron(*(Z if i in [q, q+1] else I for i in range(n_qubits)))

        for q in range(n_qubits):
            H += h * kron(*(Z if i == q else I for i in range(n_qubits)))

        return H.real

    @staticmethod
    def he4_atom(n_qubits: int = 2) -> np.ndarray:
        """
        Two-qubit toy Hamiltonian for Helium-4 VQE.
        """
        if n_qubits != 2:
            raise ValueError("He4 Hamiltonian defined for 2 qubits only.")

        Jx, Jy, Jz = 0.9, 0.9, 1.1
        h = -0.4

        H = (
            Jx * np.kron(X, X)
            + Jy * np.kron(Y, Y)
            + Jz * np.kron(Z, Z)
            + h  * (np.kron(Z, I) + np.kron(I, Z))
        )
        return H.real


# ════════════════════════════════════════════════════════════════════════════
# QAOA GRAPH PROBLEMS
# ════════════════════════════════════════════════════════════════════════════

class QAOAProblems:
    """Collection of combinatorial optimization problems for QAOA."""
    
    @staticmethod
    def random_graph(n_nodes: int, edge_prob: float = 0.5, seed: int = None) -> List[Tuple[int, int]]:
        """Generate random graph edges."""
        if seed is not None:
            np.random.seed(seed)
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() < edge_prob:
                    edges.append((i, j))
        if len(edges) == 0:
            edges = [(0, 1)]  # At least one edge
        return edges
    
    @staticmethod
    def maxcut_cost_terms(edges: List[Tuple[int, int]]) -> List[Tuple[float, Tuple[int, ...]]]:
        """
        MaxCut cost Hamiltonian terms.
        C = Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2
        """
        return [(-0.5, (i, j)) for i, j in edges]
    
    @staticmethod
    def vertex_cover_cost_terms(edges: List[Tuple[int, int]], n_nodes: int, 
                                 penalty: float = 2.0) -> List[Tuple[float, Tuple[int, ...]]]:
        """
        Vertex Cover cost Hamiltonian terms.
        Minimize number of vertices while covering all edges.
        """
        terms = []
        # Objective: minimize number of selected vertices
        for i in range(n_nodes):
            terms.append((0.5, (i,)))  # (1 - Z_i) / 2
        # Constraint: each edge must have at least one endpoint selected
        for i, j in edges:
            terms.append((penalty * 0.25, (i, j)))  # Penalty for uncovered edge
        return terms
    
    @staticmethod
    def max_independent_set_cost_terms(edges: List[Tuple[int, int]], n_nodes: int,
                                        penalty: float = 2.0) -> List[Tuple[float, Tuple[int, ...]]]:
        """
        Max Independent Set cost Hamiltonian terms.
        Maximize selected vertices with no adjacent pairs.
        """
        terms = []
        # Objective: maximize number of selected vertices (minimize negative)
        for i in range(n_nodes):
            terms.append((-0.5, (i,)))  # -(1 - Z_i) / 2
        # Constraint: no two adjacent vertices selected
        for i, j in edges:
            terms.append((penalty * 0.25, (i, j)))  # Penalty for adjacent selection
        return terms


# ════════════════════════════════════════════════════════════════════════════
# QAOA CIRCUIT
# ════════════════════════════════════════════════════════════════════════════

class QAOACircuit:
    """QAOA circuit implementation."""
    
    @staticmethod
    def qaoa_expectation(params: np.ndarray, n_qubits: int, 
                         cost_terms: List[Tuple[float, Tuple[int, ...]]], 
                         p: int, noise_level: float = 0.0) -> float:
        """
        Compute QAOA expectation value.
        
        Args:
            params: [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
            n_qubits: Number of qubits
            cost_terms: List of (coefficient, qubit_indices) tuples
            p: QAOA depth
            noise_level: Optional noise (0.0 = noiseless)
        
        Returns:
            Expectation value of cost Hamiltonian
        """
        gammas = params[:p]
        betas = params[p:]
        
        # Initialize |+⟩^n state
        state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
        
        for layer in range(p):
            gamma = gammas[layer]
            beta = betas[layer]
            
            # Cost unitary: exp(-i γ C)
            for coef, qubits in cost_terms:
                if len(qubits) == 2:
                    i, j = qubits
                    for k in range(2**n_qubits):
                        bit_i = (k >> i) & 1
                        bit_j = (k >> j) & 1
                        z_i = 1 - 2 * bit_i
                        z_j = 1 - 2 * bit_j
                        state[k] *= np.exp(-1j * gamma * coef * z_i * z_j)
                elif len(qubits) == 1:
                    i = qubits[0]
                    for k in range(2**n_qubits):
                        bit_i = (k >> i) & 1
                        z_i = 1 - 2 * bit_i
                        state[k] *= np.exp(-1j * gamma * coef * z_i)
            
            # Mixer unitary: exp(-i β Σ X_j)
            for qubit in range(n_qubits):
                new_state = np.zeros_like(state)
                c = np.cos(beta)
                s = np.sin(beta)
                for k in range(2**n_qubits):
                    bit = (k >> qubit) & 1
                    k_flipped = k ^ (1 << qubit)
                    if bit == 0:
                        new_state[k] += c * state[k] - 1j * s * state[k_flipped]
                    else:
                        new_state[k] += -1j * s * state[k_flipped] + c * state[k]
                state = new_state
        
        # Compute expectation
        expectation = 0.0
        for k in range(2**n_qubits):
            prob = np.abs(state[k])**2
            cost = 0.0
            for coef, qubits in cost_terms:
                if len(qubits) == 2:
                    i, j = qubits
                    bit_i = (k >> i) & 1
                    bit_j = (k >> j) & 1
                    z_i = 1 - 2 * bit_i
                    z_j = 1 - 2 * bit_j
                    cost += coef * z_i * z_j
                elif len(qubits) == 1:
                    i = qubits[0]
                    bit_i = (k >> i) & 1
                    z_i = 1 - 2 * bit_i
                    cost += coef * z_i
            expectation += prob * cost
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * abs(expectation) + 0.01)
            expectation += noise
        
        return float(expectation)


# ════════════════════════════════════════════════════════════════════════════
# VQE ANSATZ
# ════════════════════════════════════════════════════════════════════════════

class Ansatz:
    """Quantum circuit ansätze for VQE."""

    @staticmethod
    def vqe_hardware_efficient(n_qubits: int, depth: int, params: np.ndarray) -> np.ndarray:
        """Hardware-efficient ansatz with Ry and CNOT gates."""
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        param_idx = 0
        for layer in range(depth):
            for q in range(n_qubits):
                theta = params[param_idx] if param_idx < len(params) else 0.0
                param_idx += 1
                cos_t = np.cos(theta / 2)
                sin_t = np.sin(theta / 2)
                Ry = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex)
                ops = [I] * n_qubits
                ops[q] = Ry
                U = kron_n(*ops)
                state = U @ state

            for q in range(n_qubits - 1):
                CNOT = np.eye(dim, dtype=complex)
                for i in range(dim):
                    bits = [(i >> b) & 1 for b in range(n_qubits)]
                    if bits[q] == 1:
                        bits[q + 1] = 1 - bits[q + 1]
                        j = sum(b << idx for idx, b in enumerate(bits))
                        CNOT[i, i] = 0
                        CNOT[j, i] = 1
                        CNOT[i, j] = 1
                        CNOT[j, j] = 0
                state = CNOT @ state

        return state


# ════════════════════════════════════════════════════════════════════════════
# PROBLEM CATALOG
# ════════════════════════════════════════════════════════════════════════════

PROBLEM_CATALOG: Dict[str, Dict[str, Any]] = {
    # ─────────────────────────────────────────────────────────────────────────
    # VQE Problems (Chemistry & Physics)
    # ─────────────────────────────────────────────────────────────────────────
    'h2_molecule': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 2,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.h2_molecule,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'smooth',
        'description': 'H2 molecule - smooth molecular landscape'
    },

    'lih_molecule': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 4,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.lih_molecule,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'smooth',
        'description': 'LiH molecule - larger molecular system'
    },

    'transverse_ising': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 4,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.transverse_ising,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'moderate',
        'description': 'Transverse field Ising model'
    },

    'heisenberg_xxz': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 4,
        'depth': 4,
        'hamiltonian_fn': Hamiltonians.heisenberg_xxz,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'frustrated',
        'description': 'Heisenberg XXZ - frustrated anisotropic'
    },

    'xy_model': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 4,
        'depth': 4,
        'hamiltonian_fn': Hamiltonians.xy_model,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'moderate',
        'description': 'XY model - moderate landscape'
    },

    'h3_chain': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 3,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.h3_chain,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'smooth',
        'description': 'H3 chain - smooth molecular'
    },

    'ferro_ising': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 4,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.ferro_ising,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'smooth',
        'description': 'Ferromagnetic Ising - smooth'
    },

    'antiferro_heisenberg': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 4,
        'depth': 4,
        'hamiltonian_fn': Hamiltonians.antiferro_heisenberg,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'frustrated',
        'description': 'Antiferromagnetic Heisenberg - frustrated'
    },

    'be2_molecule': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 4,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.be2_molecule,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'smooth',
        'description': 'Be2 molecule - beryllium dimer'
    },

    'he4_atom': {
        'type': 'VQE',
        'problem_mode': 'vqe',
        'n_qubits': 2,
        'depth': 2,
        'hamiltonian_fn': Hamiltonians.he4_atom,
        'recommended_signals': ['parameter_velocity'],
        'landscape': 'smooth',
        'description': 'Helium-4 atom - simple 2-qubit system'
    },

    # ─────────────────────────────────────────────────────────────────────────
    # QAOA Problems (Combinatorial Optimization)
    # ─────────────────────────────────────────────────────────────────────────
    'maxcut_5': {
        'type': 'QAOA',
        'problem_mode': 'qaoa',
        'n_qubits': 5,
        'p': 3,
        'graph_type': 'random',
        'edge_prob': 0.5,
        'cost_fn': QAOAProblems.maxcut_cost_terms,
        'recommended_signals': ['energy_curvature', 'realized_improvement'],
        'landscape': 'rugged',
        'description': 'MaxCut on 5-node random graph'
    },

    'maxcut_8': {
        'type': 'QAOA',
        'problem_mode': 'qaoa',
        'n_qubits': 8,
        'p': 4,
        'graph_type': 'random',
        'edge_prob': 0.4,
        'cost_fn': QAOAProblems.maxcut_cost_terms,
        'recommended_signals': ['energy_curvature', 'realized_improvement'],
        'landscape': 'rugged',
        'description': 'MaxCut on 8-node random graph'
    },

    'vertex_cover_5': {
        'type': 'QAOA',
        'problem_mode': 'qaoa',
        'n_qubits': 5,
        'p': 3,
        'graph_type': 'random',
        'edge_prob': 0.5,
        'cost_fn': QAOAProblems.vertex_cover_cost_terms,
        'recommended_signals': ['energy_curvature', 'realized_improvement'],
        'landscape': 'rugged',
        'description': 'Vertex Cover on 5-node random graph'
    },

    'max_independent_set_5': {
        'type': 'QAOA',
        'problem_mode': 'qaoa',
        'n_qubits': 5,
        'p': 3,
        'graph_type': 'random',
        'edge_prob': 0.5,
        'cost_fn': QAOAProblems.max_independent_set_cost_terms,
        'recommended_signals': ['energy_curvature', 'realized_improvement'],
        'landscape': 'rugged',
        'description': 'Max Independent Set on 5-node random graph'
    },
}


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def get_problem(name: str) -> Dict[str, Any]:
    """Get problem configuration by name."""
    if name not in PROBLEM_CATALOG:
        raise ValueError(f"Unknown problem: {name}. Available: {list(PROBLEM_CATALOG.keys())}")
    return PROBLEM_CATALOG[name]


def list_problems(problem_type: str = None) -> list:
    """
    List available problems.
    
    Args:
        problem_type: Optional filter - 'VQE', 'QAOA', or None for all
    """
    if problem_type is None:
        return list(PROBLEM_CATALOG.keys())
    return [k for k, v in PROBLEM_CATALOG.items() if v['type'] == problem_type]


def get_energy_function(problem_name: str, seed: int = None, noise_level: float = 0.0) -> Callable:
    """
    Create energy function for a given problem.
    
    Args:
        problem_name: Name of problem from catalog
        seed: Random seed for graph generation (QAOA only)
        noise_level: Noise level (0.0 = noiseless)
    
    Returns:
        Energy function that takes params and returns float
    """
    prob = get_problem(problem_name)
    
    if prob['type'] == 'VQE':
        n_qubits = prob['n_qubits']
        depth = prob['depth']
        H = prob['hamiltonian_fn'](n_qubits)

        def energy_fn(params: np.ndarray) -> float:
            state = Ansatz.vqe_hardware_efficient(n_qubits, depth, params)
            energy = np.real(state.conj() @ H @ state).item()
            if noise_level > 0:
                energy += np.random.normal(0, noise_level * abs(energy) + 0.01)
            return energy

        return energy_fn
    
    elif prob['type'] == 'QAOA':
        n_qubits = prob['n_qubits']
        p = prob['p']
        edge_prob = prob.get('edge_prob', 0.5)
        
        # Generate graph
        edges = QAOAProblems.random_graph(n_qubits, edge_prob, seed)
        
        # Get cost terms
        cost_fn = prob['cost_fn']
        if 'vertex_cover' in problem_name or 'independent_set' in problem_name:
            cost_terms = cost_fn(edges, n_qubits)
        else:
            cost_terms = cost_fn(edges)
        
        def energy_fn(params: np.ndarray) -> float:
            return QAOACircuit.qaoa_expectation(params, n_qubits, cost_terms, p, noise_level)
        
        return energy_fn
    
    else:
        raise ValueError(f"Unknown problem type: {prob['type']}")


def get_ground_state_energy(problem_name: str) -> float:
    """Compute exact ground state energy for a problem."""
    prob = get_problem(problem_name)
    
    if prob['type'] == 'VQE':
        n_qubits = prob['n_qubits']
        H = prob['hamiltonian_fn'](n_qubits)
        eigenvalues = np.linalg.eigvalsh(H)
        return eigenvalues[0]
    else:
        # For QAOA, ground state depends on graph instance
        # Return None or compute for specific instance
        return None


def get_n_params(problem_name: str) -> int:
    """Get number of parameters for a problem."""
    prob = get_problem(problem_name)
    
    if prob['type'] == 'VQE':
        return prob['n_qubits'] * prob['depth']
    elif prob['type'] == 'QAOA':
        return 2 * prob['p']  # gamma_1..p, beta_1..p
    else:
        raise ValueError(f"Unknown problem type: {prob['type']}")


def get_problem_mode(problem_name: str) -> str:
    """Get the recommended Mobiu-Q problem mode for a problem."""
    prob = get_problem(problem_name)
    return prob.get('problem_mode', 'vqe')


# ════════════════════════════════════════════════════════════════════════════
# CATALOG INFO
# ════════════════════════════════════════════════════════════════════════════

def print_catalog():
    """Print all available problems."""
    vqe_problems = list_problems('VQE')
    qaoa_problems = list_problems('QAOA')
    
    print("=" * 60)
    print("MOBIU-Q PROBLEM CATALOG")
    print("=" * 60)
    
    print(f"\n📊 VQE Problems ({len(vqe_problems)}):")
    print("-" * 40)
    for name in vqe_problems:
        prob = PROBLEM_CATALOG[name]
        print(f"  • {name}: {prob['description']}")
        print(f"    {prob['n_qubits']} qubits, depth={prob['depth']}, {prob['landscape']} landscape")
    
    print(f"\n🔀 QAOA Problems ({len(qaoa_problems)}):")
    print("-" * 40)
    for name in qaoa_problems:
        prob = PROBLEM_CATALOG[name]
        print(f"  • {name}: {prob['description']}")
        print(f"    {prob['n_qubits']} qubits, p={prob['p']}, {prob['landscape']} landscape")
    
    print("\n" + "=" * 60)
    print(f"Total: {len(PROBLEM_CATALOG)} problems")
    print("=" * 60)


# Print catalog summary on import
_vqe = list_problems('VQE')
_qaoa = list_problems('QAOA')
print(f"✅ Mobiu-Q Catalog loaded — {len(_vqe)} VQE + {len(_qaoa)} QAOA problems")
print(f"   VQE: {_vqe}")
print(f"   QAOA: {_qaoa}")