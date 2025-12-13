# mobiu_q_catalog.py
# ==================
# Mobiu-Q Problem Catalog — Hamiltonians, Ansätze, and Problem Definitions
# Clean version with 11 core problems (no EXTRA_PROBLEMS)

import numpy as np
from typing import Callable, Dict, Any
# TestingConfig removed - not needed for client


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
# HAMILTONIANS
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
    def maxcut(n_qubits: int = 3) -> np.ndarray:
        """MaxCut Hamiltonian for triangle graph."""
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        edges = [(0, 1), (1, 2), (0, 2)] if n_qubits == 3 else [(i, i+1) for i in range(n_qubits-1)]
        for i, j in edges:
            ops = [I] * n_qubits
            ops[i] = Z
            ops[j] = Z
            H += 0.5 * (np.eye(dim) - kron_n(*ops))
        return H

    import numpy as np

    @staticmethod
    def be2_molecule(n_qubits: int = 4) -> np.ndarray:
        """
        Simplified Be2 Hamiltonian approximation for 4-qubit VQE tests.
        Derived from minimal STO‑3G symmetry reduction (toy variant).
        """
        if n_qubits != 4:
            raise ValueError("Be2 Hamiltonian defined for 4 qubits only.")

        # Pauli‑like couplings (XX + YY + ZZ terms)
        Jx, Jy, Jz = 0.62, 0.58, 0.79
        h = -0.35

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        def kron(*ops):
            out = ops[0]
            for op in ops[1:]:
                out = np.kron(out, op)
            return out

        H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

        # two‑body couplings
        for q in range(n_qubits - 1):
            H += Jx * kron(*(X if i in [q, q+1] else I for i in range(n_qubits)))
            H += Jy * kron(*(Y if i in [q, q+1] else I for i in range(n_qubits)))
            H += Jz * kron(*(Z if i in [q, q+1] else I for i in range(n_qubits)))

        # local field contribution
        for q in range(n_qubits):
            H += h * kron(*(Z if i == q else I for i in range(n_qubits)))

        return H.real

    # ============================================================
    # ⚛️  He₄ atom (2 qubits) – toy harmonic model
    # ============================================================
    @staticmethod
    def he4_atom(n_qubits: int = 2) -> np.ndarray:
        """
        Two‑qubit toy Hamiltonian for Helium‑4 VQE.
        Emulates correlated electron spin‑pairing with simple XXZ + field.
        """
        if n_qubits != 2:
            raise ValueError("He4 Hamiltonian defined for 2 qubits only.")

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)

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
# ANSATZ
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
# PROBLEM CATALOG — 11 Core Problems
# ════════════════════════════════════════════════════════════════════════════

PROBLEM_CATALOG: Dict[str, Dict[str, Any]] = {
    'h2_molecule': {
        'type': 'VQE',
        'n_qubits': 2,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.h2_molecule,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'smooth',
        'description': 'H2 molecule - smooth molecular landscape'
    },

    'lih_molecule': {
        'type': 'VQE',
        'n_qubits': 4,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.lih_molecule,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'smooth',
        'description': 'LiH molecule - larger molecular system'
    },

    'transverse_ising': {
        'type': 'VQE',
        'n_qubits': 4,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.transverse_ising,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'moderate',
        'description': 'Transverse field Ising model'
    },

    'heisenberg_xxz': {
        'type': 'VQE',
        'n_qubits': 4,
        'depth': 4,
        'hamiltonian_fn': Hamiltonians.heisenberg_xxz,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'frustrated',
        'description': 'Heisenberg XXZ - frustrated anisotropic'
    },

    'xy_model': {
        'type': 'VQE',
        'n_qubits': 4,
        'depth': 4,
        'hamiltonian_fn': Hamiltonians.xy_model,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'moderate',
        'description': 'XY model - moderate landscape'
    },

    'h3_chain': {
        'type': 'VQE',
        'n_qubits': 3,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.h3_chain,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'smooth',
        'description': 'H3 chain - smooth molecular'
    },

    'ferro_ising': {
        'type': 'VQE',
        'n_qubits': 4,
        'depth': 3,
        'hamiltonian_fn': Hamiltonians.ferro_ising,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'smooth',
        'description': 'Ferromagnetic Ising - smooth'
    },

    'antiferro_heisenberg': {
        'type': 'VQE',
        'n_qubits': 4,
        'depth': 4,
        'hamiltonian_fn': Hamiltonians.antiferro_heisenberg,
        'recommended_signals': ['energy_curvature'],
        'landscape': 'frustrated',
        'description': 'Antiferromagnetic Heisenberg - frustrated'
    },

    "be2_molecule": {
        "type": "VQE",
        "n_qubits": 4,
        "depth": 3,
        "hamiltonian_fn": Hamiltonians.be2_molecule,
        "recommended_signals": ["energy_curvature"],
    },

    "he4_atom": {
        "type": "VQE",
        "n_qubits": 2,
        "depth": 2,
        "hamiltonian_fn": Hamiltonians.he4_atom,
        "recommended_signals": ["parameter_velocity"],
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


def list_problems() -> list:
    """List all available problems."""
    return list(PROBLEM_CATALOG.keys())


def get_energy_function(problem_name: str) -> Callable:
    """Create energy function for a given problem."""
    prob = get_problem(problem_name)
    n_qubits = prob['n_qubits']
    depth = prob['depth']
    H = prob['hamiltonian_fn'](n_qubits)

    def energy_fn(params: np.ndarray) -> float:
        state = Ansatz.vqe_hardware_efficient(n_qubits, depth, params)
        return np.real(state.conj() @ H @ state).item()

    return energy_fn


def get_ground_state_energy(problem_name: str) -> float:
    """Compute exact ground state energy for a problem."""
    prob = get_problem(problem_name)
    n_qubits = prob['n_qubits']
    H = prob['hamiltonian_fn'](n_qubits)
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0]


# Catalog loaded silently - use list_problems() to see available problems