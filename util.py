import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, kron, identity
from scipy.sparse.linalg import eigsh


def binary_format(bit: int, N: int):
    bit_str = bin(bit)[2:]
    zeros_to_add = N - len(bit_str)
    return "".join("0" * zeros_to_add + bit_str)


def convert_to_total_basis(A: int, A_qubits: list[int]):
    integer_val = 0
    A_str = binary_format(A, len(A_qubits))
    for i, val in enumerate(A_str):
        integer_val += 2 ** list(reversed(A_qubits))[i] * (1 if val == "1" else 0)

    return integer_val


def reduced_density_matrix(A: list[int], N: int, rho: np.ndarray) -> np.ndarray:
    """
    Function to compute the reduced density matrix.

    @param A: The list of qubits in the A system
    @param N: The total number of qubits making up the hilbert space
    @param rho: The density matrix of the system

    """
    not_A = set(range(N)).difference(set(A))
    N = [convert_to_total_basis(x, sorted(list(not_A))) for x in range(2 ** len(not_A))]
    i_list = [x for x in range(2 ** len(A))]
    k_list = [x for x in range(2 ** len(A))]

    rho_A = np.zeros((2 ** len(A), 2 ** len(A)), dtype=np.complex64)
    for i_str in i_list:
        for k_str in k_list:
            i = convert_to_total_basis(i_str, A)
            k = convert_to_total_basis(k_str, A)
            sum = 0
            for j in N:
                bra = i | j
                ket = k | j

                sum += rho[bra, ket]

            rho_A[i_str, k_str] = sum
    return rho_A


def get_pauli_matrices():
    sx = csc_matrix(np.array([[0, 1], [1, 0]]))
    sz = csc_matrix(np.array([[1, 0], [0, -1]]))
    si = csc_matrix(np.eye(2))
    return sx, sz, si


def build_ising_hamiltonian(N, J, h):
    sx, sz, si = get_pauli_matrices()

    # Pre-compute the identity for efficiency
    I_list = [si] * N

    H = csc_matrix((2**N, 2**N), dtype="complex128")

    # Nearest-neighbor ZZ interaction term
    for i in range(N - 1):
        op_list = I_list[:]
        op_list[i] = sz
        op_list[i + 1] = sz
        H += J * kron_prod(op_list)

    # Transverse X field term
    for i in range(N):
        op_list = I_list[:]
        op_list[i] = sx
        H += h * kron_prod(op_list)

    return H


def kron_prod(op_list):
    """Calculates the Kronecker product of a list of operators."""
    result = op_list[0]
    for op in op_list[1:]:
        result = kron(result, op, format="csc")
    return result


if __name__ == "__main__":
    H = build_ising_hamiltonian(4, 1, 1.05).toarray()
    with (
        open("RefHamiltonian.txt", "w") as f,
        np.printoptions(threshold=100000000, linewidth=10000000000),
    ):
        f.write(str(H))

        D = np.linalg.eigvals(H)

        f.write("\n")
        f.write(str(D))
