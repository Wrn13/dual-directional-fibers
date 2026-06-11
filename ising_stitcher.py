import sys
import itertools as it
import pickle as pk
import numpy as np
import torch as tr
import matplotlib as mp
import matplotlib.pyplot as plt

import dfibers.solvers as sv
import dfibers.traversal as tv
import dfibers.fixed_points as fx
from dfibers.logging_utilities import Logger
import dfibers.numerical_utilities as nu
from hamiltonian_params import single_dual_params

np.set_printoptions(linewidth=10000000, threshold=1000000)
tr.set_printoptions(linewidth=1000)
tr.set_default_dtype(tr.float64)
rng = np.random.default_rng(seed=1)


def kron_prod(op_list):
    """Calculates the Kronecker product of a list of operators."""
    result = op_list[0]
    for op in op_list[1:]:
        result = tr.kron(result, op)
    return result


pauli = [
    tr.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=tr.complex128),
    tr.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=tr.complex128),
    tr.tensor([[0.0, -1j], [1j, 0.0]], dtype=tr.complex128),
    tr.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=tr.complex128),
]

# number of qubits
n_qb = 6


def get_Hamiltonian(c):
    """Creates the general ising Hamiltonian with boundary terms"""
    # c = [K,12], K is batch size
    K = c.shape[0]
    I_list = [pauli[0]] * n_qb
    H = tr.zeros((K, 2**n_qb, 2**n_qb), dtype=tr.complex128)

    # Interaction terms and Single Site terms
    for i in range(n_qb - 1):
        for p in range(1, 4):
            interaction_list = I_list[:]
            interaction_list[i] = pauli[p]
            interaction_list[i + 1] = pauli[p]

            single_term = I_list[:]
            single_term[i] = pauli[p]

            H += c[:, p - 1].view(K, 1, 1) * kron_prod(interaction_list) + c[
                :, 3 + p - 1
            ].view(K, 1, 1) * kron_prod(single_term)

    # Boundary Terms
    for p in range(1, 4):
        site_one = I_list[:]
        site_one[0] = pauli[p]

        site_N = I_list[:]
        site_N[n_qb - 1] = pauli[p]

        H += c[None, :, 6 + p - 1].view(K, 1, 1) * kron_prod(site_one) + c[
            None, :, 9 + p - 1
        ].view(K, 1, 1) * kron_prod(site_N)

    return H


def get_spectrum(c):
    # c is (K, 12) tensor, K is batch size
    H = get_Hamiltonian(c)
    try:
        spectrum = tr.linalg.eigvalsh(H)  # (K, 2**n_qb)
    except tr._C._LinAlgError:
        print("Erorr computing eigenvalues")
        with open("ErrorHamiltonian.txt", "w") as f:
            f.write(str(H))
            f.write("__________________________________\n")
        return np.inf
    return spectrum


def get_loss_factory(c0):
    target_spectrum = get_spectrum(c0)

    def get_loss(c):
        # c is (K, 9) tensor, K is batch size
        spectrum = get_spectrum(c)

        return tr.sum((spectrum - target_spectrum) ** 2, dim=1)  # (K,)

    return get_loss

def f_factory(get_loss):
    def f(v):
        # v is (9, K) numpy array, return (9, K) batch of gradients
        c = tr.tensor(v.T, requires_grad=True)  # torch wants batch first
        loss = get_loss(c)
        loss.sum().backward()
        return c.grad.numpy().T

    return f


def Df_factory(get_loss):
    def Df(v):
        # v is (12, K) numpy array, return (K, 12, 12) batch of hessians (jacobian of gradients)
        hess_fun = tr.func.hessian(get_loss)
        Dfv = []
        for k in range(v.shape[1]):
            c = tr.tensor(v[:, k : k + 1].T)

            Dfv.append(hess_fun(c).squeeze())  # squeeze singleton batch dimensions
        Dfv = tr.stack(Dfv)

        return Dfv.numpy()

    return Df


def ef(v):
    return 1e-9


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    do_fiber = True

    # constant direction
    c_dir = np.array([[rng.normal() for _ in range(12)]]).T

    if do_fiber:
        h = 1.5
        J = 1
        # c_targ = tr.tensor([[0, 0, J, h, 0, 0, 0, 0, 0, h, 0, 0]])
        c_targ = tr.tensor([[0, 0, h, J, 0, 0, 0, 0, h, -J, 0, 0]])
        # c_targ = tr.tensor(
        #    [[0.1, 0.11, 0.111, 1, 1.1, 0.12, 0.13, 0.02, 0.21, 0.9, 0.3, 0.01]]
        # )
        get_loss = get_loss_factory(c_targ)
        f = f_factory(get_loss)
        Df = Df_factory(get_loss)

        # get initial fiber point
        v0 = c_targ.numpy().T

        # # f(v0) = 0, so use default random choice of direction vector

        # Set up fiber arguments
        fiber_kwargs = {
            "f": f,
            "Df": Df,
            "ef": ef,
            "compute_step_amount": lambda trace: (0.1, 0, False),
            "v": v0,
            "c": c_dir,
            "terminate": lambda trace: (
                get_loss(tr.tensor(trace.x[:12].T)) > 0.01
            ).any(),
            "max_step_size": 100,
            "max_traverse_steps": 50000,  # 000,
            "max_solve_iterations": 2**5,
            "logger": Logger(sys.stdout),
        }

        # Run in one direction
        solution = sv.fiber_solver(**fiber_kwargs)
        X1 = np.concatenate(solution["Fiber trace"].points, axis=1)
        V1 = X1[:-1, :]
        A1 = X1[-1, :]
        R1 = solution["Fixed points"]
        z = solution["Fiber trace"].z_initial

        # Run in other direction (negate initial tangent)
        fiber_kwargs["z"] = -z
        solution = sv.fiber_solver(**fiber_kwargs)
        X2 = np.concatenate(solution["Fiber trace"].points, axis=1)
        V2 = X2[:-1, :]
        A2 = X2[-1, :]
        R2 = solution["Fixed points"]


        # Join fiber segments and roots
        V = np.concatenate((np.fliplr(V1), V2), axis=1)
        A = np.concatenate((A1[::-1], A2), axis=0)
        R = np.concatenate((R1, R2), axis=1)

        R, fixed = fx.refine_points(R, f, ef, Df)
        R = R[:, fixed]

        duplicates = lambda U, v: (np.fabs(U - v) < 0.1).all(axis=0)
        R = fx.get_unique_points(R, duplicates)

        with open("results/ising_seed_1.pkl", "wb") as f:
            pk.dump((c_targ, V, A, R), f)

    with open("results/ising_seed_1.pkl", "rb") as f:
        (c_targ, V, A, R) = pk.load(f)

    get_loss = get_loss_factory(c_targ)
    f = f_factory(get_loss)

    C = f(V)
    print("Initial chosen c: ", c_dir)
    print("constant direction vectors:")
    print(C[:, :3])

    normal, kappa, t = nu.find_curvature_normal(V.T)
    plt.plot(np.linspace(0,len(kappa), len(kappa)), kappa)
    plt.xlabel("Index")
    plt.ylabel("Curvature")
    plt.show()

