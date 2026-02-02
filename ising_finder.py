import sys
import itertools as it
import pickle as pk
import numpy as np
import torch as tr
import matplotlib as mp
import matplotlib.pyplot as pt

import dfibers.solvers as sv
import dfibers.traversal as tv
import dfibers.fixed_points as fx
from dfibers.logging_utilities import Logger
from hamiltonian_params import single_dual_params

np.set_printoptions(linewidth=10000000, threshold=1000000)
tr.set_printoptions(linewidth=1000)
tr.set_default_dtype(tr.float64)
rng = np.random.default_rng(seed=7)


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
    mp.rcParams["text.usetex"] = True

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

        with open("DiffHamiltonain.txt", "w") as file:
            file.write(str(get_Hamiltonian(c_targ).numpy()[0]))

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
        print(len(A1))

        # Run in other direction (negate initial tangent)
        fiber_kwargs["z"] = -z
        solution = sv.fiber_solver(**fiber_kwargs)
        X2 = np.concatenate(solution["Fiber trace"].points, axis=1)
        V2 = X2[:-1, :]
        A2 = X2[-1, :]
        R2 = solution["Fixed points"]
        print(len(A2))

        # Join fiber segments and roots
        V = np.concatenate((np.fliplr(V1), V2), axis=1)
        A = np.concatenate((A1[::-1], A2), axis=0)
        R = np.concatenate((R1, R2), axis=1)

        R, fixed = fx.refine_points(R, f, ef, Df)
        R = R[:, fixed]

        duplicates = lambda U, v: (np.fabs(U - v) < 0.1).all(axis=0)
        R = fx.get_unique_points(R, duplicates)

        with open("results/krammerswanier_random_seed7.pkl", "wb") as f:
            pk.dump((c_targ, V, A, R), f)

    with open("results/krammerswanier_random_seed7.pkl", "rb") as f:
        (c_targ, V, A, R) = pk.load(f)

    get_loss = get_loss_factory(c_targ)
    f = f_factory(get_loss)

    C = f(V)
    print("Initial chosen c: ", c_dir)
    print("constant direction vectors:")
    print(C[:, :3])

    print("Division of C and c_dir")

    print(C[:, 0] / c_dir[:, :].T)
    # # filter duplicates
    duplicates = lambda U, v: (np.fabs(U - v) < 0.1).all(axis=0)
    R = fx.get_unique_points(R, duplicates)

    print("R after filtering L1 norm less than .1: ", R.shape)
    # filter stationary points with non-zero loss
    loss = get_loss(tr.tensor(R.T))
    keep = loss.numpy() < 1e-8
    R = R[:, keep]

    print("R after too large loss: ", R.shape)
    loss = get_loss(tr.tensor(R.T))

    print(f"{R.shape[1]} optima")

    targ_spectrum = get_spectrum(c_targ)
    spectrums = get_spectrum(tr.tensor(R.T))

    diffs = spectrums - targ_spectrum

    print(f"loss = {loss}")
    print("f(R)", np.fabs(f(R)).max(axis=0))

    trace_loss = get_loss(tr.tensor(V.T))
    local_min = np.flatnonzero(
        (trace_loss[1:-1] <= trace_loss[2:]) & (trace_loss[1:-1] <= trace_loss[:-2])
    )

    # pt.plot(A)
    # pt.show()

    fig = pt.figure(figsize=(4, 2))

    pt.subplot(1, 2, 1)
    # pt.plot(diffs.numpy().T)
    # pt.xlabel("$i$")
    # pt.ylabel("$\\lambda_i - \\lambda^*_i$")

    pt.title("Loss Landscape with J=1, H=1.5 Ising Model")
    pt.plot(trace_loss, "k-")
    pt.plot(local_min[0], 0, "bo")
    pt.plot(local_min[1], 0, "ro")
    pt.plot(local_min[-1], 0, "go")
    pt.ylabel("$||\\Lambda - \\Lambda_0||^2$")
    pt.xlabel("Step along fiber")

    pt.subplot(1, 2, 2)
    pt.title("Coefficient Values on J=1 H=1.5 Ising Model")
    pt.plot(R[:, 0], "b-")
    pt.plot(R[:, 1], "r-")
    pt.plot(R[:, -1], "g-")
    pt.xlabel("Coefficient index")
    pt.ylabel("Coefficient value")
    pt.ylim([-1.5, 2])

    # fig.suptitle("SA2")
    pt.tight_layout()
    pt.show()

    # # Find unitary to go between TPS
    # Hs = get_Hamiltonian(tr.tensor(R.T))

    # H_0, H_p = [Hs[0], Hs[-1]]
    # D_0, V_0 = tr.linalg.eig(H_0)
    # D_p, V_p = tr.linalg.eig(H_p)

    # print(V_p.shape)
    # U = V_p @ tr.inverse(V_0)
    # print("Unitary Change of Base Matrix:")
    # print(U)

    # print(U.shape)

    # indices = range(U.shape[0])

    # normed_change_of_base_unitary = np.abs(U.numpy())
    # pt.imshow(normed_change_of_base_unitary, cmap="viridis")
    # pt.title("Change of base unitary between Ising duals")
    # pt.xlabel("Dual Indices")
    # pt.ylabel("Original Indices")
    # pt.title("Change of base unitary magnitudes")
    # pt.colorbar()
    # pt.show()
