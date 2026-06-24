
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
seed = 1
rng = np.random.default_rng(seed)
np.random.seed(seed)


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

# Ising field parameters
h = 1.5
J = 1

def get_Hamiltonian(c):
    """Creates the minimal Hamiltonian
    c[K,0] = alpha
    c[K,1] = beta
    alpha = 0 beta = 1 => Ising
    alpha = 1 beta = 0 => KW dual
    H = (alpha * h + beta * J) Z_i Z_{i+1} + (alpha *J + beta * h) X_i
    - alpha J X_1 + (alpha *J + beta * h) X_n + alpha * h Z_n 
    """
    # c = [K,5], K is batch size
    K = c.shape[0]
    alpha = c[:,0].view(K,1,1)
    beta = c[:,1].view(K,1,1)
    I_list = [pauli[0]] * n_qb
    H = tr.zeros((K, 2**n_qb, 2**n_qb), dtype=tr.complex128)

    # Two-Local and One-Local terms
    for i in range(n_qb - 1):
        interaction_list = I_list[:]
        interaction_list[i] = pauli[3]
        interaction_list[i + 1] = pauli[3]


        single_term = I_list[:]
        single_term[i] = pauli[1]

        H += (alpha * h + beta * J) * kron_prod(interaction_list) + (alpha * J + beta * h) * kron_prod(single_term)

    # Boundary Terms
    site_one_x = I_list[:]
    site_one_x[0] = pauli[1]

    site_N_x = I_list[:]
    site_N_x[n_qb - 1] = pauli[1]

    site_N_z = I_list[:]
    site_N_z[n_qb - 1] = pauli[3]

    H += (-1*alpha * J * kron_prod(site_one_x) 
          + (alpha * J + beta * h) * kron_prod(site_N_x) 
          + alpha * h * kron_prod(site_N_z)
    )
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
        # c is (K, 5) tensor, K is batch size
        spectrum = get_spectrum(c)

        return tr.sum((spectrum - target_spectrum) ** 2, dim=1)  # (K,)

    return get_loss

def f_factory(get_loss):
    def f(v):
        # v is (5, K) numpy array, return (5, K) batch of gradients
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

def duplicates(U, v):
    return (np.fabs(U - v) < 0.1).all(axis=0)

def run_fiber(starting_point, direction, f, Df, get_loss, output_file):
    fiber_kwargs = {
        "f": f,
        "Df": Df,
        "ef": ef,
        "compute_step_amount": lambda trace: (0.01, 0, False),
        "v": starting_point,
        "c": direction,
        "terminate": lambda trace: (
            get_loss(tr.tensor(trace.x[:te].T)) > 1
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

    
    R = fx.get_unique_points(R, duplicates)

    with open(output_file, "wb") as file:
        pk.dump((starting_point, V, A, R), file)

def main():
    do_fiber = False
    do_perturb1 = True
    do_perturb2 = False

    # constant direction
    c_dir = np.array([[rng.normal() for _ in range(2)]]).T

    c_targ = tr.tensor([[0,1]], dtype=tr.complex128)

    get_loss = get_loss_factory(c_targ)
    f = f_factory(get_loss)
    Df = Df_factory(get_loss)

    KW_dual = tr.tensor([[1,0]])
    print(get_loss(KW_dual))

    # get initial fiber point
    v0 = c_targ.numpy().T

    if do_fiber:
        run_fiber(v0, c_dir, f, Df, get_loss, "results/minimal_ising.pkl")

    with open("results/minimal_ising.pkl", "rb") as file:
        (c_targ, V, A, R) = pk.load(file)

    # filter stationary points with non-zero loss
    loss = get_loss(tr.tensor(R.T))
    keep = loss.numpy() < 1e-7
    R = R[:, keep]
    loss = get_loss(tr.tensor(R.T))

    print(f"{R.shape[1]} optima")
    C = f(V)

    normal, kappa, t = nu.find_curvature_normal(V.T)
    curvature_peaks = 2

    # Step 1: Get the indices of elements that are NOT NaN
    valid_idx = np.where(~np.isnan(kappa))[0]

    # Step 2: Use argpartition on the valid subset elements
    # We use negative indexing (-k) to partition the largest values to the end
    sub_partition = np.argpartition(kappa[valid_idx], -curvature_peaks)[-curvature_peaks:]

    # Step 3: Map back to the original array's indices
    top_k_idx = valid_idx[sub_partition]

    # Step 4: Sort them if you need them in strict descending order
    top_k_idx = top_k_idx[np.argsort(-kappa[top_k_idx])]
    print(top_k_idx)

    # plt.plot(np.arange(0,len(kappa), 1), kappa)
    # plt.scatter(top_k_idx, kappa[top_k_idx], color="red")
    # plt.title("Curvature across fiber")
    # plt.xlabel("Index")
    # plt.ylabel("Curvature")
    # plt.show()

    # trace_loss = get_loss(tr.tensor(V.T))
    # plt.plot(trace_loss)
    # plt.xlabel("Index")
    # plt.ylabel("Loss")
    # plt.show()

    # plt.plot(*V, color="black", linestyle='-')
    # plt.quiver(*np.concatenate((V,C),axis=0),color='black')
    # plt.xlabel("alpha")
    # plt.ylabel("beta")
    # plt.show()

    #Peaks at 2902
    normal_dir = normal[2902, :]
    # Restart at around 2902
    if do_perturb1:
        init_vector = V[:, 2902] - normal_dir
        init_vector = np.atleast_2d(init_vector).T
        init_vector = tr.tensor([[1,0]], dtype=tr.complex128).T
        run_fiber(init_vector, c_dir, f, Df, get_loss, "results/minimal_ising_stitch_1.pkl")

    with open("results/minimal_ising_stitch_1.pkl", "rb") as file:
        (c_targ_1, V_1, A_1, R_1) = pk.load(file)

    # filter stationary points with non-zero loss
    loss = get_loss(tr.tensor(R_1.T))
    keep = loss.numpy() < 1e-7
    R_1 = R_1[:, keep]
    loss = get_loss(tr.tensor(R_1.T))
    print("Optima: ", R_1.shape)

    if do_perturb2:
        init_vector = V[:, 123] + normal_dir
        init_vector = np.atleast_2d(init_vector).T
        run_fiber(init_vector, c_dir, f, Df, get_loss, "results/ising_seed_1_stitch_2.pkl")

    #with open("results/ising_seed_1_stitch_2.pkl", "rb") as file:
    #   (c_targ_2, V_2, A_2, R_2) = pk.load(file)

    # filter stationary points with non-zero loss
    #loss = get_loss(tr.tensor(R_2.T))
    #keep = loss.numpy() < 1e-7
    #R_2 = R_2[:, keep]
    #loss = get_loss(tr.tensor(R_2.T))

    #print("Optima: ", R_2.shape)

    plt.subplot(1,2,1)
    plt.plot(range(2), R)
    plt.xlabel("Coefficient")
    plt.ylabel("Value")

    plt.subplot(1,2,2)
    plt.plot(range(2), R_1)
    plt.xlabel("Coefficient")
    plt.ylabel("Value")

    #plt.subplot(1,3,3)
    #plt.plot(range(5), R_2)
    #plt.xlabel("Coefficient")
    #plt.ylabel("Value")
    #plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()



    