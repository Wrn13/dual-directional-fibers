
import sys
import itertools as it
import pickle as pk
from typing import Callable
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
            get_loss(tr.tensor(trace.x.T)) > 1
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


def evaluate_loss_gradient_field(
    f: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float] = (-0.5, 1.5),
    y_range: tuple[float, float] = (-0.5, 1.5),
    n_grid: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Evaluate the loss-gradient vector field :math:`f=\nabla_c L` on a grid.

    The directional fiber is the set :math:`\{c : f(c)=\eta\,\hat c_\mathrm{dir}\}`
    (Katz & Reggia, IEEE TNNLS 2018), so this field is exactly the object the
    fiber is threaded through. Evaluation is a single *batched* call: the grid
    of :math:`n\_grid^2` points is passed as one ``(2, K)`` array, matching the
    contract of ``f`` produced by ``f_factory`` (in -> ``(2, K)``, out -> ``(2, K)``).

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Vector field from ``f_factory``. Maps parameter points of shape
        ``(2, K)`` (row 0 = ``alpha``, row 1 = ``beta``) to the gradient
        :math:`\nabla L` of shape ``(2, K)``.
    x_range, y_range : tuple[float, float]
        Inclusive bounds for the ``alpha`` (x) and ``beta`` (y) axes.
    n_grid : int
        Samples per axis; total ``n_grid**2`` field evaluations.

    Returns
    -------
    XX, YY : np.ndarray
        ``meshgrid`` coordinate arrays, shape ``(n_grid, n_grid)``.
    U, W : np.ndarray
        Gradient components :math:`\partial L/\partial\alpha` and
        :math:`\partial L/\partial\beta` on the grid, same shape as ``XX``.
    """
    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.linspace(y_range[0], y_range[1], n_grid)
    XX, YY = np.meshgrid(xs, ys)                    # (n_grid, n_grid)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=0)  # (2, n_grid**2)
    F = f(pts)                                      # (2, n_grid**2)
    U = F[0].reshape(XX.shape)
    W = F[1].reshape(XX.shape)
    return XX, YY, U, W

def _draw_fiber_vectors(
    ax,
    fiber: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    color: str,
    count: int,
    normalize: bool,
) -> None:
    r"""Overlay the field :math:`f(c)` sampled at points along one fiber.

    On the fiber, :math:`f(c)=\eta\,\hat c_\mathrm{dir}`; these arrows therefore
    lie along :math:`\pm\hat c_\mathrm{dir}`, flipping by :math:`180^\circ` at the
    sign changes of :math:`\eta` (the transverse zero crossings are the optima).

    Parameters
    ----------
    fiber : np.ndarray
        Fiber trace of shape ``(2, N)`` (row 0 = ``alpha``, row 1 = ``beta``).
    count : int
        Target number of arrows; the stride is ``N // count``.
    normalize : bool
        If ``True`` draw unit-length arrows (direction only); if ``False`` keep
        the raw :math:`\eta` magnitudes (shows how :math:`\eta` grows/flips).
    """
    C = f(fiber)                                   # (2, N), batched single call
    n = fiber.shape[1]
    stride = max(1, n // max(1, count))
    P = fiber[:, ::stride]
    Cs = C[:, ::stride]
    if normalize:
        mag = np.hypot(Cs[0], Cs[1])
        safe = np.isfinite(mag) & (mag > 0.0)
        denom = np.where(mag == 0.0, 1.0, mag)
        Cs = np.where(safe, Cs / denom, np.nan)
        ax.quiver(P[0], P[1], Cs[0], Cs[1], color=color,
                  angles="xy", scale_units="xy", scale=12,
                  width=0.006, zorder=4, alpha=0.95)
    else:
        ax.quiver(P[0], P[1], Cs[0], Cs[1], color=color,
                  angles="xy", scale=30, width=0.006, zorder=4, alpha=0.95)
        

def plot_fibers_and_field(
    V: np.ndarray,
    V_1: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    direction: np.ndarray | None = None,
    x_range: tuple[float, float] = (-0.5, 1.5),
    y_range: tuple[float, float] = (-0.5, 1.5),
    n_grid: int = 21,
    normalize: bool = True,
    descent: bool = False,
    output_path: str | None = None,
    show_fiber_vectors: bool = True,        # <-- draw f(c) at points along each fiber
    fiber_vector_count: int = 40,           # <-- approx arrows per fiber (sets the stride)
    fiber_vector_normalize: bool = True,    # <-- unit-length on-fiber arrows
) -> None:
    r"""Plot the two directional fibers over the loss-gradient vector field.

    Parameters
    ----------
    V, V_1 : np.ndarray
        Fiber traces of shape ``(2, N)`` and ``(2, M)`` as stored by
        ``run_fiber`` (the ``V`` arrays from ``minimal_ising.pkl`` and
        ``minimal_ising_stitch_1.pkl``). Row 0 = ``alpha``, row 1 = ``beta``.
    f : Callable[[np.ndarray], np.ndarray]
        Loss-gradient field; see :func:`evaluate_loss_gradient_field`.
    dir : np.ndarray
        Direction of the fiber
    x_range, y_range : tuple[float, float]
        Plot window. Fibers extending past these bounds are clipped by the axes.
    n_grid : int
        Arrows per axis for the quiver field.
    normalize : bool
        If ``True``, draw unit-length arrows (direction only) and encode
        :math:`\log_{10}\|\nabla L\|` as color. The gradient magnitude spans
        several decades, so raw-length arrows are unreadable; normalization is
        the standard remedy for phase-portrait-style plots.
    descent : bool
        If ``True``, draw :math:`-\nabla L` (gradient-descent flow, arrows point
        toward optima) instead of :math:`+\nabla L`.
    output_path : str | None
        If given, save to this path; otherwise ``plt.show()``.
    show_fiber_vectors: bool
        Draws f(c) for points along the fiber. Default True
    fiber_vector_count: int
        The number of arrows per fiber
    fiber_vector_normalize: bool
        Makes vectors normalized length
    """
    XX, YY, U, W = evaluate_loss_gradient_field(f, x_range, y_range, n_grid)
    if descent:
        U, W = -U, -W

    mag = np.hypot(U, W)
    finite = np.isfinite(mag)

    # Direction-only arrows; mask non-finite (e.g. autograd through eigvalsh at
    # near-degenerate spectra can return NaN) and the exact zeros at optima.
    if normalize:
        safe = finite & (mag > 0)
        denom = np.where(mag == 0.0, 1.0, mag)
        U = np.where(safe, U / denom, np.nan)
        W = np.where(safe, W / denom, np.nan)
    else:
        U = np.where(finite, U, np.nan)
        W = np.where(finite, W, np.nan)

    color = np.log10(np.where(finite, np.hypot(*[c for c in (U, W)]) * 0 + mag, np.nan) + 1e-12)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    q = ax.quiver(
        XX, YY, U, W, color,
        cmap="viridis", angles="xy", pivot="mid",
        scale=30 if normalize else None,
        width=0.004, alpha=0.9,
    )
    fig.colorbar(q, ax=ax, label=r"$\log_{10}\,\|\nabla L\|$", shrink=0.85)

    # The two directional fibers.
    ax.plot(V[0], V[1], color="black", lw=2.0, zorder=3,
            label=r"Fiber 1 (seed $(\alpha,\beta)=(0,1)$, Ising)")
    ax.plot(V_1[0], V_1[1], color="crimson", lw=2.0, zorder=3,
            label=r"Fiber 2 (seed $(\alpha,\beta)=(1,0)$, KW dual)")
    
    if show_fiber_vectors:
        _draw_fiber_vectors(ax, V,   f, "black",   fiber_vector_count, fiber_vector_normalize)
        _draw_fiber_vectors(ax, V_1, f, "crimson", fiber_vector_count, fiber_vector_normalize)

    # The isospectral pair: zeros of the field threaded by the fibers.
    ax.scatter([0.0, 1.0], [1.0, 0.0], s=70, marker="o",
               facecolors="white", edgecolors="black", linewidths=1.5,
               zorder=4)
    ax.annotate("Ising", (0.0, 1.0), textcoords="offset points",
                xytext=(8, 8), fontsize=11)
    ax.annotate("KW dual", (1.0, 0.0), textcoords="offset points",
                xytext=(8, 8), fontsize=11) 

    # Reference arrow for the fixed fiber direction \hat c_dir. The fiber is
    # gamma = {c : f(c) = eta * c_dir}, so along each fiber the field arrows
    # must be parallel (eta>0) or antiparallel (eta<0) to this vector.
    if direction is not None:
        d = np.asarray(direction, dtype=float).ravel()   # handles (2,1) or (2,)
        norm = np.linalg.norm(d)
        if norm == 0.0:
            raise ValueError("`direction` is the zero vector; cannot normalize.")
        d = d / norm

        L = 0.3  # arrow length in data units

        direction_anchor = (.5,.5)
        ax.quiver(
            direction_anchor[0], direction_anchor[1], d[0] * L, d[1] * L,
            angles="xy", scale_units="xy", scale=1,
            color="black", width=0.011, zorder=5,
        )
        ax.scatter(*direction_anchor, s=20, color="black", zorder=6)
        ax.annotate(
            r"$\hat c_\mathrm{dir}$",
            np.asarray(direction_anchor) + d * L,
            textcoords="offset points", xytext=(6, 6),
            color="black", fontsize=12,
        )


    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"Directional fibers over $\nabla_c L$"
                 + (r"  ($-\nabla L$ shown)" if descent else ""))
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()

def main():
    do_fiber = False
    do_perturb1 = False
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

    plt.plot(np.arange(0,len(kappa), 1), kappa)
    plt.scatter(top_k_idx, kappa[top_k_idx], color="red")
    plt.title("Curvature across fiber")
    plt.xlabel("Index")
    plt.ylabel("Curvature")
    plt.show()

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
    normal_dir = normal[7126, :]
    # Restart at around 2902
    if do_perturb1:
        init_vector = V[:, 7126] - normal_dir
        init_vector = np.atleast_2d(init_vector).T
       # init_vector = tr.tensor([[1,0]], dtype=tr.complex128).T
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

    # plt.subplot(1,2,1)
    # plt.plot(range(2), R)
    # plt.xlabel("Coefficient")
    # plt.ylabel("Value")

    # plt.subplot(1,2,2)
    # plt.plot(range(2), R_1)
    # plt.xlabel("Coefficient")
    # plt.ylabel("Value")

    #plt.subplot(1,3,3)
    #plt.plot(range(5), R_2)
    #plt.xlabel("Coefficient")
    #plt.ylabel("Value")
    #plt.tight_layout()
    plt.show()

    # f is already built above: f = f_factory(get_loss)
    plot_fibers_and_field(V, V_1, f, c_dir, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5),
                          n_grid=50, normalize=True)

if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()



    