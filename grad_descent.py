"""Gradient-descent search for dual tensor product structures.

Random-restart batched gradient descent over the full 12-dimensional Ising
coefficient space, looking for points that are isospectral with a target
Hamiltonian (zero spectrum-difference loss). This complements the directional
fiber traversals in ising_finder.py, which are exhaustive along a single curve
but sensitive to the choice of direction vector.

Hamiltonian structure is imported from ising_finder.py; coefficient layout is
[XX, YY, ZZ, X, Y, Z, X_1, Y_1, Z_1, X_N, Y_N, Z_N].
"""

import pickle as pk
import numpy as np
import torch as tr
import matplotlib as mp
import matplotlib.pyplot as plt

from ising_finder import get_spectrum, get_loss_factory

np.set_printoptions(linewidth=10000000, threshold=1000000)
tr.set_printoptions(linewidth=1000)
tr.set_default_dtype(tr.float64)
seed = 1
rng = np.random.default_rng(seed)
np.random.seed(seed)

# number of Hamiltonian coefficients expected by get_Hamiltonian
N_PARAMS = 12


def random_inits(n_restarts, scale=1.0):
    """(K, 12) float64 batch of random restarts over the full coefficient space."""
    return tr.tensor(rng.normal(scale=scale, size=(n_restarts, N_PARAMS)))


def run_grad_descent(get_loss, c_init, n_steps=2000, lr=1e-2, tol=1e-8, log_every=100):
    """Descend every restart in parallel.

    c_init is (K, 12). Returns (c_best, loss_best, loss_history), where c_best is
    the (K, 12) best-seen point per restart, loss_best is (K,), and loss_history
    is (n_steps, 2) holding the min and median loss at each step.
    """
    c = c_init.clone().detach().requires_grad_(True)
    opt = tr.optim.Adam([c], lr=lr)

    c_best = c_init.clone().detach()
    loss_best = tr.full((c_init.shape[0],), np.inf)
    loss_history = np.empty((n_steps, 2))

    for step in range(n_steps):
        opt.zero_grad()
        loss = get_loss(c)  # (K,), rows are independent
        loss.sum().backward()

        with tr.no_grad():
            improved = loss < loss_best
            loss_best[improved] = loss[improved]
            c_best[improved] = c[improved]
            loss_history[step] = [loss.min().item(), loss.median().item()]

        if step % log_every == 0 or step == n_steps - 1:
            print(
                f"step {step:5d}: min {loss_best.min().item():.3e}"
                f"  median {loss.median().item():.3e}"
                f"  below tol {(loss_best < tol).sum().item()}/{len(loss_best)}"
            )

        opt.step()

    return c_best, loss_best, loss_history


def main():
    do_search = True

    n_restarts = 256
    n_steps = 2000
    tol = 1e-8
    output_file = f"results/grad_descent_seed{seed}.pkl"

    h = 1.5
    J = 1
    c_targ = tr.tensor([[0, 0, h, J, 0, 0, 0, 0, h, -J, 0, 0]])

    get_loss = get_loss_factory(c_targ)

    if do_search:
        C_init = random_inits(n_restarts)
        print(f"initial loss: min {get_loss(C_init).min().item():.3e}")

        C_final, losses, loss_history = run_grad_descent(
            get_loss, C_init, n_steps=n_steps, tol=tol
        )

        with open(output_file, "wb") as file:
            pk.dump(
                (
                    c_targ,
                    C_init.numpy(),
                    C_final.numpy(),
                    losses.numpy(),
                    loss_history,
                ),
                file,
            )

    with open(output_file, "rb") as file:
        (c_targ, C_init, C_final, losses, loss_history) = pk.load(file)

    print(f"\n{losses.shape[0]} restarts, final loss:")
    print(f"  min    {losses.min():.3e}")
    print(f"  median {np.median(losses):.3e}")
    print(f"  max    {losses.max():.3e}")

    # keep the isospectral points
    keep = losses < tol
    C_dual = C_final[keep]
    print(f"\n{C_dual.shape[0]} points with loss < {tol:.0e}")

    if C_dual.shape[0] > 0:
        # distance to the target, so trivial recovery of c_targ is visible
        dist = np.fabs(C_dual - c_targ.numpy()).max(axis=1)
        order = np.argsort(dist)
        print("max |c - c_targ| per point:")
        print(dist[order])
        print("coefficients:")
        print(C_dual[order])

        # confirm the spectra really do match, per point
        spec_err = (
            (get_spectrum(tr.tensor(C_dual)) - get_spectrum(c_targ))
            .abs()
            .max(dim=1)
            .values.numpy()
        )
        print("max spectrum deviation per point:")
        print(spec_err[order])

    plt.semilogy(loss_history[:, 0], "k-", label="min")
    plt.semilogy(loss_history[:, 1], "r-", label="median")
    plt.title(f"Gradient descent from {n_restarts} random restarts")
    plt.xlabel("Step")
    plt.ylabel("$||\\Lambda - \\Lambda_0||^2$")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()
