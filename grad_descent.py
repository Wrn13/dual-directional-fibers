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
from matplotlib.colors import LinearSegmentedColormap

from ising_finder import get_spectrum, get_loss_factory, n_qb

np.set_printoptions(linewidth=10000000, threshold=1000000)
tr.set_printoptions(linewidth=1000)
tr.set_default_dtype(tr.float64)
seed = 1
rng = np.random.default_rng(seed)
np.random.seed(seed)

# number of Hamiltonian coefficients expected by get_Hamiltonian
N_PARAMS = 12

# operator each coefficient multiplies, in get_Hamiltonian's order
COEFF_LABELS = [
    "$XX$", "$YY$", "$ZZ$", "$X$", "$Y$", "$Z$",
    "$X_1$", "$Y_1$", "$Z_1$", "$X_N$", "$Y_N$", "$Z_N$",
]

# Categorical series slots, assigned in this fixed order and never cycled.
# Validated for CVD separation and lightness/chroma against a white surface.
SERIES_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
# Shape pairs with hue so identity never rests on color alone (three of the
# slots above sit below 3:1 contrast on white); the printed coefficient table
# is the accompanying table view.
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MAX_SERIES = len(SERIES_COLORS)

# Chart chrome
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Diverging pair for signed coefficients: warm/cool poles, neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "coeff_diverging", ["#2a78d6", "#f0efec", "#e34948"]
)


def random_inits(n_restarts, scale=1.0):
    """(K, 12) float64 batch of random restarts over the full coefficient space."""
    return tr.tensor(rng.normal(scale=scale, size=(n_restarts, N_PARAMS)))


def run_grad_descent(get_loss, c_init, n_steps=2000, lr=1e-2, tol=1e-8, log_every=100):
    """Descend every restart in parallel.

    c_init is (K, 12). Returns (c_best, loss_best, loss_history), where c_best is
    the (K, 12) best-seen point per restart, loss_best is (K,), and loss_history
    is (n_steps, K) holding every restart's own loss at each step.
    """
    c = c_init.clone().detach().requires_grad_(True)
    opt = tr.optim.Adam([c], lr=lr)

    c_best = c_init.clone().detach()
    loss_best = tr.full((c_init.shape[0],), np.inf)
    loss_history = np.empty((n_steps, c_init.shape[0]))

    for step in range(n_steps):
        opt.zero_grad()
        loss = get_loss(c)  # (K,), rows are independent
        loss.sum().backward()

        with tr.no_grad():
            improved = loss < loss_best
            loss_best[improved] = loss[improved]
            c_best[improved] = c[improved]
            loss_history[step] = loss.numpy()

        if step % log_every == 0 or step == n_steps - 1:
            print(
                f"step {step:5d}: min {loss_best.min().item():.3e}"
                f"  median {loss.median().item():.3e}"
                f"  below tol {(loss_best < tol).sum().item()}/{len(loss_best)}"
            )

        opt.step()

    return c_best, loss_best, loss_history


def _style_axes(ax, ygrid=True):
    """Recessive grid and axes so the data carries the figure."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def plot_dual_coefficients(C_dual, c_targ, style=None, ax=None):
    """Plot the coefficients of every dual found, against the target.

    C_dual is (n_duals, 12) and c_targ is (1, 12) or (12,). `style` picks the
    form, and by default follows the dual count, since what the reader can
    actually do with the figure changes with it:

      "profile"      - one marker series per dual across the 12 named
                       operators. Default up to 8 duals; past that the
                       categorical slots would have to be cycled.
      "heatmap"      - duals x coefficients grid on a diverging scale. Default
                       from 9 to 20, where each dual still gets a readable row.
      "distribution" - where every dual's value for each operator lands, as a
                       jittered strip. Default past 20, where per-dual identity
                       stops being legible and the population is the point.
    """
    c_targ = np.asarray(c_targ).reshape(-1)
    n_duals = C_dual.shape[0]

    if style is None:
        if n_duals <= MAX_SERIES:
            style = "profile"
        elif n_duals <= 20:
            style = "heatmap"
        else:
            style = "distribution"

    if ax is None:
        width = 8.6 if style == "profile" else 6.5
        height = 3.6
        if style == "heatmap":
            # keep rows thick enough to read
            height = min(3.6 + 0.16 * n_duals, 7.5)
        _, ax = plt.subplots(figsize=(width, height), layout="constrained")

    x = np.arange(N_PARAMS)

    if style == "profile":
        shown = min(n_duals, MAX_SERIES)
        if n_duals > MAX_SERIES:
            suggestion = "heatmap" if n_duals <= 20 else "distribution"
            print(
                f"note: showing the first {MAX_SERIES} of {n_duals} duals; "
                f"use style={suggestion!r} for all of them"
            )

        ax.axhline(0, color=BASELINE, linewidth=1, zorder=0)

        # target is context, not a series, so it wears ink rather than a hue
        ax.plot(
            x, c_targ, "--", color=INK, linewidth=2, zorder=2,
            label="target (Ising)",
        )

        for i in range(shown):
            ax.plot(
                x, C_dual[i],
                color=SERIES_COLORS[i], marker=SERIES_MARKERS[i],
                markersize=7, linewidth=1.6, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"dual {i}",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(COEFF_LABELS)
        ax.set_xlim(-0.4, N_PARAMS - 0.6)
        ax.set_ylabel("Coefficient value")
        ax.set_xlabel("Hamiltonian term")
        title = f"Coefficients of {n_duals} isospectral duals"
        if shown < n_duals:
            title = f"Coefficients of {shown} of {n_duals} isospectral duals"
        ax.set_title(title, color=INK)
        # outside the axes: the data range varies run to run, so any in-axes
        # placement eventually lands on top of a series
        ax.legend(
            frameon=False, fontsize=8, labelcolor=INK,
            loc="upper left", bbox_to_anchor=(1.01, 1.0),
        )
        _style_axes(ax)

    elif style == "heatmap":
        # target stacked on top so the comparison is immediate
        grid = np.vstack([c_targ, C_dual])
        n_rows = grid.shape[0]
        limit = np.fabs(grid).max()

        # the gap between cells only reads as a gap while the cells are thick
        # enough to survive it
        edge = 1.5 if n_rows <= 20 else 0.0

        mesh = ax.pcolormesh(
            grid, cmap=DIVERGING, vmin=-limit, vmax=limit,
            edgecolors="white", linewidth=edge,
        )
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Coefficient value", color=INK)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=MUTED, labelcolor=INK)

        # separate the reference row from the duals
        ax.axhline(1, color=INK, linewidth=1.5)

        if n_rows <= 20:
            yticks = np.arange(n_rows) + 0.5
            ylabels = ["target"] + [f"dual {i}" for i in range(n_duals)]
        else:
            # thin the labels rather than let 100+ of them collide
            step = int(np.ceil(n_duals / 10))
            kept = range(0, n_duals, step)
            yticks = np.array([0] + [1 + i for i in kept]) + 0.5
            ylabels = ["target"] + [f"dual {i}" for i in kept]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        ax.set_xticks(x + 0.5)
        ax.set_xticklabels(COEFF_LABELS)
        ax.invert_yaxis()
        ax.set_xlabel("Hamiltonian term")
        ax.set_title(f"Coefficients of {n_duals} isospectral duals", color=INK)
        _style_axes(ax, ygrid=False)

    elif style == "distribution":
        # one series (the population of duals), so one hue; the target is the
        # reference it is read against
        ax.axhline(0, color=BASELINE, linewidth=1, zorder=0)

        # own generator: cosmetic jitter must not advance the search's stream
        jitter = np.random.default_rng(0).uniform(-0.28, 0.28, size=C_dual.shape)
        ax.scatter(
            x + jitter, C_dual,
            s=14, color=SERIES_COLORS[0], alpha=0.35, linewidths=0,
            zorder=2, label=f"{n_duals} duals",
        )
        ax.plot(
            x, c_targ, "_", color=INK, markersize=18, markeredgewidth=2.5,
            zorder=3, label="target (Ising)",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(COEFF_LABELS)
        ax.set_xlim(-0.5, N_PARAMS - 0.5)
        ax.set_ylabel("Coefficient value")
        ax.set_xlabel("Hamiltonian term")
        ax.set_title(
            f"Coefficient spread across {n_duals} isospectral duals", color=INK
        )
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")
        _style_axes(ax)

    else:
        raise ValueError(
            f"unknown style {style!r}, expected 'profile', 'heatmap' "
            "or 'distribution'"
        )

    return ax


def plot_loss_history(loss_history, ax=None):
    """One line per restart, showing where each descent ended up.

    loss_history is (n_steps, K). Up to 8 restarts each get their own
    categorical hue; past that the restarts stop being individually
    identifiable and become a single translucent population.
    """
    n_steps, n_restarts = loss_history.shape

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")

    if n_restarts <= MAX_SERIES:
        for k in range(n_restarts):
            ax.semilogy(
                loss_history[:, k], color=SERIES_COLORS[k], linewidth=2,
                label=f"restart {k}",
            )
        ax.legend(
            frameon=False, fontsize=8, labelcolor=INK,
            loc="upper left", bbox_to_anchor=(1.01, 1.0),
        )
    else:
        # one hue, thin and translucent: the individual restart is no longer
        # the unit of interest, the spread of trajectories is
        ax.semilogy(
            loss_history, color=SERIES_COLORS[0], linewidth=0.8, alpha=0.12,
        )

    ax.set_title(f"Loss per restart over {n_restarts} restarts", color=INK)
    ax.set_xlabel("Step")
    ax.set_ylabel("$||\\Lambda - \\Lambda_0||^2$")
    ax.set_xlim(0, n_steps - 1)
    _style_axes(ax)
    return ax


def main():
    do_search = True

    n_restarts = 4
    n_steps = 2000
    tol = 1e-8
    # n_qb is part of the key: the spectrum, and so every stored loss, is only
    # meaningful for the chain length the search ran at
    output_file = f"results/grad_descent_n{n_qb}_seed{seed}.pkl"

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
                    n_qb,
                    c_targ,
                    C_init.numpy(),
                    C_final.numpy(),
                    losses.numpy(),
                    loss_history,
                ),
                file,
            )

    with open(output_file, "rb") as file:
        (n_qb_saved, c_targ, C_init, C_final, losses, loss_history) = pk.load(file)

    if n_qb_saved != n_qb:
        raise ValueError(
            f"{output_file} was written with n_qb={n_qb_saved} but ising_finder "
            f"is now n_qb={n_qb}; the stored losses do not describe this "
            "Hamiltonian. Re-run with do_search = True."
        )

    if loss_history.shape[1] != losses.shape[0]:
        raise ValueError(
            f"{output_file} holds a {loss_history.shape} loss history for "
            f"{losses.shape[0]} restarts; it predates the per-restart history. "
            "Re-run with do_search = True."
        )

    # keep the isospectral points
    keep = losses < tol
    C_dual = C_final[keep]
    print(f"\n{C_dual.shape[0]} points with loss < {tol:.0e}")

    if C_dual.shape[0] > 0:
        print("coefficients:")
        print(C_dual)

        # confirm the spectra really do match, per point
        spec_err = (
            (get_spectrum(tr.tensor(C_dual)) - get_spectrum(c_targ))
            .abs()
            .max(dim=1)
            .values.numpy()
        )
        print("max spectrum deviation per point:")
        print(spec_err)

        plot_dual_coefficients(C_dual, c_targ.numpy())

    plot_loss_history(loss_history)
    plt.show()


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()
