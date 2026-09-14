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

# eigenvalues closer than this count as one degenerate level. Well above the
# ~1e-8 spread eigvalsh leaves on a converged dual, well below the level
# spacing of the Ising spectra this searches over.
DEGEN_TOL = 1e-6


def random_inits(n_restarts, scale=1.0):
    """(K, 12) float64 batch of random restarts over the full coefficient space."""
    return tr.tensor(rng.normal(scale=scale, size=(n_restarts, N_PARAMS)))


def perturbed_inits(c_center, n_restarts, scale=0.1):
    """(K, 12) float64 batch of small random perturbations around one point.

    c_center is (1, 12) or (12,). Every restart is c_center plus isotropic
    Gaussian noise of the given scale, so the restarts probe the neighbourhood
    of a single point rather than the whole coefficient space.
    """
    c_center = tr.as_tensor(c_center, dtype=tr.get_default_dtype()).reshape(1, -1)
    if c_center.shape[1] != N_PARAMS:
        raise ValueError(
            f"c_center has {c_center.shape[1]} coefficients, expected {N_PARAMS}"
        )
    noise = tr.tensor(rng.normal(scale=scale, size=(n_restarts, N_PARAMS)))
    return c_center + noise


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


def _dual_line_style(i, n_duals):
    """Per-dual line style: categorical while identity is legible, else one hue.

    Returns kwargs for plotting dual i of n_duals. Mirrors _loss_line_style, so
    a dual keeps the same hue in both figures while the counts are small.
    """
    if n_duals <= MAX_SERIES:
        return dict(
            color=SERIES_COLORS[i], marker=SERIES_MARKERS[i],
            markersize=7, linewidth=1.6,
            markeredgecolor="white", markeredgewidth=0.8,
            label=f"dual {i}",
        )
    # the individual dual is no longer the unit of interest, the spread is
    return dict(color=SERIES_COLORS[0], linewidth=0.9, alpha=0.35)


def plot_dual_coefficients(C_dual, c_targ, style=None, ax=None):
    """Plot the coefficients of every dual found, against the target.

    C_dual is (n_duals, 12) and c_targ is (1, 12) or (12,). `style` picks the
    form:

      "profile"      - one line per dual across the 12 named operators. The
                       default at any count: up to 8 duals each get their own
                       categorical hue and marker, past that they keep being
                       drawn as a single translucent population.
      "distribution" - where every dual's value for each operator lands, as a
                       jittered strip. Opt in when the population, rather than
                       any individual dual, is the point.
    """
    c_targ = np.asarray(c_targ).reshape(-1)
    n_duals = C_dual.shape[0]
    # the live search draws this at one dual, so the singular case is common
    noun = "dual" if n_duals == 1 else "duals"

    if style is None:
        style = "profile"

    if ax is None:
        # the profile legend sits outside the axes, so it needs the extra width
        width = 8.6 if style == "profile" and n_duals <= MAX_SERIES else 6.5
        _, ax = plt.subplots(figsize=(width, 3.6), layout="constrained")

    x = np.arange(N_PARAMS)

    if style == "profile":
        ax.axhline(0, color=BASELINE, linewidth=1, zorder=0)

        # target is context, not a series, so it wears ink rather than a hue
        ax.plot(
            x, c_targ, "--", color=INK, linewidth=2, zorder=2,
            label="target (Ising)",
        )

        for i in range(n_duals):
            ax.plot(x, C_dual[i], zorder=3, **_dual_line_style(i, n_duals))

        ax.set_xticks(x)
        ax.set_xticklabels(COEFF_LABELS)
        ax.set_xlim(-0.4, N_PARAMS - 0.6)
        ax.set_ylabel("Coefficient value")
        ax.set_xlabel("Hamiltonian term")
        ax.set_title(f"Coefficients of {n_duals} isospectral {noun}", color=INK)
        # outside the axes: the data range varies run to run, so any in-axes
        # placement eventually lands on top of a series
        if n_duals <= MAX_SERIES:
            ax.legend(
                frameon=False, fontsize=8, labelcolor=INK,
                loc="upper left", bbox_to_anchor=(1.01, 1.0),
            )
        _style_axes(ax)

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
            f"Coefficient spread across {n_duals} isospectral {noun}", color=INK
        )
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")
        _style_axes(ax)

    else:
        raise ValueError(
            f"unknown style {style!r}, expected 'profile' or 'distribution'"
        )

    return ax


def spectrum_degeneracy(spectrum, tol=DEGEN_TOL):
    """Group one spectrum's eigenvalues into degenerate levels.

    spectrum is (2**n_qb,), sorted or not. Returns (levels, multiplicities):
    the mean energy of each run of eigenvalues whose neighbours sit within tol,
    and how many eigenvalues that level holds. Grouping is by neighbouring gap,
    so a level can be wider than tol if its members chain across it - at the
    tolerances here that only happens for spectra that are near-degenerate
    anyway, which is worth seeing as one level.
    """
    lam = np.sort(np.asarray(spectrum).reshape(-1))
    starts = np.flatnonzero(np.diff(lam) > tol) + 1
    groups = np.split(lam, starts)
    levels = np.array([g.mean() for g in groups])
    mult = np.array([g.size for g in groups])
    return levels, mult


def degeneracy_pattern(spectrum, tol=DEGEN_TOL):
    """Multiplicities as a compact string, e.g. "1-2-2-2-1", low level first."""
    _, mult = spectrum_degeneracy(spectrum, tol)
    return "-".join(str(m) for m in mult)


def print_degeneracy(spectrum, label="target", tol=DEGEN_TOL, absolute=False):
    """Print the level table of one spectrum: energy, multiplicity, spread.

    With absolute, the table is over |lambda| instead, where a level of two
    counts a +/- pair rather than a true degeneracy.
    """
    lam = np.sort(np.abs(np.asarray(spectrum).reshape(-1)) if absolute
                  else np.asarray(spectrum).reshape(-1))
    levels, mult = spectrum_degeneracy(lam, tol)
    what = "|spectrum|" if absolute else "spectrum"
    print(
        f"{label} {what}: {lam.size} states in {levels.size} levels"
        f"  [{degeneracy_pattern(lam, tol)}]"
    )
    start = 0
    for lv, m in zip(levels, mult):
        members = lam[start : start + m]
        start += m
        # the width of a supposedly degenerate level is how much to trust it
        width = "" if m == 1 else f"   (width {np.ptp(members):.2e})"
        print(f"  {lv:+.9f}  x{m}{width}")


def reflection_residual(spectrum):
    """How far the spectrum is from being symmetric under lambda -> -lambda.

    Sorted ascending, a spectrum with that symmetry satisfies
    lambda_i = -lambda_{n-1-i}, so this returns the largest violation of that
    pairing. Zero to numerical precision means every eigenvalue has a partner
    of equal magnitude and opposite sign, which is exactly the case where the
    |lambda| view collapses to half as many levels as the signed one.
    """
    lam = np.sort(np.asarray(spectrum).reshape(-1))
    return float(np.abs(lam + lam[::-1]).max())


def print_symmetry(spectrum, label="target", tol=DEGEN_TOL):
    """Report the lambda -> -lambda pairing, then the |lambda| level table."""
    residual = reflection_residual(spectrum)
    verdict = "symmetric" if residual <= tol else "NOT symmetric"
    print(
        f"{label} spectrum is {verdict} under lambda -> -lambda"
        f"  (max pairing residual {residual:.2e})"
    )
    print_degeneracy(spectrum, label=label, tol=tol, absolute=True)


def _fanned_levels(spectrum, x0, tol, width=0.16):
    """(x, sorted eigenvalues) with each degenerate multiplet spread sideways.

    Coincident eigenvalues draw exactly on top of each other, which is
    precisely the thing being counted, so members of a level fan out around x0.
    """
    lam = np.sort(np.asarray(spectrum).reshape(-1))
    starts = np.flatnonzero(np.diff(lam) > tol) + 1
    x = np.empty(lam.size)
    for g in np.split(np.arange(lam.size), starts):
        x[g] = x0 if g.size == 1 else x0 + np.linspace(-width, width, g.size)
    return x, lam


def _degeneracy_marker_style(i, n_duals):
    """Per-dual marker style, matching the hue that dual wears elsewhere."""
    if n_duals <= MAX_SERIES:
        return dict(
            color=SERIES_COLORS[i], marker=SERIES_MARKERS[i], markersize=6,
            linestyle="none", markeredgecolor="white", markeredgewidth=0.6,
        )
    # the individual dual is no longer the unit of interest, the spread is
    return dict(
        color=SERIES_COLORS[0], marker="o", markersize=4,
        linestyle="none", alpha=0.35, markeredgewidth=0,
    )


def plot_spectrum_degeneracy(
    spec_targ, spec_duals=None, tol=DEGEN_TOL, ax=None, absolute=False
):
    """Energy-level view of the target spectrum and every dual's spectrum.

    spec_targ is (2**n_qb,) or (1, 2**n_qb); spec_duals is (n_duals, 2**n_qb)
    or None for the target alone. The target is drawn as one horizontal line
    per distinct level, labelled with its multiplicity where that is more than
    one - an unlabelled line is a singlet, which the title's state and level
    counts confirm. Each dual gets its own column of eigenvalue markers, with
    degenerate multiplets fanned out so a doubly occupied level reads as two
    markers rather than one. Past 8 duals they share a single jittered column.

    With absolute, the same view is drawn over |lambda|. Levels that only
    appear there are +/- pairs rather than degeneracies: a spectrum symmetric
    under lambda -> -lambda folds onto half as many levels, all of them even,
    so the two views side by side separate that symmetry from real degeneracy.
    """
    spec_targ = np.asarray(spec_targ).reshape(-1)

    S = (
        np.empty((0, spec_targ.size))
        if spec_duals is None
        else np.asarray(spec_duals).reshape(-1, spec_targ.size)
    )
    if absolute:
        spec_targ = np.abs(spec_targ)
        S = np.abs(S)

    levels, mult = spectrum_degeneracy(spec_targ, tol)
    n_duals = S.shape[0]
    n_cols = n_duals if n_duals <= MAX_SERIES else 1

    if ax is None:
        # floor keeps the title readable when there is only one dual column
        _, ax = plt.subplots(
            figsize=(max(4.6, 3.6 + 0.7 * n_cols), 3.8), layout="constrained"
        )

    half = 0.42
    for lv, m in zip(levels, mult):
        # target is context, not a series, so it wears ink rather than a hue
        ax.plot([-half, half], [lv, lv], color=INK, linewidth=2, zorder=3)
        if m > 1:
            ax.annotate(
                f"$\\times${m}", xy=(-half - 0.06, lv), ha="right", va="center",
                fontsize=9, color=INK, zorder=4,
            )

    x_cols = np.arange(1, n_cols + 1)
    if n_duals and n_duals <= MAX_SERIES:
        for i in range(n_duals):
            x, lam = _fanned_levels(S[i], x_cols[0] + i, tol)
            ax.plot(x, lam, zorder=2, **_degeneracy_marker_style(i, n_duals))
        col_labels = [f"dual {i}" for i in range(n_duals)]
    elif n_duals:
        # own generator: cosmetic jitter must not advance the search's stream
        jitter = np.random.default_rng(0).uniform(-0.3, 0.3, size=S.shape)
        ax.plot(
            (1 + jitter).ravel(), S.ravel(), zorder=2,
            **_degeneracy_marker_style(0, n_duals),
        )
        col_labels = [f"{n_duals} duals"]
    else:
        col_labels = []

    ax.set_xticks([0, *x_cols])
    ax.set_xticklabels(["target", *col_labels])
    ax.set_xlim(-1.15, n_cols + 0.6)
    ax.set_ylabel("$|\\lambda_i|$" if absolute else "Eigenvalue $\\lambda_i$")
    ax.set_xlabel(f"levels grouped within {tol:g}")
    # the multiplicity labels already carry the pattern; the title only has to
    # say how many states collapsed into how few levels
    ax.set_title(
        f"{'Magnitude' if absolute else 'Spectrum'} degeneracy: "
        f"{spec_targ.size} states in {levels.size} levels",
        color=INK, fontsize=11,
    )
    _style_axes(ax)
    return ax


def plot_spectrum_views(spec_targ, spec_duals=None, tol=DEGEN_TOL, axes=None):
    """The signed and absolute degeneracy views side by side.

    Reading them together is what separates the two ways levels can coincide:
    degeneracy proper shows up in both panels, while a level that only appears
    in |lambda| is a +/- pair, i.e. the spectrum's symmetry under
    lambda -> -lambda. Returns the two axes.
    """
    if axes is None:
        n_duals = 0 if spec_duals is None else np.asarray(spec_duals).reshape(
            -1, np.asarray(spec_targ).size
        ).shape[0]
        n_cols = n_duals if n_duals <= MAX_SERIES else 1
        width = max(4.6, 3.6 + 0.7 * n_cols)
        _, axes = plt.subplots(
            1, 2, figsize=(2 * width, 3.8), layout="constrained"
        )

    plot_spectrum_degeneracy(spec_targ, spec_duals, tol=tol, ax=axes[0])
    plot_spectrum_degeneracy(
        spec_targ, spec_duals, tol=tol, ax=axes[1], absolute=True
    )
    return axes


def _setup_loss_axes(ax, n_steps):
    """Chrome shared by the finished-history and step-through loss figures."""
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("$||\\Lambda - \\Lambda_0||^2$")
    ax.set_xlim(0, n_steps - 1)
    _style_axes(ax)


def _loss_line_style(k, n_restarts):
    """Per-restart line style: categorical while identity is legible, else one hue.

    Returns kwargs for plotting restart k of n_restarts.
    """
    if n_restarts <= MAX_SERIES:
        return dict(
            color=SERIES_COLORS[k], linewidth=2, label=f"restart {k}",
        )
    # the individual restart is no longer the unit of interest, the spread is
    return dict(color=SERIES_COLORS[0], linewidth=0.8, alpha=0.12)


def _loss_legend(ax, n_restarts):
    if n_restarts <= MAX_SERIES:
        ax.legend(
            frameon=False, fontsize=8, labelcolor=INK,
            loc="upper left", bbox_to_anchor=(1.01, 1.0),
        )


def plot_loss_history(loss_history, ax=None):
    """One line per restart, showing where each descent ended up.

    loss_history is (n_steps, K). Up to 8 restarts each get their own
    categorical hue; past that the restarts stop being individually
    identifiable and become a single translucent population.
    """
    n_steps, n_restarts = loss_history.shape

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")

    for k in range(n_restarts):
        ax.plot(loss_history[:, k], **_loss_line_style(k, n_restarts))
    _loss_legend(ax, n_restarts)

    ax.set_title(f"Loss per restart over {n_restarts} restarts", color=INK)
    _setup_loss_axes(ax, n_steps)
    return ax


def run_grad_descent_sequential(
    get_loss, c_init, n_steps=2000, lr=1e-2, tol=1e-8, log_every=100,
    c_targ=None, ax=None, deg_tol=DEGEN_TOL,
):
    """Descend one restart at a time, adding each finished curve to a live figure.

    Each restart runs to completion through run_grad_descent on a batch of one,
    then its loss curve is drawn and the run blocks for a keypress or click
    before the next restart starts. Returns the same three values with the same
    shapes as run_grad_descent, so callers and the saved artifact are unaffected.

    Pass c_targ to also get two more live figures: the duals' coefficients, and
    the degeneracy of their spectra against the target's, in both the signed
    and the |lambda| view. Both grow whenever a restart lands below tol. The
    degeneracy figure opens on the target alone, before the first restart,
    since its level structure is what the duals are being read against;
    deg_tol sets how close eigenvalues must sit to count as one level.
    """
    n_restarts = c_init.shape[0]

    # no event loop under a headless backend, so blocking there would hang
    interactive = not mp.get_backend().lower().startswith("agg")
    if interactive:
        plt.ion()

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
    _setup_loss_axes(ax, n_steps)

    c_best = tr.empty_like(c_init)
    loss_best = tr.empty(n_restarts)
    loss_history = np.empty((n_steps, n_restarts))

    duals = []
    coeff_ax = None

    dual_spectra = []
    deg_axes = None
    spec_targ = None
    if c_targ is not None:
        spec_targ = get_spectrum(
            tr.as_tensor(np.asarray(c_targ), dtype=tr.get_default_dtype())
            .reshape(1, -1)
        )[0].numpy()
        # sized for every dual this run could add, since the figure is reused
        # (cleared and redrawn) rather than remade as the columns arrive
        deg_width = max(4.6, 3.6 + 0.7 * min(n_restarts, MAX_SERIES))
        _, deg_axes = plt.subplots(
            1, 2, figsize=(2 * deg_width, 3.8), layout="constrained"
        )
        plot_spectrum_views(spec_targ, tol=deg_tol, axes=deg_axes)

    for k in range(n_restarts):
        print(f"\n--- restart {k}/{n_restarts} ---")
        c_k, loss_k, history_k = run_grad_descent(
            get_loss, c_init[k : k + 1],
            n_steps=n_steps, lr=lr, tol=tol, log_every=log_every,
        )

        c_best[k] = c_k[0]
        loss_best[k] = loss_k[0]
        loss_history[:, k] = history_k[:, 0]

        ax.plot(loss_history[:, k], **_loss_line_style(k, n_restarts))
        _loss_legend(ax, n_restarts)
        # converged curves reach ~1e-29, so the y range has to grow with them
        ax.relim()
        ax.autoscale_view()

        if loss_best[k].item() < tol:
            duals.append(c_best[k].numpy().copy())
            print(f"  dual {len(duals) - 1}: {duals[-1]}")

            if c_targ is not None:
                if coeff_ax is None:
                    _, coeff_ax = plt.subplots(
                        figsize=(8.6, 3.6), layout="constrained"
                    )
                # redraw every dual rather than append one line: the styling
                # switches from per-dual hues to a single population once the
                # count passes MAX_SERIES, which the earlier lines share in
                coeff_ax.clear()
                plot_dual_coefficients(np.array(duals), c_targ, ax=coeff_ax)

                spec_k = get_spectrum(c_best[k : k + 1])[0].numpy()
                dual_spectra.append(spec_k)
                # same reason as the coefficient figure: the styling switches
                # from per-dual hues to one population past MAX_SERIES
                for deg_ax in deg_axes:
                    deg_ax.clear()
                plot_spectrum_views(
                    spec_targ, np.array(dual_spectra), tol=deg_tol, axes=deg_axes
                )

                # isospectral points must share the target's levels; a mismatch
                # means the dual is still short of converged, or two levels sit
                # closer together than deg_tol can tell apart
                for what, spec_pair in (
                    ("degeneracy", (spec_k, spec_targ)),
                    ("|degeneracy|", (np.abs(spec_k), np.abs(spec_targ))),
                ):
                    pattern, targ_pattern = (
                        degeneracy_pattern(s, deg_tol) for s in spec_pair
                    )
                    agrees = "matches target" if pattern == targ_pattern else (
                        f"DIFFERS from target {targ_pattern}"
                    )
                    print(f"  {what} {pattern} ({agrees})")
                print(
                    f"  reflection residual {reflection_residual(spec_k):.2e}"
                    f" (target {reflection_residual(spec_targ):.2e})"
                )

        last = k == n_restarts - 1
        hint = "" if last else " - press any key for the next"
        ax.set_title(f"restart {k + 1}/{n_restarts} done{hint}", color=INK)

        if interactive:
            figs = [ax.figure]
            if coeff_ax is not None:
                figs.append(coeff_ax.figure)
            if deg_axes is not None:
                figs.append(deg_axes[0].figure)
            for fig in figs:
                fig.canvas.draw_idle()
            plt.pause(0.001)
            if not last:
                plt.waitforbuttonpress()

    ax.set_title(f"Loss per restart over {n_restarts} restarts", color=INK)
    if interactive:
        plt.ioff()

    return c_best, loss_best, loss_history


def main():
    do_search = True
    # one restart at a time, adding each curve as it finishes
    step_through = True

    n_restarts = 12
    n_steps = 2000
    tol = 1e-12
    # how close two eigenvalues must sit to be read as one degenerate level
    deg_tol = DEGEN_TOL
    # width of the restart cloud around c_center; set c_center = None to go back
    # to restarts drawn over the whole coefficient space
    perturb_scale = 0.1

    h = 1.5
    J = 1
    c_targ = tr.tensor([[0, 0, h, J, 0, 0, 0, 0, h, -J, 0, 0]])

    # point the restarts cluster around; the target itself asks which duals sit
    # near it, and is a fixed point of the loss, so expect restarts that fall
    # straight back to it alongside any genuinely distinct duals
    c_center = c_targ

    # n_qb is part of the key: the spectrum, and so every stored loss, is only
    # meaningful for the chain length the search ran at. The restart cloud is
    # too, so a local run does not overwrite a global one.
    local = "" if c_center is None else f"_local{perturb_scale:g}"
    output_file = f"results/grad_descent_n{n_qb}_seed{seed}{local}.pkl"

    get_loss = get_loss_factory(c_targ)

    # the level structure every dual has to reproduce, printed before the
    # search so the duals arriving below can be read against it
    print_degeneracy(get_spectrum(c_targ)[0].numpy(), label="target", tol=deg_tol)
    print_symmetry(get_spectrum(c_targ)[0].numpy(), label="target", tol=deg_tol)

    if do_search:
        if c_center is None:
            C_init = random_inits(n_restarts)
        else:
            C_init = perturbed_inits(c_center, n_restarts, scale=perturb_scale)
        print(f"initial loss: min {get_loss(C_init).min().item():.3e}")

        if step_through:
            C_final, losses, loss_history = run_grad_descent_sequential(
                get_loss, C_init, n_steps=n_steps, tol=tol,
                c_targ=c_targ.numpy(), deg_tol=deg_tol,
            )
        else:
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

    # the step-through run already built both figures as the restarts landed
    drew_live = do_search and step_through

    # keep the isospectral points
    keep = losses < tol
    C_dual = C_final[keep]
    print(f"\n{C_dual.shape[0]} points with loss < {tol:.0e}")

    if C_dual.shape[0] > 0:
        print("coefficients:")
        print(C_dual)

        # confirm the spectra really do match, per point
        spec_targ = get_spectrum(c_targ)
        spec_dual = get_spectrum(tr.tensor(C_dual))
        spec_err = (spec_dual - spec_targ).abs().max(dim=1).values.numpy()
        print("max spectrum deviation per point:")
        print(spec_err)

        spec_targ = spec_targ[0].numpy()
        spec_dual = spec_dual.numpy()

        targ_pattern = degeneracy_pattern(spec_targ, deg_tol)
        targ_abs_pattern = degeneracy_pattern(np.abs(spec_targ), deg_tol)
        print(
            f"degeneracy per point (target {targ_pattern}, "
            f"|target| {targ_abs_pattern}):"
        )
        for i, spec in enumerate(spec_dual):
            pattern = degeneracy_pattern(spec, deg_tol)
            abs_pattern = degeneracy_pattern(np.abs(spec), deg_tol)
            flag = (
                ""
                if (pattern, abs_pattern) == (targ_pattern, targ_abs_pattern)
                else "   <- differs from target"
            )
            print(
                f"  dual {i}: {pattern}   |.| {abs_pattern}"
                f"   reflection {reflection_residual(spec):.2e}{flag}"
            )

        if not drew_live:
            plot_dual_coefficients(C_dual, c_targ.numpy())
            plot_spectrum_views(spec_targ, spec_dual, tol=deg_tol)

    if not drew_live:
        plot_loss_history(loss_history)
    plt.show()


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()
