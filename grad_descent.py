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
from functools import partial

import numpy as np
import torch as tr
import matplotlib as mp
import matplotlib.pyplot as plt

from ising_finder import get_spectrum, get_loss_factory, n_qb
from duality import print_duality

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

# Darker shade of SERIES_COLORS[0], for the many-line population views where
# the lines overlap and a mid-tone hue washes out
POPULATION_INK = "#1c5aa6"

# The two Hessian strips are one entity measured two ways, so they share a
# hue, and not the one the loss curves above them wear
HESS_COLOR = SERIES_COLORS[1]

# Chart chrome
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# eigenvalues closer than this count as one degenerate level. Well above the
# ~1e-8 spread eigvalsh leaves on a converged dual, well below the level
# spacing of the Ising spectra this searches over.
DEGEN_TOL = 1e-6

# relative cut for calling a Hessian eigenvalue zero, see hessian_spectrum
HESS_RTOL = 1e-6

# relative cut for calling a spectrum-Jacobian singular value zero
JAC_RTOL = 1e-9


def _pow10(x):
    """Mathtext for a tolerance: 1e-06 -> 10^{-6}, 5e-06 -> 5\\times10^{-6}."""
    m, e = f"{x:.0e}".split("e")
    e = int(e)
    return f"10^{{{e}}}" if m == "1" else f"{m}\\times10^{{{e}}}"


def _kept_points(n, loss_tol=None):
    """'3 points with loss < 1e-12': all a plot knows about a search's endpoints.

    A kept endpoint is known to have ended below the loss cut. Whether it is
    an exact dual, or distinct from the target, is not something the plots
    check, so the labels do not claim it.
    """
    noun = "point" if n == 1 else "points"
    cut = "" if loss_tol is None else f" with loss < {loss_tol:.0e}"
    return f"{n} {noun}{cut}"


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


def spectrum_jacobian(c):
    """(2**n_qb, 12) Jacobian of the spectrum with respect to the coefficients.

    The loss Hessian at a zero-residual point is 2 J^T J, so this carries the
    same rank - but its singular values are O(1) with a clean gap, while the
    Hessian's "zero" eigenvalues sit at the residual scale and are only as
    trustworthy as the point is converged. Prefer this wherever the rank is
    the quantity of interest.

    Only valid where the spectrum is non-degenerate: eigenvalues are not
    differentiable where two of them cross, and jacobian_rank refuses there.
    """
    f = lambda u: get_spectrum(u.reshape(1, -1))[0]
    v = tr.as_tensor(np.asarray(c), dtype=tr.get_default_dtype()).detach().reshape(-1)
    return tr.func.jacrev(f)(v).detach().numpy()


def jacobian_rank(c, rtol=JAC_RTOL, gap_tol=DEGEN_TOL):
    """(rank, singular values) of the spectrum Jacobian at one point.

    The fiber of the spectrum map through a regular point has dimension
    12 - rank, which is the number of independent directions the duals of
    that point extend along. Returns rank None where the spectrum is
    degenerate to within gap_tol, since the Jacobian of sorted eigenvalues is
    undefined at a crossing and the rank it reports there is meaningless.
    """
    lam = np.sort(get_spectrum(
        tr.as_tensor(np.asarray(c), dtype=tr.get_default_dtype()).reshape(1, -1)
    )[0].numpy())
    if lam.size > 1 and np.diff(lam).min() <= gap_tol:
        return None, None
    s = np.linalg.svd(spectrum_jacobian(c), compute_uv=False)
    return int((s > rtol * s.max()).sum()), s


def loss_hessian(get_loss, c):
    """(12, 12) Hessian of the loss at one coefficient point.

    c is (12,) or (1, 12). Forward-over-reverse, so this costs a handful of
    spectrum evaluations rather than a finite-difference sweep.
    """
    v = tr.as_tensor(c, dtype=tr.get_default_dtype()).detach().reshape(-1)

    def loss_one(u):
        return get_loss(u.reshape(1, -1))[0]

    return tr.func.hessian(loss_one)(v).detach().numpy()


def hessian_spectrum(H, tol=None, rtol=HESS_RTOL):
    """(eigenvalues ascending, numerical rank, tolerance used) of a Hessian.

    An eigenvalue counts as zero below tol, which defaults to rtol times the
    largest |eigenvalue|. The nullity 12 - rank is then the dimension of the
    flat subspace at that point, i.e. how many directions a curve of duals
    through it has to run along.

    rtol is deliberately far looser than machine precision. For a sum of
    squares the Hessian is 2 J^T J + 2 sum_i r_i grad^2 lambda_i, so the
    directions that J kills still carry curvature proportional to the
    residual r: a restart sitting at loss 1e-12 has "zero" eigenvalues near
    1e-6, and an eps-scale tolerance would call it full rank right up until
    the loss reaches 1e-30. At 1e-6 the gap between the flat and the curved
    directions is several orders wide, and the printed eigenvalues show where
    the cut fell.
    """
    Hs = np.asarray(H)
    Hs = 0.5 * (Hs + Hs.T)
    w = np.linalg.eigvalsh(Hs)
    if tol is None:
        tol = rtol * np.abs(w).max()
    rank = int((np.abs(w) > tol).sum())
    return w, rank, tol


def print_hessian(get_loss, c, label="", tol=None, rtol=HESS_RTOL, loss=None):
    """Print the Hessian eigenvalues and numerical rank at one point.

    Returns (eigenvalues, rank, tolerance), or None where the Hessian came
    back non-finite.
    """
    H = loss_hessian(get_loss, c)
    if not np.isfinite(H).all():
        # eigvalsh is not differentiable where two eigenvalues collide, so a
        # degenerate spectrum can hand back a non-finite Hessian
        print(f"{label}Hessian not finite (degenerate spectrum?)")
        return None

    w, rank, tol_used = hessian_spectrum(H, tol, rtol)
    n_pos = int((w > tol_used).sum())
    n_neg = int((w < -tol_used).sum())
    loss_str = "" if loss is None else f"loss {loss:.3e}  "
    print(
        f"{label}{loss_str}rank {rank}/{w.size}  nullity {w.size - rank}"
        f"  ({n_pos} positive, {n_neg} negative, tol {tol_used:.2e})"
    )
    print("    eigenvalues " + "  ".join(f"{x:+.3e}" for x in w))
    return w, rank, tol_used


def log_hessian(hess_log, step, report):
    """Append (step, rank, eigenvalues, tol) from a print_hessian report.

    The whole eigenvalue list is kept, not a summary of it, since which
    curvature moved is what plot_hessian_log sets against the loss decrease.
    A no-op for a missing log or a non-finite Hessian, so a runner can call it
    unconditionally next to its print.
    """
    if hess_log is None or report is None:
        return
    w, rank, tol = report
    hess_log.append((step, rank, np.asarray(w).copy(), float(tol)))


def run_grad_descent(
    get_loss, c_init, n_steps=2000, lr=1e-2, tol=1e-8, log_every=100,
    hess_every=None, hess_index=0, hess_log=None,
):
    """Descend every restart in parallel.

    c_init is (K, 12). Returns (c_best, loss_best, loss_history), where c_best is
    the (K, 12) best-seen point per restart, loss_best is (K,), and loss_history
    is (n_steps, K) holding every restart's own loss at each step.

    With hess_every set, the Hessian eigenvalues and numerical rank of restart
    hess_index are printed every hess_every steps and again at the point that
    restart finished on, which is how the curvature along one single search
    reads out. Only that one restart is reported, since the point is to follow
    the rank down one descent rather than to sample all of them. Pass a list
    as hess_log to also collect those samples as (step, rank, eigenvalues,
    tol), which is what plot_hessian_log draws.
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

        if hess_every and (step % hess_every == 0 or step == n_steps - 1):
            with tr.no_grad():
                c_here = c[hess_index].clone()
            log_hessian(hess_log, step, print_hessian(
                get_loss, c_here,
                label=f"  step {step:5d} Hessian: ",
                loss=loss[hess_index].item(),
            ))

        opt.step()

    if hess_every:
        print_hessian(
            get_loss, c_best[hess_index],
            label="  best point Hessian: ", loss=loss_best[hess_index].item(),
        )

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
            label=f"point {i}",
        )
    # the individual dual is no longer the unit of interest, the spread is
    return dict(color=SERIES_COLORS[0], linewidth=0.9, alpha=0.35)


def plot_dual_coefficients(C_dual, c_targ, style=None, ax=None, loss_tol=None):
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

    loss_tol, when given, is the cut the points were kept under, and the title
    states it rather than calling them duals.
    """
    c_targ = np.asarray(c_targ).reshape(-1)
    n_duals = C_dual.shape[0]
    kept = _kept_points(n_duals, loss_tol)

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
            label="target",
        )

        for i in range(n_duals):
            ax.plot(x, C_dual[i], zorder=3, **_dual_line_style(i, n_duals))

        ax.set_xticks(x)
        ax.set_xticklabels(COEFF_LABELS)
        ax.set_xlim(-0.4, N_PARAMS - 0.6)
        ax.set_ylabel("Coefficient value")
        ax.set_xlabel("Hamiltonian term")
        ax.set_title(f"Coefficients: target and {kept}", color=INK)
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
            zorder=2, label=_kept_points(n_duals),
        )
        ax.plot(
            x, c_targ, "_", color=INK, markersize=18, markeredgewidth=2.5,
            zorder=3, label="target",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(COEFF_LABELS)
        ax.set_xlim(-0.5, N_PARAMS - 0.5)
        ax.set_ylabel("Coefficient value")
        ax.set_xlabel("Hamiltonian term")
        ax.set_title(
            f"Coefficient spread: target and {kept}", color=INK
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
    spec_targ, spec_duals=None, tol=DEGEN_TOL, ax=None, absolute=False,
    col_notes=None,
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
        col_labels = [f"point {i}" for i in range(n_duals)]
        # a second line under each column, for whatever is being read against
        # the spectrum - the rank read-out, in the searches here
        if col_notes is not None:
            col_labels = [
                lab + "\n" + note for lab, note in zip(col_labels, col_notes)
            ]
    elif n_duals:
        # own generator: cosmetic jitter must not advance the search's stream
        jitter = np.random.default_rng(0).uniform(-0.3, 0.3, size=S.shape)
        ax.plot(
            (1 + jitter).ravel(), S.ravel(), zorder=2,
            **_degeneracy_marker_style(0, n_duals),
        )
        col_labels = [_kept_points(n_duals)]
    else:
        col_labels = []

    ax.set_xticks([0, *x_cols])
    ax.set_xticklabels(["target", *col_labels])
    ax.set_xlim(-1.15, n_cols + 0.6)
    ax.set_ylabel("$|\\lambda_i|$" if absolute else "Eigenvalue $\\lambda_i$")
    ax.set_xlabel(
        f"target lines: values within {tol:g} merged, $\\times m$ = multiplicity"
    )
    # the multiplicity labels already carry the pattern; the title only has to
    # say how many of the target's values collapsed into how few levels. The
    # counts are the target's alone - the columns are drawn, not counted.
    what = "$|\\lambda_i|$" if absolute else "eigenvalues"
    ax.set_title(
        f"Target {what}: {spec_targ.size} values in {levels.size} levels",
        color=INK, fontsize=11,
    )
    _style_axes(ax)
    return ax


def plot_spectrum_views(
    spec_targ, spec_duals=None, tol=DEGEN_TOL, axes=None, col_notes=None
):
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

    plot_spectrum_degeneracy(
        spec_targ, spec_duals, tol=tol, ax=axes[0], col_notes=col_notes
    )
    plot_spectrum_degeneracy(
        spec_targ, spec_duals, tol=tol, ax=axes[1], absolute=True,
        col_notes=col_notes,
    )
    return axes


def plot_dual_ranks(ranks_h, ranks_j, ax=None, label=None, loss_tol=None):
    """Numerical rank at each kept point, of the loss Hessian and of d lambda/dc.

    Both are ranks out of the 12 coefficients, so they share one axis. At a
    zero-residual point the Hessian is 2 J^T J, so its eigenvalues are 2 s^2
    for the singular values s of J - but the two counts use different cuts
    (HESS_RTOL on 2 s^2 is about sqrt(HESS_RTOL) on s, far looser than
    JAC_RTOL), so they can part even at a well converged point. Where they
    do, J has a singular value that is small but not zero, which is what the
    two markers separating vertically shows.

    A None entry in ranks_j is a point whose spectrum is degenerate, where the
    Jacobian of sorted eigenvalues does not exist; those are drawn as a hollow
    marker on the axis floor rather than silently skipped.

    Every label states the cut its rank was counted under, and loss_tol (when
    given) is what the points were kept under - nothing more is assumed about
    them, such as their being distinct duals.
    """
    n = len(ranks_h)
    if ax is None:
        # the legend sits outside the axes on the right, so it needs the width
        _, ax = plt.subplots(
            figsize=(max(8.4, 4.8 + 0.42 * n), 3.2), layout="constrained"
        )

    x = np.arange(n)
    ax.axhline(N_PARAMS, color=BASELINE, linewidth=1, zorder=0)
    # the columns of d lambda / d c each sum to Tr A_k = 0, so its rank is
    # capped one below the number of eigenvalues
    jac_cap = min(N_PARAMS, 2**n_qb - 1)
    if jac_cap < N_PARAMS:
        ax.axhline(
            jac_cap, color=BASELINE, linewidth=1, linestyle="--", zorder=0,
            label=f"max possible rank of $\\partial\\lambda/\\partial c$: "
                  f"$2^n - 1$ = {jac_cap}",
        )

    # the two ranks agree where J has no small singular values, so the
    # Hessian marker is a ring the Jacobian marker sits inside: agreement
    # reads as one symbol, disagreement separates vertically, and neither
    # ever hides the other
    rh = np.array([np.nan if r is None else r for r in ranks_h], float)
    ax.plot(x, rh, color=SERIES_COLORS[0], marker="o", markersize=11,
            markerfacecolor="none", markeredgewidth=1.8, linestyle="none",
            zorder=3,
            label=f"rank of loss Hessian $\\nabla^2 L$:\n"
                  f"count of $|\\mu| > {_pow10(HESS_RTOL)}\\,\\max|\\mu|$")
    known = [i for i, r in enumerate(ranks_j) if r is not None]
    ax.plot([x[i] for i in known], [ranks_j[i] for i in known],
            color=SERIES_COLORS[1], marker="s", markersize=6, zorder=4,
            linestyle="none", markeredgecolor="white", markeredgewidth=0.6,
            label=f"rank of $\\partial\\lambda/\\partial c$:\n"
                  f"count of $\\sigma > {_pow10(JAC_RTOL)}\\,\\sigma_{{max}}$")
    undefined = [i for i, r in enumerate(ranks_j) if r is None]
    if undefined:
        ax.plot([x[i] for i in undefined], [0] * len(undefined), color=MUTED,
                marker="s", markersize=7, markerfacecolor="white", zorder=4,
                linestyle="none",
                label="$\\partial\\lambda/\\partial c$ undefined\n"
                      "(degenerate spectrum), drawn at 0")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n)])
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.8, N_PARAMS + 0.8)
    ax.set_yticks([0, N_PARAMS // 2, N_PARAMS])
    ax.set_xlabel("point index")
    ax.set_ylabel(f"numerical rank (of {N_PARAMS})")
    head = f"Rank at {_kept_points(n, loss_tol)}"
    if label is not None:
        head = f"{label}: {head}"
    ax.set_title(
        head + f"\n{N_PARAMS} $-$ rank($\\partial\\lambda/\\partial c$) = "
        "number of coefficient directions\nthat leave $\\lambda$ unchanged "
        "to first order",
        color=INK, fontsize=9,
    )
    # outside the axes: four entries would sit on top of the markers inside
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _style_axes(ax)
    return ax


def _setup_loss_axes(ax, n_steps, xlabel=True):
    """Chrome shared by the finished-history and step-through loss figures."""
    ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel("Step")
    # sorted eigenvalues on both sides, as get_loss compares them
    ax.set_ylabel("loss $L = \\sum_i (\\lambda_i - \\lambda_i^{target})^2$")
    ax.set_xlim(0, n_steps - 1)
    _style_axes(ax)


def _setup_hessian_axes(rate_ax, eig_ax, n_steps):
    """Chrome for the two strips that sit under a loss panel.

    mu is the Hessian's own eigenvalue, kept apart from the lambda of the
    Hamiltonian spectrum that the loss is built out of.
    """
    # zero marked: under symlog a step that raised the loss is a negative
    # rate, and it has to read as below the line rather than as a small one
    rate_ax.axhline(0, color=BASELINE, linewidth=1, zorder=0)
    rate_ax.set_ylabel("decades of loss\nper step")

    # log: the question is how far each curvature sits from the others and
    # from the zero cut, which spans many decades
    eig_ax.set_yscale("log")
    eig_ax.set_ylabel("$|\\mu|$")
    eig_ax.set_xlabel("Step")

    for a in (rate_ax, eig_ax):
        a.set_xlim(0, n_steps - 1)
        _style_axes(a)


def make_loss_figure(n_steps, with_hessian=False):
    """(loss_ax, hess_axes) - the loss panel, and the strips under it.

    hess_axes is (loss_ax, rate_ax, eig_ax), all sharing one step axis, or
    None where the Hessian is not being followed. The per-step loss decrease
    and the Hessian eigenvalues get a strip each rather than two scales on one
    panel: they are different quantities in different units, so a shared
    y-axis would mean reading two of them off one gridline. The shared step
    axis is what lines a change in the one up with a change in the other.
    """
    if not with_hessian:
        _, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
        _setup_loss_axes(ax, n_steps)
        return ax, None

    _, (ax, rate_ax, eig_ax) = plt.subplots(
        3, 1, figsize=(6.4, 7.8), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.1, 1.6]},
        layout="constrained",
    )
    # the three share one step axis, so only the bottom strip labels it
    _setup_loss_axes(ax, n_steps, xlabel=False)
    _setup_hessian_axes(rate_ax, eig_ax, n_steps)
    return ax, (ax, rate_ax, eig_ax)


def _hessian_color(restart, n_restarts):
    """The hue the Hessian strips wear.

    Where the restarts are individually coloured, the strips take the hue of
    the one they follow, so they read as that curve's own curvature rather
    than as another series. Where the loss curves have collapsed into one
    population in a single hue, the strips take a hue that population does
    not use.
    """
    if restart is not None and n_restarts is not None and n_restarts <= MAX_SERIES:
        return SERIES_COLORS[restart]
    return HESS_COLOR


# |mu| at or below this is drawn on the floor of the log axis
EIG_FLOOR = 1e-20


def _panel_title(ax, text):
    """Short left-aligned heading, the same on every panel of the stack."""
    ax.set_title(text, color=INK, fontsize=10, loc="left", pad=5)


def plot_hessian_log(hess_axes, hess_log, restart=None, n_restarts=None,
                     loss_curve=None):
    """Draw one search's loss decrease against its Hessian spectrum.

    hess_log is the list of (step, rank, eigenvalues, tol) samples a runner
    filled in, and loss_curve is that same restart's (n_steps,) loss history.

    The middle strip is how fast the loss fell: log10(L_t / L_{t+1}) per
    step, averaged over each interval between Hessian samples where they are
    sparser than every step, so each value pairs with one sample below it.

    The bottom strip is the Hessian spectrum, one line per eigenvalue (sorted
    by |mu|, so line j is the j-th stiffest direction). Lines in the restart's
    hue are counted in the rank; the shaded band is everything at or below
    the cut, counted as zero, and lines inside it are grey. Triangles mark
    negative eigenvalues - directions the loss curves down along. The rank is
    printed at the first and last sample; it is the number of hued lines.
    """
    if hess_axes is None or not hess_log:
        return

    _, rate_ax, eig_ax = hess_axes
    color = _hessian_color(restart, n_restarts)
    which = "one search" if restart is None else f"restart {restart}"
    steps = np.array([s for s, _, _, _ in hess_log])
    ranks = [r for _, r, _, _ in hess_log]
    W = np.array([w for _, _, w, _ in hess_log])  # (n_samples, 12)
    tols = np.array([t for _, _, _, t in hess_log])
    dense = bool(np.all(np.diff(steps) <= 1))

    if loss_curve is not None:
        # the mean rate over each interval between samples, drawn at the
        # sample that starts it: the progress made while the Hessian was
        # (roughly) the one sampled there. Sampled every step, it is the
        # per-step rate itself. The raw per-step trace is not drawn: for Adam
        # it swings both ways every step, which is noise, not signal.
        L = np.asarray(loss_curve, dtype=float)
        ends = np.append(steps[1:], L.size - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.log10(L[steps] / L[ends]) / (ends - steps)
        # an interval the runner spent frozen is no steps at all, not a rate
        # of zero; the runners carry a frozen loss forward unchanged
        rate[~np.isfinite(rate) | (L[steps] == L[ends])] = np.nan
        rate_ax.plot(steps, rate, color=color, linewidth=2, marker="o",
                     markersize=4, markeredgecolor="white",
                     markeredgewidth=0.6, zorder=4)

        # log: fast early steps and a slow tail can be decades apart, and both
        # matter. Only a step that raised the loss needs symlog, with its
        # linear band capped four decades under the fastest rate.
        finite = rate[np.isfinite(rate)]
        if finite.size and (finite > 0).all():
            rate_ax.set_yscale("log")
        elif finite.size:
            seen = np.abs(finite[finite != 0])
            lin = max(10 ** np.floor(np.log10(seen.min())), seen.max() * 1e-4)
            rate_ax.set_yscale("symlog", linthresh=lin, linscale=0.5)
            rate_ax.set_ylabel("decades of loss\nper step (symlog)")
        how = "" if dense else ", mean between Hessian samples"
        _panel_title(rate_ax, f"How fast the loss falls ({which}{how})")

    A = np.maximum(np.abs(W), EIG_FLOOR)
    order = np.argsort(-A, axis=1)
    A = np.take_along_axis(A, order, axis=1)
    Wsorted = np.take_along_axis(W, order, axis=1)
    above = A > tols[:, None]

    # the band of "counted as zero", so a line's membership reads off where
    # it sits rather than off a marker style
    lo = min(A.min(), tols.min()) / 5
    eig_ax.fill_between(steps, lo, tols, color=GRID, alpha=0.8, linewidth=0,
                        zorder=0, step=None)
    eig_ax.plot(steps, tols, color=MUTED, linewidth=1, linestyle="--", zorder=1)

    for j in range(A.shape[1]):
        # each line in two pieces, hued while counted and grey once below the
        # cut. A segment takes the state of the sample it ends on, so each
        # piece also claims the sample just before it and the line stays whole
        on = above[:, j] | np.r_[above[1:, j], False]
        off = ~above[:, j] | np.r_[~above[1:, j], False]
        eig_ax.plot(steps, np.where(on, A[:, j], np.nan), color=color,
                    linewidth=1.4, zorder=3)
        eig_ax.plot(steps, np.where(off, A[:, j], np.nan), color=MUTED,
                    linewidth=1.0, alpha=0.7, zorder=2)

    neg = above & (Wsorted < 0)
    if neg.any():
        S = np.broadcast_to(steps[:, None], A.shape)
        eig_ax.plot(S[neg], A[neg], linestyle="none", marker="v", markersize=5,
                    color=color, markeredgecolor="white", markeredgewidth=0.5,
                    zorder=4, label="$\mu < 0$")
        eig_ax.legend(frameon=False, fontsize=8, labelcolor=INK,
                      loc="lower right", handletextpad=0.3)

    # direct labels in place of a legend: the band, and the rank at each end
    eig_ax.annotate(
        f"counted as 0: $|\mu| \leq {_pow10(HESS_RTOL)}\,\max|\mu|$",
        xy=(0.01, 0.03), xycoords="axes fraction", fontsize=8, color=MUTED,
    )
    top = A.max() * 20
    for i, ha, dx in ((0, "left", 2), (len(steps) - 1, "right", -2)):
        eig_ax.annotate(f"rank {ranks[i]}", xy=(steps[i], A[i, 0]),
                        xytext=(dx, 6), textcoords="offset points", ha=ha,
                        fontsize=8.5, color=INK, zorder=5)
    eig_ax.set_ylim(lo, top)
    _panel_title(eig_ax, f"Hessian eigenvalues ({which})")


def _loss_line_style(k, n_restarts, focus=None):
    """Per-restart line style: categorical while identity is legible, else one hue.

    Returns kwargs for plotting restart k of n_restarts. With focus set, every
    restart but that one keeps its hue but recedes, since the strips under
    the loss panel follow only the focused one.
    """
    if n_restarts <= MAX_SERIES:
        style = dict(color=SERIES_COLORS[k], linewidth=2, label=f"restart {k}")
        if focus is not None and k != focus:
            style.update(linewidth=1.1, alpha=0.35)
        return style
    # the individual restart is no longer the unit of interest, the spread is,
    # but the population still has to read as lines rather than a haze: a
    # darker shade of the first slot, mostly opaque
    return dict(color=POPULATION_INK, linewidth=1.1, alpha=0.7)


def _loss_legend(ax, n_restarts, on_figure=False):
    """Legend for the loss panel, outside the axes on its right.

    With on_figure the space is reserved on the figure rather than taken out
    of the loss axes. That is what the stacked layout needs: a legend charged
    to one axes shrinks only that axes, and the loss panel would stop lining
    up with the Hessian strips it shares a step axis with.
    """
    if n_restarts > MAX_SERIES:
        return

    style = dict(frameon=False, fontsize=8, labelcolor=INK)
    if not on_figure:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), **style)
        return

    fig = ax.figure
    # the sequential run relegends after every restart, and figure legends
    # accumulate where an axes legend would have been replaced
    for old in list(fig.legends):
        old.remove()
    # a column down the right of the whole figure, so every panel narrows by
    # the same amount and the shared step axis stays aligned
    fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper",
               **style)


def plot_loss_history(loss_history, hess_log=None, hess_restart=None, ax=None):
    """One line per restart, showing where each descent ended up.

    loss_history is (n_steps, K). Up to 8 restarts each get their own
    categorical hue; past that the restarts stop being individually
    identifiable and become a single translucent population.

    Pass hess_log, the samples a runner collected along one restart, to get
    the loss-decrease and Hessian-spectrum strips under the loss panel;
    hess_restart names which restart they belong to, and so whose loss curve
    the decrease is taken from.
    """
    n_steps, n_restarts = loss_history.shape

    hess_axes = None
    if ax is None:
        ax, hess_axes = make_loss_figure(n_steps, with_hessian=bool(hess_log))
    else:
        _setup_loss_axes(ax, n_steps)

    k_h = 0 if hess_restart is None else hess_restart
    # the strips follow one restart, so the rest recede behind it
    focus = k_h if hess_axes is not None else None
    for k in range(n_restarts):
        ax.plot(loss_history[:, k], **_loss_line_style(k, n_restarts, focus))
    _loss_legend(ax, n_restarts, on_figure=hess_axes is not None)

    ax.set_title(f"Loss per restart over {n_restarts} restarts", color=INK)
    plot_hessian_log(hess_axes, hess_log, hess_restart, n_restarts,
                     loss_curve=loss_history[:, k_h])
    return ax


def run_grad_descent_sequential(
    get_loss, c_init, n_steps=2000, lr=1e-2, tol=1e-8, log_every=100,
    c_targ=None, ax=None, deg_tol=DEGEN_TOL, run_restart=None,
    hess_every=None, hess_restart=0,
):
    """Descend one restart at a time, adding each finished curve to a live figure.

    Each restart runs to completion through run_grad_descent on a batch of one,
    then its loss curve is drawn and the run blocks for a keypress or click
    before the next restart starts. Returns the same three values with the same
    shapes as run_grad_descent, so callers and the saved artifact are unaffected.

    run_restart swaps in another optimizer for the per-restart descent: any
    callable with run_grad_descent's signature and return values, which is how
    newton_raphson.py reuses this whole live-figure loop. It defaults to
    run_grad_descent at the given lr.

    With hess_every set, restart hess_restart - one single search - prints its
    Hessian eigenvalues and rank every hess_every steps, and those samples are
    drawn as the loss-decrease and Hessian-spectrum strips under the loss
    panel.

    Pass c_targ to also get two more live figures: the duals' coefficients, and
    the degeneracy of their spectra against the target's, in both the signed
    and the |lambda| view. Both grow whenever a restart lands below tol. The
    degeneracy figure opens on the target alone, before the first restart,
    since its level structure is what the duals are being read against;
    deg_tol sets how close eigenvalues must sit to count as one level.
    """
    n_restarts = c_init.shape[0]
    if run_restart is None:
        run_restart = partial(run_grad_descent, lr=lr)

    # no event loop under a headless backend, so blocking there would hang
    interactive = not mp.get_backend().lower().startswith("agg")
    if interactive:
        plt.ion()

    hess_axes = None
    if ax is None:
        ax, hess_axes = make_loss_figure(n_steps, with_hessian=bool(hess_every))
    else:
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
        # only the followed restart collects samples, and only it has strips
        hess_log = [] if (hess_every and k == hess_restart) else None
        c_k, loss_k, history_k = run_restart(
            get_loss, c_init[k : k + 1],
            n_steps=n_steps, tol=tol, log_every=log_every,
            hess_every=hess_every if k == hess_restart else None,
            hess_log=hess_log,
        )
        plot_hessian_log(hess_axes, hess_log, restart=k, n_restarts=n_restarts,
                         loss_curve=history_k[:, 0])

        c_best[k] = c_k[0]
        loss_best[k] = loss_k[0]
        loss_history[:, k] = history_k[:, 0]

        # the strips follow one restart, so the rest recede behind it
        focus = hess_restart if hess_axes is not None else None
        ax.plot(loss_history[:, k], **_loss_line_style(k, n_restarts, focus))
        _loss_legend(ax, n_restarts, on_figure=hess_axes is not None)
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
                plot_dual_coefficients(
                    np.array(duals), c_targ, ax=coeff_ax, loss_tol=tol
                )

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
    step_through = False

    n_restarts = 12
    n_steps = 2000
    tol = 1e-12
    # print the Hessian eigenvalues and rank this often along one restart's
    # descent; None turns the read-out off
    # dense enough that the loss-decrease strip has a Hessian sample to be
    # read against every few dozen steps, not four times a run
    hess_every = 50
    # which restart is the one followed, when stepping through them in turn
    hess_restart = 0
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
    #print_degeneracy(get_spectrum(c_targ)[0].numpy(), label="target", tol=deg_tol)
    #print_symmetry(get_spectrum(c_targ)[0].numpy(), label="target", tol=deg_tol)

    # the step-through run fills and draws its own; this one is for the batch
    # path, and stays empty on a reload since it is not part of the artifact
    hess_log = []

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
                hess_every=hess_every, hess_restart=hess_restart,
            )
        else:
            C_final, losses, loss_history = run_grad_descent(
                get_loss, C_init, n_steps=n_steps, tol=tol,
                hess_every=hess_every, hess_index=hess_restart,
                hess_log=hess_log,
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

        # zero loss is only isospectrality; this rules out the trivial maps
        print_duality(C_dual, c_targ.numpy())

        if not drew_live:
            plot_dual_coefficients(C_dual, c_targ.numpy(), loss_tol=tol)
            plot_spectrum_views(spec_targ, spec_dual, tol=deg_tol)

    if not drew_live:
        plot_loss_history(loss_history, hess_log, hess_restart)
    plt.show()


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()
