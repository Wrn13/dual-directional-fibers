"""Gradient-descent search for dual tensor product structures.

Random-restart batched descent over the 12 Ising coefficients for points
isospectral with a target Hamiltonian, plus the project's plotting helpers.
Coefficient layout: [XX, YY, ZZ, X, Y, Z, X_1, Y_1, Z_1, X_N, Y_N, Z_N].
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

N_PARAMS = 12
COEFF_LABELS = [
    "$XX$", "$YY$", "$ZZ$", "$X$", "$Y$", "$Z$",
    "$X_1$", "$Y_1$", "$Z_1$", "$X_N$", "$Y_N$", "$Z_N$",
]

# categorical series slots, assigned in order and never cycled
SERIES_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MAX_SERIES = len(SERIES_COLORS)
POPULATION_INK = "#1c5aa6"  # many-line views, where a mid-tone washes out
HESS_COLOR = SERIES_COLORS[1]
RANK_RAMP = ["#e4edf8", "#c3d7ee", "#93b8e2", "#6f9ed2", "#2a78d6", "#1c5aa6",
             "#123a6b"]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DEGEN_TOL = 1e-6  # eigenvalues closer than this are one degenerate level
HESS_RTOL = 1e-6  # relative zero cut for Hessian eigenvalues, see hessian_spectrum
JAC_RTOL = 1e-9   # relative zero cut for spectrum-Jacobian singular values
EIG_FLOOR = 1e-20  # |mu| at or below this is drawn on the log axis floor


def _pow10(x):
    """Mathtext for a tolerance: 1e-06 -> 10^{-6}, 5e-06 -> 5\\times10^{-6}."""
    m, e = f"{x:.0e}".split("e")
    e = int(e)
    return f"10^{{{e}}}" if m == "1" else f"{m}\\times10^{{{e}}}"


def _kept_points(n, loss_tol=None):
    """'3 points with loss < 1e-12' - no claim that they are distinct duals."""
    noun = "point" if n == 1 else "points"
    cut = "" if loss_tol is None else f" with loss < {loss_tol:.0e}"
    return f"{n} {noun}{cut}"


def _prefixed(label, head):
    return head if label is None else f"{label}: {head}"


def _spectrum(c):
    """(2**n_qb,) numpy spectrum at one coefficient point."""
    c = tr.as_tensor(np.asarray(c), dtype=tr.get_default_dtype()).reshape(1, -1)
    return get_spectrum(c)[0].numpy()


def random_inits(n_restarts, scale=1.0):
    """(K, 12) batch of random restarts over the full coefficient space."""
    return tr.tensor(rng.normal(scale=scale, size=(n_restarts, N_PARAMS)))


def perturbed_inits(c_center, n_restarts, scale=0.1):
    """(K, 12) batch of c_center plus isotropic Gaussian noise of the given scale."""
    c_center = tr.as_tensor(c_center, dtype=tr.get_default_dtype()).reshape(1, -1)
    if c_center.shape[1] != N_PARAMS:
        raise ValueError(
            f"c_center has {c_center.shape[1]} coefficients, expected {N_PARAMS}"
        )
    return c_center + tr.tensor(rng.normal(scale=scale, size=(n_restarts, N_PARAMS)))


def spectrum_jacobian(c):
    """(2**n_qb, 12) Jacobian of the spectrum with respect to the coefficients.

    Same rank as the loss Hessian at a zero-residual point, but with a clean
    singular-value gap. Only valid where the spectrum is non-degenerate.
    """
    f = lambda u: get_spectrum(u.reshape(1, -1))[0]
    v = tr.as_tensor(np.asarray(c), dtype=tr.get_default_dtype()).detach().reshape(-1)
    return tr.func.jacrev(f)(v).detach().numpy()


def jacobian_rank(c, rtol=JAC_RTOL, gap_tol=DEGEN_TOL):
    """(rank, singular values) of the spectrum Jacobian; (None, None) if degenerate."""
    lam = np.sort(_spectrum(c))
    if lam.size > 1 and np.diff(lam).min() <= gap_tol:
        return None, None
    s = np.linalg.svd(spectrum_jacobian(c), compute_uv=False)
    return int((s > rtol * s.max()).sum()), s


def jacobian_null_space(c, rtol=JAC_RTOL, gap_tol=DEGEN_TOL):
    """(12, 12 - rank) orthonormal null-space basis of the Jacobian, None if degenerate."""
    rank, _ = jacobian_rank(c, rtol=rtol, gap_tol=gap_tol)
    if rank is None:
        return None
    return np.linalg.svd(spectrum_jacobian(c))[2][rank:].T


def null_operator_share(N):
    """(12,) diagonal of the projector N N^T: each operator's share of the null space."""
    return (N**2).sum(axis=1)


def null_space_overlap(N1, N2):
    """||N1^T N2||_F^2 / min(d1, d2): 1 if one space contains the other, 0 if
    orthogonal, nan if either is empty."""
    d = min(N1.shape[1], N2.shape[1])
    if d == 0:
        return np.nan
    return float(np.linalg.norm(N1.T @ N2) ** 2 / d)


def loss_hessian(get_loss, c):
    """(12, 12) Hessian of the loss at one point c, (12,) or (1, 12)."""
    v = tr.as_tensor(c, dtype=tr.get_default_dtype()).detach().reshape(-1)
    return tr.func.hessian(lambda u: get_loss(u.reshape(1, -1))[0])(v).detach().numpy()


def hessian_spectrum(H, tol=None, rtol=HESS_RTOL):
    """(ascending eigenvalues, numerical rank, tolerance used) of a Hessian.

    rtol is far looser than machine precision on purpose: directions J kills
    still carry curvature ~ the residual, so at loss 1e-12 the "zero"
    eigenvalues sit near 1e-6.
    """
    Hs = np.asarray(H)
    w = np.linalg.eigvalsh(0.5 * (Hs + Hs.T))
    if tol is None:
        tol = rtol * np.abs(w).max()
    return w, int((np.abs(w) > tol).sum()), tol


def print_hessian(get_loss, c, label="", tol=None, rtol=HESS_RTOL, loss=None):
    """Print Hessian eigenvalues and rank at c; return hessian_spectrum's tuple,
    or None if the Hessian is non-finite (degenerate spectrum)."""
    H = loss_hessian(get_loss, c)
    if not np.isfinite(H).all():
        print(f"{label}Hessian not finite (degenerate spectrum?)")
        return None

    w, rank, tol_used = hessian_spectrum(H, tol, rtol)
    loss_str = "" if loss is None else f"loss {loss:.3e}  "
    print(
        f"{label}{loss_str}rank {rank}/{w.size}  nullity {w.size - rank}"
        f"  ({int((w > tol_used).sum())} positive,"
        f" {int((w < -tol_used).sum())} negative, tol {tol_used:.2e})"
    )
    print("    eigenvalues " + "  ".join(f"{x:+.3e}" for x in w))
    return w, rank, tol_used


def hessian_rank(get_loss, c, rtol=HESS_RTOL):
    """Numerical rank of the loss Hessian at c, nan where not finite."""
    H = loss_hessian(get_loss, c)
    if not np.isfinite(H).all():
        return np.nan
    return hessian_spectrum(H, rtol=rtol)[1]


def log_ranks(rank_log, step, get_loss, c, active):
    """Append (step, (K,) Hessian ranks), nan for inactive restarts. No-op if rank_log is None."""
    if rank_log is None:
        return
    ranks = np.full(c.shape[0], np.nan)
    for k in np.flatnonzero(active):
        ranks[k] = hessian_rank(get_loss, c[k])
    rank_log.append((step, ranks))


def log_hessian(hess_log, step, report):
    """Append (step, rank, eigenvalues, tol) from a print_hessian report.
    No-op if hess_log or report is None."""
    if hess_log is None or report is None:
        return
    w, rank, tol = report
    hess_log.append((step, rank, np.asarray(w).copy(), float(tol)))


def run_grad_descent(
    get_loss, c_init, n_steps=2000, lr=1e-2, tol=1e-8, log_every=100,
    hess_every=None, hess_index=0, hess_log=None,
):
    """Adam on every restart in parallel.

    c_init is (K, 12). Returns (c_best (K, 12), loss_best (K,), loss_history
    (n_steps, K)). With hess_every, restart hess_index's Hessian is printed
    (and logged to hess_log) every hess_every steps and at its best point.
    """
    c = c_init.clone().detach().requires_grad_(True)
    opt = tr.optim.Adam([c], lr=lr)

    c_best = c_init.clone().detach()
    loss_best = tr.full((c_init.shape[0],), np.inf)
    loss_history = np.empty((n_steps, c_init.shape[0]))

    for step in range(n_steps):
        opt.zero_grad()
        loss = get_loss(c)
        loss.sum().backward()

        with tr.no_grad():
            improved = loss < loss_best
            loss_best[improved] = loss[improved]
            c_best[improved] = c[improved]
            loss_history[step] = loss.numpy()

        last = step == n_steps - 1
        if step % log_every == 0 or last:
            print(
                f"step {step:5d}: min {loss_best.min().item():.3e}"
                f"  median {loss.median().item():.3e}"
                f"  below tol {(loss_best < tol).sum().item()}/{len(loss_best)}"
            )

        if hess_every and (step % hess_every == 0 or last):
            with tr.no_grad():
                c_here = c[hess_index].clone()
            log_hessian(hess_log, step, print_hessian(
                get_loss, c_here, label=f"  step {step:5d} Hessian: ",
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


def _legend_right(ax, fontsize=8):
    """Legend outside the axes, top right."""
    ax.legend(frameon=False, fontsize=fontsize, labelcolor=INK,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _hide_spines(ax):
    for side in ax.spines.values():
        side.set_visible(False)


def _dual_line_style(i, n_duals):
    """Line kwargs for dual i: own hue and marker up to MAX_SERIES, else one population."""
    if n_duals <= MAX_SERIES:
        return dict(
            color=SERIES_COLORS[i], marker=SERIES_MARKERS[i],
            markersize=7, linewidth=1.6,
            markeredgecolor="white", markeredgewidth=0.8,
            label=f"point {i}",
        )
    return dict(color=SERIES_COLORS[0], linewidth=0.9, alpha=0.35)


def plot_dual_coefficients(C_dual, c_targ, style=None, ax=None, loss_tol=None):
    """Coefficients of every dual against the target.

    C_dual is (n_duals, 12). style is "profile" (default, one line per dual)
    or "distribution" (jittered strip per operator).
    """
    c_targ = np.asarray(c_targ).reshape(-1)
    n_duals = C_dual.shape[0]
    kept = _kept_points(n_duals, loss_tol)
    style = style or "profile"
    if style not in ("profile", "distribution"):
        raise ValueError(
            f"unknown style {style!r}, expected 'profile' or 'distribution'"
        )

    if ax is None:
        width = 8.6 if style == "profile" and n_duals <= MAX_SERIES else 6.5
        _, ax = plt.subplots(figsize=(width, 3.6), layout="constrained")

    x = np.arange(N_PARAMS)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=0)

    if style == "profile":
        ax.plot(x, c_targ, "--", color=INK, linewidth=2, zorder=2, label="target")
        for i in range(n_duals):
            ax.plot(x, C_dual[i], zorder=3, **_dual_line_style(i, n_duals))
        ax.set_xlim(-0.4, N_PARAMS - 0.6)
        ax.set_title(f"Coefficients: target and {kept}", color=INK)
    else:
        # own generator: cosmetic jitter must not advance the search's stream
        jitter = np.random.default_rng(0).uniform(-0.28, 0.28, size=C_dual.shape)
        ax.scatter(
            x + jitter, C_dual, s=14, color=SERIES_COLORS[0], alpha=0.35,
            linewidths=0, zorder=2, label=_kept_points(n_duals),
        )
        ax.plot(x, c_targ, "_", color=INK, markersize=18, markeredgewidth=2.5,
                zorder=3, label="target")
        ax.set_xlim(-0.5, N_PARAMS - 0.5)
        ax.set_title(f"Coefficient spread: target and {kept}", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(COEFF_LABELS)
    ax.set_ylabel("Coefficient value")
    ax.set_xlabel("Hamiltonian term")
    if style == "distribution":
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")
    elif n_duals <= MAX_SERIES:
        _legend_right(ax)
    _style_axes(ax)
    return ax


def spectrum_degeneracy(spectrum, tol=DEGEN_TOL):
    """(level energies, multiplicities) grouping sorted eigenvalues whose
    neighbouring gap is within tol (so a level can chain wider than tol)."""
    lam = np.sort(np.asarray(spectrum).reshape(-1))
    groups = np.split(lam, np.flatnonzero(np.diff(lam) > tol) + 1)
    return np.array([g.mean() for g in groups]), np.array([g.size for g in groups])


def degeneracy_pattern(spectrum, tol=DEGEN_TOL):
    """Multiplicities as a compact string, e.g. "1-2-2-2-1", low level first."""
    return "-".join(str(m) for m in spectrum_degeneracy(spectrum, tol)[1])


def print_degeneracy(spectrum, label="target", tol=DEGEN_TOL, absolute=False):
    """Print the level table (energy, multiplicity, width) of a spectrum, or of |spectrum|."""
    lam = np.asarray(spectrum).reshape(-1)
    lam = np.sort(np.abs(lam) if absolute else lam)
    levels, mult = spectrum_degeneracy(lam, tol)
    what = "|spectrum|" if absolute else "spectrum"
    print(
        f"{label} {what}: {lam.size} states in {levels.size} levels"
        f"  [{degeneracy_pattern(lam, tol)}]"
    )
    start = 0
    for lv, m in zip(levels, mult):
        width = "" if m == 1 else f"   (width {np.ptp(lam[start : start + m]):.2e})"
        start += m
        print(f"  {lv:+.9f}  x{m}{width}")


def reflection_residual(spectrum):
    """Largest |lambda_i + lambda_{n-1-i}| of the sorted spectrum; 0 means symmetric
    under lambda -> -lambda."""
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
    """(x, sorted eigenvalues) with each degenerate multiplet fanned out around x0."""
    lam = np.sort(np.asarray(spectrum).reshape(-1))
    x = np.empty(lam.size)
    for g in np.split(np.arange(lam.size), np.flatnonzero(np.diff(lam) > tol) + 1):
        x[g] = x0 if g.size == 1 else x0 + np.linspace(-width, width, g.size)
    return x, lam


def _degeneracy_marker_style(i, n_duals):
    """Marker kwargs for dual i, matching its hue elsewhere."""
    if n_duals <= MAX_SERIES:
        return dict(
            color=SERIES_COLORS[i], marker=SERIES_MARKERS[i], markersize=6,
            linestyle="none", markeredgecolor="white", markeredgewidth=0.6,
        )
    return dict(
        color=SERIES_COLORS[0], marker="o", markersize=4,
        linestyle="none", alpha=0.35, markeredgewidth=0,
    )


def _degeneracy_width(n_cols):
    return max(4.6, 3.6 + 0.7 * n_cols)


def plot_spectrum_degeneracy(
    spec_targ, spec_duals=None, tol=DEGEN_TOL, ax=None, absolute=False,
    col_notes=None,
):
    """Target levels (lines, x multiplicity) beside each dual's eigenvalues.

    spec_targ is (2**n_qb,); spec_duals is (n_duals, 2**n_qb) or None. Past
    MAX_SERIES duals they share one jittered column. absolute draws |lambda|.
    col_notes adds a second line under each dual's column label.
    """
    spec_targ = np.asarray(spec_targ).reshape(-1)
    S = (np.empty((0, spec_targ.size)) if spec_duals is None
         else np.asarray(spec_duals).reshape(-1, spec_targ.size))
    if absolute:
        spec_targ, S = np.abs(spec_targ), np.abs(S)

    levels, mult = spectrum_degeneracy(spec_targ, tol)
    n_duals = S.shape[0]
    n_cols = n_duals if n_duals <= MAX_SERIES else 1

    if ax is None:
        _, ax = plt.subplots(figsize=(_degeneracy_width(n_cols), 3.8),
                             layout="constrained")

    half = 0.42
    for lv, m in zip(levels, mult):
        ax.plot([-half, half], [lv, lv], color=INK, linewidth=2, zorder=3)
        if m > 1:
            ax.annotate(
                f"$\\times${m}", xy=(-half - 0.06, lv), ha="right", va="center",
                fontsize=9, color=INK, zorder=4,
            )

    if n_duals and n_duals <= MAX_SERIES:
        for i in range(n_duals):
            x, lam = _fanned_levels(S[i], 1 + i, tol)
            ax.plot(x, lam, zorder=2, **_degeneracy_marker_style(i, n_duals))
        col_labels = [f"point {i}" for i in range(n_duals)]
        if col_notes is not None:
            col_labels = [lab + "\n" + note for lab, note in zip(col_labels, col_notes)]
    elif n_duals:
        # own generator: cosmetic jitter must not advance the search's stream
        jitter = np.random.default_rng(0).uniform(-0.3, 0.3, size=S.shape)
        ax.plot((1 + jitter).ravel(), S.ravel(), zorder=2,
                **_degeneracy_marker_style(0, n_duals))
        col_labels = [_kept_points(n_duals)]
    else:
        col_labels = []

    ax.set_xticks([0, *np.arange(1, n_cols + 1)])
    ax.set_xticklabels(["target", *col_labels])
    ax.set_xlim(-1.15, n_cols + 0.6)
    ax.set_ylabel("$|\\lambda_i|$" if absolute else "Eigenvalue $\\lambda_i$")
    ax.set_xlabel(
        f"target lines: values within {tol:g} merged, $\\times m$ = multiplicity"
    )
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
    """Signed and |lambda| degeneracy views side by side; returns the two axes.

    A level only in the |lambda| panel is a +/- pair, not a true degeneracy.
    """
    if axes is None:
        n_duals = 0 if spec_duals is None else np.asarray(spec_duals).reshape(
            -1, np.asarray(spec_targ).size
        ).shape[0]
        n_cols = n_duals if n_duals <= MAX_SERIES else 1
        _, axes = plt.subplots(
            1, 2, figsize=(2 * _degeneracy_width(n_cols), 3.8), layout="constrained"
        )
    for ax, absolute in zip(axes, (False, True)):
        plot_spectrum_degeneracy(spec_targ, spec_duals, tol=tol, ax=ax,
                                 absolute=absolute, col_notes=col_notes)
    return axes


def plot_dual_ranks(ranks_h, ranks_j, ax=None, label=None, loss_tol=None):
    """Hessian rank (ring) and d lambda/dc rank (square) at each kept point.

    The two use different cuts (HESS_RTOL on 2 s^2 is ~sqrt(HESS_RTOL) on s),
    so they can part even at a converged point. None in ranks_j (degenerate
    spectrum) is drawn hollow at 0.
    """
    n = len(ranks_h)
    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(8.4, 4.8 + 0.42 * n), 3.2), layout="constrained"
        )

    x = np.arange(n)
    ax.axhline(N_PARAMS, color=BASELINE, linewidth=1, zorder=0)
    # each column of d lambda/dc sums to Tr A_k = 0, capping the rank
    jac_cap = min(N_PARAMS, 2**n_qb - 1)
    if jac_cap < N_PARAMS:
        ax.axhline(
            jac_cap, color=BASELINE, linewidth=1, linestyle="--", zorder=0,
            label=f"max possible rank of $\\partial\\lambda/\\partial c$: "
                  f"$2^n - 1$ = {jac_cap}",
        )

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
    ax.set_title(
        _prefixed(label, f"Rank at {_kept_points(n, loss_tol)}")
        + f"\n{N_PARAMS} $-$ rank($\\partial\\lambda/\\partial c$) = "
        "number of coefficient directions\nthat leave $\\lambda$ unchanged "
        "to first order",
        color=INK, fontsize=9,
    )
    _legend_right(ax, fontsize=7.5)
    _style_axes(ax)
    return ax


def _setup_loss_axes(ax, n_steps, xlabel=True):
    """Log-scale loss axes."""
    ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel("Step")
    ax.set_ylabel("loss $L = \\sum_i (\\lambda_i - \\lambda_i^{target})^2$")
    ax.set_xlim(0, n_steps - 1)
    _style_axes(ax)


def make_loss_figure(n_steps, with_hessian=False):
    """(loss_ax, hess_axes): hess_axes is (loss_ax, rate_ax, eig_ax) sharing the
    step axis, or None without the Hessian strips."""
    if not with_hessian:
        _, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
        _setup_loss_axes(ax, n_steps)
        return ax, None

    _, (ax, rate_ax, eig_ax) = plt.subplots(
        3, 1, figsize=(6.4, 7.8), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.1, 1.6]},
        layout="constrained",
    )
    _setup_loss_axes(ax, n_steps, xlabel=False)
    rate_ax.axhline(0, color=BASELINE, linewidth=1, zorder=0)
    rate_ax.set_ylabel("decades of loss\nper step")
    eig_ax.set_yscale("log")
    eig_ax.set_ylabel("$|\\mu|$")  # mu: Hessian eigenvalue, not the spectrum's lambda
    eig_ax.set_xlabel("Step")
    for a in (rate_ax, eig_ax):
        a.set_xlim(0, n_steps - 1)
        _style_axes(a)
    return ax, (ax, rate_ax, eig_ax)


def _loss_axes(ax, n_steps, with_hessian):
    """(ax, hess_axes), making the loss figure only when no ax is given."""
    if ax is None:
        return make_loss_figure(n_steps, with_hessian)
    _setup_loss_axes(ax, n_steps)
    return ax, None


def _panel_title(ax, text):
    ax.set_title(text, color=INK, fontsize=10, loc="left", pad=5)


def plot_hessian_log(hess_axes, hess_log, restart=None, n_restarts=None,
                     loss_curve=None):
    """Draw one search's loss-decrease rate and Hessian spectrum into the strips.

    hess_log holds (step, rank, eigenvalues, tol) samples; loss_curve is that
    restart's (n_steps,) loss. The rate is log10(L_t / L_{t+1}) averaged over
    each interval between samples. Eigenvalue lines are hued while above the
    zero cut and grey inside the shaded band; triangles mark negative ones.
    """
    if hess_axes is None or not hess_log:
        return

    _, rate_ax, eig_ax = hess_axes
    # the followed restart's own hue while restarts are individually coloured
    color = (SERIES_COLORS[restart] if restart is not None and n_restarts is not None
             and n_restarts <= MAX_SERIES else HESS_COLOR)
    which = "one search" if restart is None else f"restart {restart}"
    steps = np.array([s for s, _, _, _ in hess_log])
    ranks = [r for _, r, _, _ in hess_log]
    W = np.array([w for _, _, w, _ in hess_log])  # (n_samples, 12)
    tols = np.array([t for _, _, _, t in hess_log])

    if loss_curve is not None:
        L = np.asarray(loss_curve, dtype=float)
        ends = np.append(steps[1:], L.size - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.log10(L[steps] / L[ends]) / (ends - steps)
        # a frozen interval (loss carried forward unchanged) is no rate, not zero
        rate[~np.isfinite(rate) | (L[steps] == L[ends])] = np.nan
        rate_ax.plot(steps, rate, color=color, linewidth=2, marker="o",
                     markersize=4, markeredgecolor="white",
                     markeredgewidth=0.6, zorder=4)

        # symlog only when some step raised the loss
        finite = rate[np.isfinite(rate)]
        if finite.size and (finite > 0).all():
            rate_ax.set_yscale("log")
        elif finite.size:
            seen = np.abs(finite[finite != 0])
            lin = max(10 ** np.floor(np.log10(seen.min())), seen.max() * 1e-4)
            rate_ax.set_yscale("symlog", linthresh=lin, linscale=0.5)
            rate_ax.set_ylabel("decades of loss\nper step (symlog)")
        how = "" if np.all(np.diff(steps) <= 1) else ", mean between Hessian samples"
        _panel_title(rate_ax, f"How fast the loss falls ({which}{how})")

    A = np.maximum(np.abs(W), EIG_FLOOR)
    order = np.argsort(-A, axis=1)
    A = np.take_along_axis(A, order, axis=1)
    Wsorted = np.take_along_axis(W, order, axis=1)
    above = A > tols[:, None]

    lo = min(A.min(), tols.min()) / 5
    eig_ax.fill_between(steps, lo, tols, color=GRID, alpha=0.8, linewidth=0,
                        zorder=0, step=None)
    eig_ax.plot(steps, tols, color=MUTED, linewidth=1, linestyle="--", zorder=1)

    for j in range(A.shape[1]):
        # a segment takes the state of the sample it ends on, so each piece
        # also claims the sample before it and the line stays unbroken
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
                    zorder=4, label=r"$\mu < 0$")
        eig_ax.legend(frameon=False, fontsize=8, labelcolor=INK,
                      loc="lower right", handletextpad=0.3)

    eig_ax.annotate(
        rf"counted as 0: $|\mu| \leq {_pow10(HESS_RTOL)}\,\max|\mu|$",
        xy=(0.01, 0.03), xycoords="axes fraction", fontsize=8, color=MUTED,
    )
    for i, ha, dx in ((0, "left", 2), (len(steps) - 1, "right", -2)):
        eig_ax.annotate(f"rank {ranks[i]}", xy=(steps[i], A[i, 0]),
                        xytext=(dx, 6), textcoords="offset points", ha=ha,
                        fontsize=8.5, color=INK, zorder=5)
    eig_ax.set_ylim(lo, A.max() * 20)
    _panel_title(eig_ax, f"Hessian eigenvalues ({which})")


def _loss_line_style(k, n_restarts, focus=None):
    """Line kwargs for restart k; with focus set, the other restarts recede."""
    if n_restarts <= MAX_SERIES:
        style = dict(color=SERIES_COLORS[k], linewidth=2, label=f"restart {k}")
        if focus is not None and k != focus:
            style.update(linewidth=1.1, alpha=0.35)
        return style
    return dict(color=POPULATION_INK, linewidth=1.1, alpha=0.7)


def _loss_legend(ax, n_restarts, on_figure=False):
    """Loss legend right of the axes, or (on_figure) right of the whole figure so
    the stacked panels stay aligned."""
    if n_restarts > MAX_SERIES:
        return
    if not on_figure:
        _legend_right(ax)
        return
    fig = ax.figure
    # figure legends accumulate where an axes legend would be replaced
    for old in list(fig.legends):
        old.remove()
    fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper",
               frameon=False, fontsize=8, labelcolor=INK)


def plot_rank_map(rank_log, final_ranks=None, final_losses=None, tol=None,
                  label=None, ax=None):
    """Hessian rank of every restart along its search, one row per restart.

    rank_log is the list of (step, (K,) ranks), nan where a restart had
    stopped (drawn hatched). final_ranks/final_losses (K,) label each row's
    end; losses not below tol are muted.
    """
    steps = np.array([s for s, _ in rank_log])
    R = np.array([r for _, r in rank_log]).T  # (K, n_samples)
    n_restarts = R.shape[0]
    stride = int(np.min(np.diff(steps))) if steps.size > 1 else 1
    edges = np.append(steps, steps[-1] + stride)

    known = R[np.isfinite(R)]
    if final_ranks is not None:
        known = np.append(known, np.asarray(final_ranks, float))
    known = known[np.isfinite(known)]
    lo, hi = (int(known.min()), int(known.max())) if known.size else (0, N_PARAMS)
    n_levels = hi - lo + 1
    ramp = mp.colors.LinearSegmentedColormap.from_list("rank", RANK_RAMP)
    cmap = mp.colors.ListedColormap(ramp(np.linspace(0.1, 1.0, n_levels)))
    norm = mp.colors.BoundaryNorm(np.arange(lo - 0.5, hi + 1.5), n_levels)

    if ax is None:
        _, ax = plt.subplots(
            figsize=(8.4, 1.6 + 0.34 * n_restarts), layout="constrained"
        )

    ax.add_patch(mp.patches.Rectangle(
        (edges[0], -0.5), edges[-1] - edges[0], n_restarts,
        facecolor="white", edgecolor=GRID, hatch="////", linewidth=0, zorder=0,
    ))
    mesh = ax.pcolormesh(
        edges, np.arange(n_restarts + 1) - 0.5, np.ma.masked_invalid(R),
        cmap=cmap, norm=norm, zorder=1,
    )
    for y in np.arange(1, n_restarts) - 0.5:
        ax.axhline(y, color="white", linewidth=2, zorder=2)

    # write the rank into each run of one value wide enough to hold it
    min_width = 0.05 * (edges[-1] - edges[0])
    for k, row in enumerate(R):
        start = 0
        for i in range(1, row.size + 1):
            if i < row.size and (row[i] == row[start]
                                 or (np.isnan(row[i]) and np.isnan(row[start]))):
                continue
            if np.isfinite(row[start]) and edges[i] - edges[start] >= min_width:
                r = int(row[start])
                dark = (r - lo) / max(n_levels - 1, 1) > 0.45
                ax.text(0.5 * (edges[start] + edges[i]), k, str(r),
                        ha="center", va="center", fontsize=8,
                        color="white" if dark else INK, zorder=3)
            start = i

    if final_ranks is not None or final_losses is not None:
        # tick labels on a twin rather than free text, so the layout reserves room
        right = ax.twinx()
        right.set_ylim(n_restarts - 0.5, -0.5)
        right.set_yticks(np.arange(n_restarts))
        labels = []
        for k in range(n_restarts):
            parts = []
            if final_ranks is not None and np.isfinite(final_ranks[k]):
                parts.append(f"rank {int(final_ranks[k])}")
            if final_losses is not None:
                parts.append(f"loss {final_losses[k]:.0e}")
            labels.append("   ".join(parts))
        right.set_yticklabels(labels, fontsize=8)
        for k, tick in enumerate(right.get_yticklabels()):
            reached = tol is None or final_losses is None or final_losses[k] < tol
            tick.set_color(INK if reached else MUTED)
        right.tick_params(axis="y", length=0)
        right.set_ylabel("at the best point reached", color=MUTED, fontsize=8)
        _hide_spines(right)

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(n_restarts - 0.5, -0.5)
    ax.set_yticks(np.arange(n_restarts))
    ax.set_yticklabels([f"restart {k}" for k in range(n_restarts)])
    ax.set_xlabel("Step")
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.tick_params(axis="x", colors=MUTED, labelcolor=INK)
    _hide_spines(ax)

    cb = ax.figure.colorbar(mesh, ax=ax, location="bottom", shrink=0.5,
                            aspect=30, pad=0.02, ticks=np.arange(lo, hi + 1))
    cb.set_label(
        f"Hessian rank (of {N_PARAMS}): count of "
        rf"$|\mu| > {_pow10(HESS_RTOL)}\,\max|\mu|$     "
        "hatched: restart stopped (reached tol or frozen)",
        color=INK, fontsize=8,
    )
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelcolor=INK, labelsize=8)

    every = "every step" if stride == 1 else f"every {stride} steps"
    ax.set_title(
        f"{_prefixed(label, 'Hessian rank of every restart')}  (sampled {every})",
        color=INK, fontsize=10, loc="left",
    )
    return ax


def _annotated_heatmap(ax, M, cmap, fmt="{:.2f}", fontsize=6.5):
    """imshow on [0, 1] with every finite cell's value printed in it."""
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1,
                   aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        if np.isfinite(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize, color="white" if v > 0.55 else INK)
    ax.tick_params(length=0, labelcolor=INK)
    _hide_spines(ax)
    return im


def plot_null_space_comparison(C_dual, c_targ, label=None, loss_tol=None):
    """Compare the spectrum-Jacobian null space at the target and every dual.

    Left: null_operator_share per point (row). Right: pairwise
    null_space_overlap. Degenerate points (no Jacobian) are empty rows.
    """
    points = [np.asarray(c_targ, float).reshape(-1)]
    points += [np.asarray(c, float).reshape(-1) for c in C_dual]
    names = ["target"] + [f"point {i}" for i in range(len(C_dual))]
    bases = [jacobian_null_space(c) for c in points]
    n = len(points)

    share = np.full((n, N_PARAMS), np.nan)
    overlap = np.full((n, n), np.nan)
    for i, Ni in enumerate(bases):
        if Ni is None:
            continue
        share[i] = null_operator_share(Ni)
        for j, Nj in enumerate(bases):
            if Nj is not None:
                overlap[i, j] = null_space_overlap(Ni, Nj)

    cmap = mp.colors.LinearSegmentedColormap.from_list(
        "share", ["#f4f3ee"] + RANK_RAMP[1:]
    )
    fig, (ax_s, ax_o) = plt.subplots(
        1, 2, figsize=(7.0 + 0.42 * n, 1.9 + 0.36 * n), layout="constrained",
        gridspec_kw={"width_ratios": [N_PARAMS, max(n, 1)]},
    )

    _annotated_heatmap(ax_s, share, cmap)
    ax_s.set_xticks(np.arange(N_PARAMS))
    ax_s.set_xticklabels(COEFF_LABELS)
    ax_s.set_yticks(np.arange(n))
    ax_s.set_yticklabels([
        f"{nm}  (nullity {'n/a' if N is None else N.shape[1]})"
        for nm, N in zip(names, bases)
    ], fontsize=8)
    ax_s.set_title(
        "Share of each operator in the null space\n"
        "1 = flat on its own, 0 = always moves the spectrum",
        color=INK, fontsize=9, loc="left",
    )

    im = _annotated_heatmap(ax_o, overlap, cmap)
    ax_o.set_xticks(np.arange(n))
    ax_o.set_xticklabels(["T"] + [str(i) for i in range(n - 1)], fontsize=8)
    ax_o.set_yticks(np.arange(n))
    ax_o.set_yticklabels([])
    # unrelated subspaces of nullity d1 <= d2 in R^12 overlap by d2/12 on average
    dims = [N.shape[1] for N in bases if N is not None]
    if dims:
        d = max(set(dims), key=dims.count)
        chance = f"; unrelated spaces of nullity {d} average {d / N_PARAMS:.2f}"
    else:
        chance = ""
    ax_o.set_title(
        "Overlap of null spaces, pair by pair\n"
        f"1 = one inside the other, 0 = orthogonal{chance}",
        color=INK, fontsize=9, loc="left",
    )
    ax_o.set_xlabel("point (T = target)", color=INK, fontsize=8)

    cb = fig.colorbar(im, ax=[ax_s, ax_o], location="bottom", shrink=0.4,
                      aspect=30, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelcolor=INK, labelsize=8)

    fig.suptitle(_prefixed(
        label, "Null space of $\\partial\\lambda/\\partial c$ at the target and "
        f"{_kept_points(len(C_dual), loss_tol)}",
    ), color=INK, fontsize=10)
    return fig


def plot_loss_history(loss_history, hess_log=None, hess_restart=None, ax=None):
    """One loss line per restart, loss_history (n_steps, K); with hess_log, the
    Hessian strips for restart hess_restart go underneath."""
    n_steps, n_restarts = loss_history.shape
    ax, hess_axes = _loss_axes(ax, n_steps, bool(hess_log))

    k_h = 0 if hess_restart is None else hess_restart
    focus = k_h if hess_axes is not None else None
    for k in range(n_restarts):
        ax.plot(loss_history[:, k], **_loss_line_style(k, n_restarts, focus))
    _loss_legend(ax, n_restarts, on_figure=hess_axes is not None)

    ax.set_title(f"Loss per restart over {n_restarts} restarts", color=INK)
    plot_hessian_log(hess_axes, hess_log, hess_restart, n_restarts,
                     loss_curve=loss_history[:, k_h])
    return ax


def _print_dual_degeneracy(spec_k, spec_targ, deg_tol):
    """Compare one dual's signed and |.| degeneracy patterns with the target's."""
    for what, spec_pair in (
        ("degeneracy", (spec_k, spec_targ)),
        ("|degeneracy|", (np.abs(spec_k), np.abs(spec_targ))),
    ):
        pattern, targ_pattern = (degeneracy_pattern(s, deg_tol) for s in spec_pair)
        agrees = ("matches target" if pattern == targ_pattern
                  else f"DIFFERS from target {targ_pattern}")
        print(f"  {what} {pattern} ({agrees})")
    print(
        f"  reflection residual {reflection_residual(spec_k):.2e}"
        f" (target {reflection_residual(spec_targ):.2e})"
    )


def run_grad_descent_sequential(
    get_loss, c_init, n_steps=2000, lr=1e-2, tol=1e-8, log_every=100,
    c_targ=None, ax=None, deg_tol=DEGEN_TOL, run_restart=None,
    hess_every=None, hess_restart=0,
):
    """Descend one restart at a time, drawing each finished curve on a live figure.

    Returns the same values as run_grad_descent. run_restart swaps in another
    optimizer with run_grad_descent's signature. With c_targ, live coefficient
    and spectrum-degeneracy figures also grow as restarts land below tol.
    """
    n_restarts = c_init.shape[0]
    if run_restart is None:
        run_restart = partial(run_grad_descent, lr=lr)

    # no event loop under a headless backend, so blocking there would hang
    interactive = not mp.get_backend().lower().startswith("agg")
    if interactive:
        plt.ion()

    ax, hess_axes = _loss_axes(ax, n_steps, bool(hess_every))
    focus = hess_restart if hess_axes is not None else None

    c_best = tr.empty_like(c_init)
    loss_best = tr.empty(n_restarts)
    loss_history = np.empty((n_steps, n_restarts))

    duals, dual_spectra = [], []
    coeff_ax = deg_axes = spec_targ = None
    if c_targ is not None:
        spec_targ = _spectrum(c_targ)
        # sized for every dual this run could add, since it is redrawn in place
        _, deg_axes = plt.subplots(
            1, 2, figsize=(2 * _degeneracy_width(min(n_restarts, MAX_SERIES)), 3.8),
            layout="constrained",
        )
        plot_spectrum_views(spec_targ, tol=deg_tol, axes=deg_axes)

    for k in range(n_restarts):
        print(f"\n--- restart {k}/{n_restarts} ---")
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

        ax.plot(loss_history[:, k], **_loss_line_style(k, n_restarts, focus))
        _loss_legend(ax, n_restarts, on_figure=hess_axes is not None)
        ax.relim()
        ax.autoscale_view()

        if loss_best[k].item() < tol:
            duals.append(c_best[k].numpy().copy())
            print(f"  dual {len(duals) - 1}: {duals[-1]}")

            if c_targ is not None:
                # redrawn in full: styling switches to one population past MAX_SERIES
                if coeff_ax is None:
                    _, coeff_ax = plt.subplots(figsize=(8.6, 3.6), layout="constrained")
                coeff_ax.clear()
                plot_dual_coefficients(np.array(duals), c_targ, ax=coeff_ax, loss_tol=tol)

                spec_k = get_spectrum(c_best[k : k + 1])[0].numpy()
                dual_spectra.append(spec_k)
                for deg_ax in deg_axes:
                    deg_ax.clear()
                plot_spectrum_views(
                    spec_targ, np.array(dual_spectra), tol=deg_tol, axes=deg_axes
                )
                _print_dual_degeneracy(spec_k, spec_targ, deg_tol)

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
    # ---- figures: set any of these to False to skip that figure ----------
    plots = dict(
        loss=True,          # loss of every restart over its descent
        loss_strips=True,   # loss decrease + Hessian spectrum of hess_restart
        spectrum=True,      # target spectrum against every point below tol
        coefficients=True,  # coefficients of every point below tol
    )
    # ------------------------------------------------------------------------

    do_search = True
    step_through = False  # one restart at a time, drawing each as it finishes

    n_restarts = 12
    n_steps = 2000
    tol = 1e-12
    hess_every = 50  # Hessian read-out interval along one restart; None for off
    hess_restart = 0
    deg_tol = DEGEN_TOL
    perturb_scale = 0.1  # restart cloud width around c_center

    h = 1.5
    J = 1
    c_targ = tr.tensor([[0, 0, h, J, 0, 0, 0, 0, h, -J, 0, 0]])
    # the target is itself a zero of the loss, so expect restarts to fall back
    # to it; None draws restarts over the whole coefficient space instead
    c_center = c_targ

    local = "" if c_center is None else f"_local{perturb_scale:g}"
    output_file = f"results/grad_descent_n{n_qb}_seed{seed}{local}.pkl"

    get_loss = get_loss_factory(c_targ)
    hess_log = []  # filled by the batch path only; empty on reload

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
            pk.dump((n_qb, c_targ, C_init.numpy(), C_final.numpy(),
                     losses.numpy(), loss_history), file)

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

    C_dual = C_final[losses < tol]
    print(f"\n{C_dual.shape[0]} points with loss < {tol:.0e}")

    if C_dual.shape[0] > 0:
        print("coefficients:")
        print(C_dual)

        spec_targ = get_spectrum(c_targ)
        spec_dual = get_spectrum(tr.tensor(C_dual))
        print("max spectrum deviation per point:")
        print((spec_dual - spec_targ).abs().max(dim=1).values.numpy())
        spec_targ = spec_targ[0].numpy()
        spec_dual = spec_dual.numpy()

        targ_patterns = (degeneracy_pattern(spec_targ, deg_tol),
                         degeneracy_pattern(np.abs(spec_targ), deg_tol))
        print(
            f"degeneracy per point (target {targ_patterns[0]}, "
            f"|target| {targ_patterns[1]}):"
        )
        for i, spec in enumerate(spec_dual):
            patterns = (degeneracy_pattern(spec, deg_tol),
                        degeneracy_pattern(np.abs(spec), deg_tol))
            flag = "" if patterns == targ_patterns else "   <- differs from target"
            print(
                f"  dual {i}: {patterns[0]}   |.| {patterns[1]}"
                f"   reflection {reflection_residual(spec):.2e}{flag}"
            )

        # zero loss is only isospectrality; this rules out the trivial maps
        print_duality(C_dual, c_targ.numpy())

    # ---- figures, toggled by plots at the top of main ----------------------
    # a step-through run already drew these live
    if not (do_search and step_through):
        if plots["loss"]:
            plot_loss_history(
                loss_history, hess_log if plots["loss_strips"] else None,
                hess_restart,
            )
        if C_dual.shape[0] > 0:
            if plots["spectrum"]:
                plot_spectrum_views(spec_targ, spec_dual, tol=deg_tol)
            if plots["coefficients"]:
                plot_dual_coefficients(C_dual, c_targ.numpy(), loss_tol=tol)
    plt.show()


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()
