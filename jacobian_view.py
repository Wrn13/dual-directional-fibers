"""What the spectrum Jacobian is, drawn.

J[i, k] = d lambda_i / d c_k is not an abstract derivative here. The
Hamiltonian is linear in the coefficients, H(c) = sum_k c_k A_k, so
dH/dc_k = A_k exactly and Hellmann-Feynman gives

    J[i, k] = <psi_i| A_k |psi_i>

a table of expectation values: row i is the operator profile of eigenstate
i, column k is how every level responds to turning up coupling k. Three
consequences are what this module draws.

  1. Every column sums to Tr A_k = 0, because every Pauli string is
     traceless. That is the one universal relation among the eigenvalues,
     and the reason the Jacobian can never have rank 8.
  2. The rank counts how many independent directions the 12 coefficients can
     push the spectrum in. Generically 7, the most the trace allows; where it
     drops, the coefficient family is tangent to the isospectral orbit rather
     than cutting it, and the dual family there is one dimension larger.
  3. A kernel direction is a perturbation that is purely off-diagonal in the
     energy eigenbasis. It rotates the eigenvectors and leaves the
     eigenvalues alone, which is exactly what a dual direction is.

Run as a script to draw all three for the KW target, against a dual and a
random point for comparison.
"""

import pickle as pk

import numpy as np
import torch as tr
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from ising_finder import get_Hamiltonian, n_qb
from grad_descent import (
    BASELINE,
    COEFF_LABELS,
    INK,
    JAC_RTOL,
    MUTED,
    N_PARAMS,
    SERIES_COLORS,
    SERIES_MARKERS,
    _style_axes,
    seed,
)

np.set_printoptions(linewidth=10000000, threshold=1000000)
tr.set_default_dtype(tr.float64)

# Diverging: two hues with a neutral midpoint, for a signed quantity read
# against zero. Blue and orange are the pair that survives colour vision
# deficiency, and the midpoint is neutral so zero never reads as a hue.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_neutral_orange",
    ["#16497f", "#5d95cf", "#b9cee4", "#e8e7e0", "#f3bd97", "#ef9463", "#b4441b"],
)
# Sequential: one hue, light to dark, for a magnitude with no sign
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "neutral_blue", ["#f4f3ee", "#c3d7ee", "#6f9ed2", "#2a78d6", "#123a6b"]
)


def operator_terms():
    """(12, 2**n, 2**n) the operator each coefficient multiplies.

    H is linear in c with no constant term, so the operator multiplying
    coefficient k is just H evaluated at the k-th unit vector.
    """
    basis = tr.eye(N_PARAMS, dtype=tr.get_default_dtype())
    return np.stack(
        [get_Hamiltonian(basis[k : k + 1])[0].numpy() for k in range(N_PARAMS)]
    )


def jacobian_by_expectation(c, A=None):
    """(J, eigenvalues, eigenvectors, A) at one coefficient point.

    J is built as <psi_i| A_k |psi_i> rather than by autodiff: same matrix to
    machine precision, but it costs one eigendecomposition instead of a
    reverse-mode pass, and it hands back the eigenvectors, which is what the
    kernel panels need.
    """
    if A is None:
        A = operator_terms()
    H = get_Hamiltonian(
        tr.as_tensor(np.asarray(c), dtype=tr.get_default_dtype()).reshape(1, -1)
    )[0].numpy()
    lam, V = np.linalg.eigh(H)
    J = np.einsum("ji,kjl,li->ik", V.conj(), A, V).real
    return J, lam, V, A


def _annotate_cells(ax, M, vmax, fmt="{:+.2f}", fontsize=6.5):
    """Print every cell's value, in whichever ink stays legible on it."""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(
                j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                color="white" if abs(v) > 0.62 * vmax else INK,
            )


def plot_jacobian_table(J, lam, ax, cax=None, annotate=True):
    """The Jacobian as what it is: expectation values, states by operators."""
    vmax = np.abs(J).max()
    # Normalize on a symmetric range, not TwoSlopeNorm: the range is already
    # centred on zero so the two render identically, but TwoSlopeNorm.inverse
    # extrapolates past 1.0 and hands back inf, which crashes the toolbar's
    # cursor read-out when the pointer lands on the largest cell.
    norm = Normalize(vmin=-vmax, vmax=vmax)
    im = ax.imshow(J, cmap=DIVERGING, norm=norm, aspect="auto")

    # the operator names go under the column-sum strip, which sits directly
    # below this and shares the same columns
    ax.set_xticks(np.arange(N_PARAMS))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(lam)))
    ax.set_yticklabels([f"$\\lambda_{{{i}}}$ = {v:+.3f}" for i, v in enumerate(lam)])
    ax.set_ylabel("eigenstate $|\\psi_i\\rangle$")
    ax.set_title(
        "$J_{ik} = \\partial\\lambda_i/\\partial c_k "
        "= \\langle\\psi_i|A_k|\\psi_i\\rangle$",
        color=INK, fontsize=11,
    )
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    if annotate:
        _annotate_cells(ax, J, vmax)
    if cax is not None:
        cb = plt.colorbar(im, cax=cax, orientation="vertical")
        cb.set_label("$\\langle\\psi_i|A_k|\\psi_i\\rangle$", color=INK, fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelcolor=INK, length=2, labelsize=8)
    return im


def plot_column_sums(J, ax, vmax):
    """The trace identity, as the strip under the table: every column sums to 0.

    The cells print the sums as computed, not a literal 0, so the strip shows
    how close to zero they actually came.
    """
    sums = J.sum(axis=0).reshape(1, -1)
    ax.imshow(sums, cmap=DIVERGING, aspect="auto",
              norm=Normalize(vmin=-vmax, vmax=vmax))
    ax.set_xticks(np.arange(N_PARAMS))
    ax.set_xticklabels(COEFF_LABELS)
    ax.set_yticks([0])
    ax.set_yticklabels(["$\\sum_i$"])
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for j in range(N_PARAMS):
        ax.text(j, 0, f"{sums[0, j]:.0e}", ha="center", va="center",
                fontsize=6, color=INK)
    n_levels = J.shape[0]
    ax.set_xlabel(
        "operator $A_k$ that coefficient $k$ multiplies.   Each column sums "
        "to $\\mathrm{Tr}\\,A_k = 0$ (printed: the computed sum),\nso the "
        f"{n_levels} rows are linearly dependent and rank($J$) "
        f"$\\leq$ {n_levels - 1}",
        fontsize=8.5, color=MUTED,
    )


# singular values at or below this are drawn on the axis floor: a log axis
# has nowhere to put an exact zero, and several of these are exactly zero
SV_FLOOR = 1e-17


def plot_singular_values(points, ax, rank_rtol=JAC_RTOL):
    """Singular values of J at several points; the rank is where they hit zero.

    Log scale, because the whole question is which singular values are zero
    and which are merely small - on a linear axis a rank-7 point whose 7th
    value is 4e-4 is indistinguishable from a rank-6 point whose 7th is 1e-16,
    and those are different strata. Exact zeros are clipped to the floor.

    Each point's values are divided by its own largest, since the rank cut is
    relative to that: one cut line is then the right one for every point.
    """
    for i, (label, c) in enumerate(points):
        J, _, _, _ = jacobian_by_expectation(c)
        s = np.linalg.svd(J, compute_uv=False)
        rank = int((s > rank_rtol * s.max()).sum())
        ax.plot(
            np.arange(1, s.size + 1), np.maximum(s / s.max(), SV_FLOOR),
            color=SERIES_COLORS[i], marker=SERIES_MARKERS[i], markersize=7,
            linewidth=1.6, markeredgecolor="white", markeredgewidth=0.8,
            label=f"{label}: rank {rank}, nullity {N_PARAMS - rank}",
        )
    ax.axhline(rank_rtol, color=BASELINE, linewidth=1, linestyle="--", zorder=0)
    ax.annotate(
        f"rank cut {rank_rtol:.0e}", xy=(1, rank_rtol), xytext=(2, 4),
        textcoords="offset points", fontsize=7.5, color=MUTED,
    )
    ax.set_yscale("log")
    ax.set_ylim(SV_FLOOR / 3, None)
    ax.set_xlabel("singular value index")
    ax.set_ylabel(f"$\\sigma / \\sigma_{{max}}$  (floor {SV_FLOOR:.0e})")
    ax.set_title(
        "Singular values of $J$; rank = count above the cut,\n"
        f"nullity = {N_PARAMS} $-$ rank",
        color=INK, fontsize=10,
    )
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left")
    _style_axes(ax)


def plot_eigenbasis_perturbation(v, A, V, ax, title, cax=None, vmax=None):
    """|V| in the energy eigenbasis, where a dual direction has no diagonal."""
    W = np.einsum("k,kab->ab", v, A)
    Wt = np.abs(V.conj().T @ W @ V)
    if vmax is None:
        vmax = Wt.max()
    im = ax.imshow(Wt, cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="equal")
    # the diagonal is the whole comparison, so mark where to look
    n = Wt.shape[0]
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color=MUTED, linewidth=0.9,
            linestyle=(0, (4, 3)), zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    diag = np.abs(np.diag(Wt)).max()
    ax.set_title(
        f"{title}\nlargest diagonal $|\\langle\\psi_i|V|\\psi_i\\rangle|$ = {diag:.1e}",
        color=INK, fontsize=9.5,
    )
    if cax is not None:
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("$|\\langle\\psi_i|V|\\psi_j\\rangle|$", color=INK, fontsize=8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelcolor=INK, length=2, labelsize=7)
    return im


def plot_jacobian_anatomy(c, comparisons=(), annotate=True, name="this point"):
    """The whole read-out for one point: table, trace identity, rank, kernel.

    comparisons is a sequence of (label, c) drawn alongside c in the singular
    value panel, so the rank at this point can be read against others.
    name is what c is called in the legend and the figure title.
    """
    J, lam, V, A = jacobian_by_expectation(c)
    U, s, Vt = np.linalg.svd(J)
    rank = int((s > JAC_RTOL * s.max()).sum())

    fig = plt.figure(figsize=(11.6, 8.2), layout="constrained")
    # the colorbars get gridspec columns of their own rather than stealing
    # width from an axes, so the table and the sum strip under it stay in
    # register column for column
    gs = fig.add_gridspec(
        3, 4, height_ratios=[1.45, 0.17, 1.05],
        width_ratios=[1.0, 1.0, 1.0, 0.045], hspace=0.05,
    )
    ax_J = fig.add_subplot(gs[0, 0:3])
    cax_J = fig.add_subplot(gs[0, 3])
    ax_sum = fig.add_subplot(gs[1, 0:3])
    ax_sv = fig.add_subplot(gs[2, 0])
    ax_ker = fig.add_subplot(gs[2, 1])
    ax_non = fig.add_subplot(gs[2, 2])
    cax_V = fig.add_subplot(gs[2, 3])

    plot_jacobian_table(J, lam, ax_J, cax=cax_J, annotate=annotate)
    plot_column_sums(J, ax_sum, np.abs(J).max())

    fig.suptitle(f"Spectrum Jacobian $J$ at the {name}", color=INK, fontsize=12)
    plot_singular_values(((name, c), *comparisons), ax_sv)

    # the last right-singular vectors span the kernel; the first does not
    W_all = [np.abs(V.conj().T @ np.einsum("k,kab->ab", u, A) @ V)
             for u in (Vt[rank], Vt[0])]
    vmax = max(w.max() for w in W_all)
    # V = sum_k u_k A_k for the right-singular vector u named in each title
    plot_eigenbasis_perturbation(
        Vt[rank], A, V, ax_ker,
        f"$V$ from null vector $u_{{{rank + 1}}}$ of $J$ ($J u = 0$)", vmax=vmax
    )
    plot_eigenbasis_perturbation(
        Vt[0], A, V, ax_non,
        f"$V$ from top singular vector $u_1$ of $J$ ($\\sigma_1$ = {s[0]:.2f})",
        cax=cax_V, vmax=vmax,
    )

    return fig


def main():
    h, J_coupling = 1.5, 1
    c_targ = np.array(
        [0, 0, h, J_coupling, 0, 0, 0, 0, h, -J_coupling, 0, 0], float
    )

    comparisons = []
    try:
        with open(f"results/newton_n{n_qb}_seed{seed}_local0.1.pkl", "rb") as file:
            _, _, _, C_final, losses, _ = pk.load(file)
        loss_tol = 1e-12
        duals = C_final[losses < loss_tol]
        if len(duals):
            comparisons.append((f"Newton endpoint, loss < {loss_tol:.0e}", duals[0]))
    except FileNotFoundError:
        print("no saved duals yet; run newton_raphson.py to add one here")
    comparisons.append(
        ("random $c \\sim N(0, 1)$", np.random.default_rng(0).normal(size=12))
    )

    fig = plot_jacobian_anatomy(c_targ, comparisons=comparisons, name="target")
    out = "images/jacobian_anatomy.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    plt.show()


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()
