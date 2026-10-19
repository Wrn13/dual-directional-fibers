"""The spectrum Jacobian J[i, k] = d lambda_i / d c_k, drawn.

H(c) = sum_k c_k A_k is linear in c, so by Hellmann-Feynman
J[i, k] = <psi_i| A_k |psi_i>. Every column sums to Tr A_k = 0 (Pauli strings
are traceless), so rank(J) <= 2^n - 1. Kernel directions are purely
off-diagonal in the energy eigenbasis: they rotate eigenvectors without moving
eigenvalues, which is what a dual direction is.
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

tr.set_default_dtype(tr.float64)

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_neutral_orange",
    ["#16497f", "#5d95cf", "#b9cee4", "#e8e7e0", "#f3bd97", "#ef9463", "#b4441b"],
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "neutral_blue", ["#f4f3ee", "#c3d7ee", "#6f9ed2", "#2a78d6", "#123a6b"]
)
# log axes can't show exact zeros; singular values at or below this sit on the floor
SV_FLOOR = 1e-17


def operator_terms():
    """(12, 2**n, 2**n): A_k is H evaluated at the k-th unit vector."""
    basis = tr.eye(N_PARAMS, dtype=tr.get_default_dtype())
    return np.stack([get_Hamiltonian(basis[k : k + 1])[0].numpy() for k in range(N_PARAMS)])


def jacobian_by_expectation(c, A=None):
    """(J, eigenvalues, eigenvectors, A) at c, with J[i, k] = <psi_i|A_k|psi_i>."""
    if A is None:
        A = operator_terms()
    H = get_Hamiltonian(tr.as_tensor(np.asarray(c), dtype=tr.get_default_dtype()).reshape(1, -1))
    lam, V = np.linalg.eigh(H[0].numpy())
    J = np.einsum("ji,kjl,li->ik", V.conj(), A, V).real
    return J, lam, V, A


def _in_eigenbasis(u, A, V):
    """|<psi_i| sum_k u_k A_k |psi_j>|."""
    return np.abs(V.conj().T @ np.einsum("k,kab->ab", u, A) @ V)


def _bare(ax):
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def _colorbar(im, cax, label, fontsize):
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, color=INK, fontsize=fontsize)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK, length=2, labelsize=fontsize - 1)


def plot_jacobian_table(J, lam, ax, cax=None, annotate=True):
    """J as a table of expectation values: eigenstates by operators."""
    vmax = np.abs(J).max()
    # Symmetric Normalize, not TwoSlopeNorm: identical here, but TwoSlopeNorm.inverse
    # returns inf at the top cell and crashes the toolbar cursor read-out.
    im = ax.imshow(J, cmap=DIVERGING, norm=Normalize(-vmax, vmax), aspect="auto")
    ax.set_xticks(np.arange(N_PARAMS))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(lam)))
    ax.set_yticklabels([f"$\\lambda_{{{i}}}$ = {v:+.3f}" for i, v in enumerate(lam)])
    ax.set_ylabel("eigenstate $|\\psi_i\\rangle$")
    ax.set_title(
        "$J_{ik} = \\partial\\lambda_i/\\partial c_k = \\langle\\psi_i|A_k|\\psi_i\\rangle$",
        color=INK, fontsize=11,
    )
    _bare(ax)
    if annotate:
        for (i, j), v in np.ndenumerate(J):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.62 * vmax else INK)
    if cax is not None:
        _colorbar(im, cax, "$\\langle\\psi_i|A_k|\\psi_i\\rangle$", 9)
    return im


def plot_column_sums(J, ax, vmax):
    """Strip of column sums under the table; prints the computed sums, not a literal 0."""
    sums = J.sum(axis=0)
    ax.imshow(sums[None], cmap=DIVERGING, aspect="auto", norm=Normalize(-vmax, vmax))
    ax.set_xticks(np.arange(N_PARAMS))
    ax.set_xticklabels(COEFF_LABELS)
    ax.set_yticks([0])
    ax.set_yticklabels(["$\\sum_i$"])
    _bare(ax)
    for j, v in enumerate(sums):
        ax.text(j, 0, f"{v:.0e}", ha="center", va="center", fontsize=6, color=INK)
    n = J.shape[0]
    ax.set_xlabel(
        "operator $A_k$ that coefficient $k$ multiplies.   Each column sums "
        "to $\\mathrm{Tr}\\,A_k = 0$ (printed: the computed sum),\nso the "
        f"{n} rows are linearly dependent and rank($J$) $\\leq$ {n - 1}",
        fontsize=8.5, color=MUTED,
    )


def plot_singular_values(points, ax, rank_rtol=JAC_RTOL):
    """Log-scale singular values of J at each (label, c); rank = count above the cut."""
    for i, (label, c) in enumerate(points):
        s = np.linalg.svd(jacobian_by_expectation(c)[0], compute_uv=False)
        rank = int((s > rank_rtol * s.max()).sum())
        # normalised per point because the rank cut is relative to each point's max
        ax.plot(
            np.arange(1, s.size + 1), np.maximum(s / s.max(), SV_FLOOR),
            color=SERIES_COLORS[i], marker=SERIES_MARKERS[i], markersize=7,
            linewidth=1.6, markeredgecolor="white", markeredgewidth=0.8,
            label=f"{label}: rank {rank}, nullity {N_PARAMS - rank}",
        )
    ax.axhline(rank_rtol, color=BASELINE, linewidth=1, linestyle="--", zorder=0)
    ax.annotate(f"rank cut {rank_rtol:.0e}", xy=(1, rank_rtol), xytext=(2, 4),
                textcoords="offset points", fontsize=7.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(SV_FLOOR / 3, None)
    ax.set_xlabel("singular value index")
    ax.set_ylabel(f"$\\sigma / \\sigma_{{max}}$  (floor {SV_FLOOR:.0e})")
    ax.set_title(
        f"Singular values of $J$; rank = count above the cut,\nnullity = {N_PARAMS} $-$ rank",
        color=INK, fontsize=10,
    )
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left")
    _style_axes(ax)


def plot_eigenbasis_perturbation(v, A, V, ax, title, cax=None, vmax=None):
    """|sum_k v_k A_k| in the energy eigenbasis, where a dual direction has no diagonal."""
    Wt = _in_eigenbasis(v, A, V)
    im = ax.imshow(Wt, cmap=SEQUENTIAL, vmin=0, vmax=Wt.max() if vmax is None else vmax,
                   aspect="equal")
    n = Wt.shape[0]
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color=MUTED, linewidth=0.9,
            linestyle=(0, (4, 3)), zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    _bare(ax)
    ax.set_title(
        f"{title}\nlargest diagonal $|\\langle\\psi_i|V|\\psi_i\\rangle|$ = {np.diag(Wt).max():.1e}",
        color=INK, fontsize=9.5,
    )
    if cax is not None:
        _colorbar(im, cax, "$|\\langle\\psi_i|V|\\psi_j\\rangle|$", 8)
    return im


def plot_jacobian_anatomy(c, comparisons=(), annotate=True, name="this point"):
    """Table, trace identity, singular values (c vs comparisons) and kernel vs top direction."""
    J, lam, V, A = jacobian_by_expectation(c)
    _, s, Vt = np.linalg.svd(J)
    rank = int((s > JAC_RTOL * s.max()).sum())

    fig = plt.figure(figsize=(11.6, 8.2), layout="constrained")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.45, 0.17, 1.05],
                          width_ratios=[1.0, 1.0, 1.0, 0.045], hspace=0.05)
    plot_jacobian_table(J, lam, fig.add_subplot(gs[0, 0:3]), cax=fig.add_subplot(gs[0, 3]),
                        annotate=annotate)
    plot_column_sums(J, fig.add_subplot(gs[1, 0:3]), np.abs(J).max())
    fig.suptitle(f"Spectrum Jacobian $J$ at the {name}", color=INK, fontsize=12)
    plot_singular_values(((name, c), *comparisons), fig.add_subplot(gs[2, 0]))

    vmax = max(_in_eigenbasis(u, A, V).max() for u in (Vt[rank], Vt[0]))
    plot_eigenbasis_perturbation(
        Vt[rank], A, V, fig.add_subplot(gs[2, 1]),
        f"$V$ from null vector $u_{{{rank + 1}}}$ of $J$ ($J u = 0$)", vmax=vmax,
    )
    plot_eigenbasis_perturbation(
        Vt[0], A, V, fig.add_subplot(gs[2, 2]),
        f"$V$ from top singular vector $u_1$ of $J$ ($\\sigma_1$ = {s[0]:.2f})",
        cax=fig.add_subplot(gs[2, 3]), vmax=vmax,
    )
    return fig


def main():
    h, J_coupling = 1.5, 1
    c_targ = np.array([0, 0, h, J_coupling, 0, 0, 0, 0, h, -J_coupling, 0, 0], float)

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
    comparisons.append(("random $c \\sim N(0, 1)$", np.random.default_rng(0).normal(size=12)))

    fig = plot_jacobian_anatomy(c_targ, comparisons=comparisons, name="target")
    out = "images/jacobian_anatomy.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    plt.show()


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False
    main()
