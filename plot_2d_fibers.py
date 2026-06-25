from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
import torch as tr  # only used by the loss wrapper you pass in

def evaluate_loss_gradient_field(f, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5), n_grid=21):
    """Evaluate the loss-gradient field f = grad L on a grid. See earlier docstring."""
    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.linspace(y_range[0], y_range[1], n_grid)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=0)
    F = f(pts)
    return XX, YY, F[0].reshape(XX.shape), F[1].reshape(XX.shape)


def evaluate_loss_field(
    loss_fn: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float] = (-0.5, 1.5),
    y_range: tuple[float, float] = (-0.5, 1.5),
    n_grid: int = 160,
    chunk: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Evaluate the scalar loss :math:`L(\alpha,\beta)` on a regular grid.

    Evaluated in column chunks to bound memory: each point builds a
    :math:`2^{n_{qb}}\times 2^{n_{qb}}` complex Hamiltonian, so the full
    ``(K, 64, 64)`` complex128 batch for a fine grid would be multiple GB. With
    ``chunk=2000`` the peak is ~130 MB of Hamiltonian per chunk.

    Parameters
    ----------
    loss_fn : Callable[[np.ndarray], np.ndarray]
        Maps grid points ``(2, K)`` (row 0 = ``alpha``, row 1 = ``beta``) to the
        scalar loss ``(K,)``. A thin wrapper over your torch ``get_loss`` (see
        call-site note below).
    n_grid : int
        Samples per axis for the background (independent of the quiver grid).
    chunk : int
        Number of grid points per evaluation batch.

    Returns
    -------
    XX, YY, L : np.ndarray
        Meshgrid coordinates and the loss surface, each shape ``(n_grid, n_grid)``.
    """
    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.linspace(y_range[0], y_range[1], n_grid)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=0)   # (2, n_grid**2)
    vals = np.empty(pts.shape[1], dtype=float)
    for s in range(0, pts.shape[1], chunk):
        vals[s:s + chunk] = np.asarray(loss_fn(pts[:, s:s + chunk]), dtype=float)
    return XX, YY, vals.reshape(XX.shape)


def _draw_fiber_vectors(ax, fiber, f, color, count, normalize):
    r"""Overlay f(c) sampled along one fiber, with a white halo for contrast.

    On the fiber :math:`f(c)=\eta\,\hat c_\mathrm{dir}`, so these arrows lie along
    :math:`\pm\hat c_\mathrm{dir}`, flipping 180 deg at the sign changes of eta.
    """
    C = f(fiber)
    n = fiber.shape[1]
    stride = max(1, n // max(1, count))
    P, Cs = fiber[:, ::stride], C[:, ::stride]
    if normalize:
        mag = np.hypot(Cs[0], Cs[1])
        safe = np.isfinite(mag) & (mag > 0.0)
        denom = np.where(mag == 0.0, 1.0, mag)
        Cs = np.where(safe, Cs / denom, np.nan)
        q = ax.quiver(P[0], P[1], Cs[0], Cs[1], color=color, angles="xy",
                      scale_units="xy", scale=12, width=0.006, zorder=4)
    else:
        q = ax.quiver(P[0], P[1], Cs[0], Cs[1], color=color, angles="xy",
                      scale=30, width=0.006, zorder=4)
    q.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def plot_fibers_and_field(
    V: np.ndarray,
    V_1: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    x_range: tuple[float, float] = (-0.5, 1.5),
    y_range: tuple[float, float] = (-0.5, 1.5),
    n_grid: int = 21,
    normalize: bool = True,
    descent: bool = False,
    direction: np.ndarray | None = None,
    direction_anchor: tuple[float, float] = (-0.30, -0.30),
    show_fiber_vectors: bool = True,
    fiber_vector_count: int = 40,
    fiber_vector_normalize: bool = True,
    loss_background: bool = True,
    n_loss_grid: int = 160,
    loss_chunk: int = 2000,
    loss_cmap: str = "magma",
    field_flat_color: str | None = None,
    output_path: str | None = None,
) -> None:
    r"""Two directional fibers over the loss surface, gradient field, and direction.

    Layers (bottom to top): scalar loss :math:`L(\alpha,\beta)` (log color),
    the gradient direction field :math:`\nabla L`, the two fibers, the on-fiber
    vectors :math:`f(c)`, and the fixed direction :math:`\hat c_\mathrm{dir}`.

    Parameters
    ----------
    loss_fn : Callable[[np.ndarray], np.ndarray] | None
        Scalar loss for the background; ``(2, K) -> (K,)``. Required if
        ``loss_background`` is ``True``.
    loss_background : bool
        Paint :math:`L` underneath everything (log scale; zeros at the optima
        are clipped to the colormap floor).
    field_flat_color : str | None
        When the background is on, the gradient *magnitude* is already implied by
        the loss surface, so the field arrows default to a single flat color
        (``"white"``) and their magnitude colorbar is dropped, leaving one
        colorbar for :math:`L`. Set explicitly to override.
    """
    if loss_background and loss_fn is None:
        raise ValueError("loss_background=True requires loss_fn.")
    if loss_background and field_flat_color is None:
        field_flat_color = "white"

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # ---- Layer 0: loss surface background (log color, zeros clipped to floor) ----
    if loss_background:
        LXX, LYY, L = evaluate_loss_field(loss_fn, x_range, y_range, n_loss_grid, loss_chunk)
        finite = np.isfinite(L)
        vmax = float(np.nanmax(L))
        pos = L[finite & (L > 0.0)]
        vmin = max(float(pos.min()) if pos.size else vmax * 1e-8, vmax * 1e-8)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        Lclip = np.clip(np.where(finite, L, vmin), vmin, vmax)
        bg = ax.pcolormesh(LXX, LYY, Lclip, norm=norm, cmap=loss_cmap,
                           shading="gouraud", zorder=0)
        fig.colorbar(bg, ax=ax, label=r"$L(\alpha,\beta)$", shrink=0.85)

    # ---- Layer 1: gradient direction field ----
    XX, YY, U, W = evaluate_loss_gradient_field(f, x_range, y_range, n_grid)
    if descent:
        U, W = -U, -W
    mag = np.hypot(U, W)
    finite = np.isfinite(mag)
    if normalize:
        safe = finite & (mag > 0.0)
        denom = np.where(mag == 0.0, 1.0, mag)
        U = np.where(safe, U / denom, np.nan)
        W = np.where(safe, W / denom, np.nan)
    else:
        U = np.where(finite, U, np.nan)
        W = np.where(finite, W, np.nan)

    if field_flat_color is not None:
        ax.quiver(XX, YY, U, W, color=field_flat_color, angles="xy", pivot="mid",
                  scale=30 if normalize else None, width=0.0035, alpha=0.6, zorder=2)
    else:
        color = np.where(finite, np.log10(mag + 1e-12), np.nan)
        q = ax.quiver(XX, YY, U, W, color, cmap="viridis", angles="xy", pivot="mid",
                      scale=30 if normalize else None, width=0.004, alpha=0.9, zorder=2)
        fig.colorbar(q, ax=ax, label=r"$\log_{10}\,\|\nabla L\|$", shrink=0.85)

    # ---- Layer 2: the two fibers (white halo so they read on the dark valley) ----
    l1, = ax.plot(V[0], V[1], color="black", lw=2.0, zorder=3,
                  label=r"Fiber 1 (seed $(0,1)$, Ising)")
    l2, = ax.plot(V_1[0], V_1[1], color="crimson", lw=2.0, zorder=3,
                  label=r"Fiber 2 (seed $(1,0)$, KW dual)")
    for ln in (l1, l2):
        ln.set_path_effects([pe.withStroke(linewidth=4.0, foreground="white")])

    # ---- Layer 3: on-fiber vectors f(c) ----
    if show_fiber_vectors:
        _draw_fiber_vectors(ax, V,   f, "black",   fiber_vector_count, fiber_vector_normalize)
        _draw_fiber_vectors(ax, V_1, f, "crimson", fiber_vector_count, fiber_vector_normalize)

    # ---- Layer 4: isospectral pair ----
    ax.scatter([0.0, 1.0], [1.0, 0.0], s=70, marker="o", facecolors="white",
               edgecolors="black", linewidths=1.5, zorder=6)
    ax.annotate("Ising", (0.0, 1.0), textcoords="offset points", xytext=(8, 8),
                fontsize=11, color="white")
    ax.annotate("KW dual", (1.0, 0.0), textcoords="offset points", xytext=(8, 8),
                fontsize=11, color="white")

    # ---- Layer 5: fixed direction \hat c_dir ----
    if direction is not None:
        d = np.asarray(direction, dtype=float).ravel()
        nrm = np.linalg.norm(d)
        if nrm == 0.0:
            raise ValueError("`direction` is the zero vector; cannot normalize.")
        d = d / nrm
        Larrow = 0.45
        ax.quiver(direction_anchor[0], direction_anchor[1], d[0] * Larrow, d[1] * Larrow,
                  angles="xy", scale_units="xy", scale=1, color="#60a5fa",
                  width=0.011, zorder=7)
        ax.scatter(*direction_anchor, s=20, color="#60a5fa", zorder=7)
        ax.annotate(r"$\hat c_\mathrm{dir}$", np.asarray(direction_anchor) + d * Larrow,
                    textcoords="offset points", xytext=(6, 6), color="#60a5fa", fontsize=12)

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"Directional fibers over $L(\alpha,\beta)$ and $\nabla_c L$"
                 + (r"  ($-\nabla L$ shown)" if descent else ""))
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()