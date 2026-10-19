"""Newton-Raphson search for dual tensor product structures.

The search of grad_descent.py (restarts over the 12 Ising coefficients, looking
for points isospectral with a target) stepped with Newton-Raphson instead of
Adam; everything else is imported from grad_descent.

Coefficient layout is [XX, YY, ZZ, X, Y, Z, X_1, Y_1, Z_1, X_N, Y_N, Z_N].
"""

import pickle as pk
from functools import partial

import numpy as np
import torch as tr
import matplotlib as mp
import matplotlib.pyplot as plt

from ising_finder import get_spectrum, get_loss_factory, n_qb
from duality import print_duality
from grad_descent import (
    DEGEN_TOL,
    INK,
    MAX_SERIES,
    N_PARAMS,
    degeneracy_pattern,
    hessian_rank,
    jacobian_rank,
    log_hessian,
    log_ranks,
    perturbed_inits,
    plot_dual_coefficients,
    plot_dual_ranks,
    plot_loss_history,
    plot_null_space_comparison,
    plot_rank_map,
    plot_spectrum_views,
    print_degeneracy,
    print_hessian,
    print_symmetry,
    random_inits,
    reflection_residual,
    run_grad_descent_sequential,
    seed,
)

np.set_printoptions(linewidth=10000000, threshold=1000000)
tr.set_printoptions(linewidth=1000)
tr.set_default_dtype(tr.float64)

# relative singular-value cutoff for the Newton solve; the Hessian is exactly
# rank deficient at a dual, and this keeps the step finite there
NEWTON_RCOND = 1e-10


def newton_direction(H, g, rcond=NEWTON_RCOND, damping=0.0):
    """Minimum-norm solution dc of (H + damping I) dc = -g, H (12, 12), g (12,).

    Least squares, not solve: H is singular along the dual families, and lstsq
    drops those flat directions instead of dividing by their zero curvature.
    """
    A = np.asarray(H)
    A = 0.5 * (A + A.T)
    if damping:
        A = A + damping * np.eye(A.shape[0])
    dc, *_ = np.linalg.lstsq(A, -np.asarray(g), rcond=rcond)
    return dc


def newton_iterate(
    get_loss, c, rcond=NEWTON_RCOND, damping=0.0, step_scale=1.0, backtrack=0
):
    """One Newton step from c (12,); returns (c_next, loss_here, alpha).

    With backtrack = m the step is halved up to m times while it raises the
    loss. alpha is 0 if no tried step improved, nan if derivatives were not
    finite.
    """

    def loss_one(u):
        return get_loss(u.reshape(1, -1))[0]

    v = tr.as_tensor(c, dtype=tr.get_default_dtype()).detach().reshape(-1)
    loss_here = loss_one(v).item()

    g = tr.func.grad(loss_one)(v)
    H = tr.func.hessian(loss_one)(v)
    if not (tr.isfinite(g).all() and tr.isfinite(H).all()):
        # eigvalsh is not differentiable where two eigenvalues collide
        return v, loss_here, float("nan")

    dc = tr.as_tensor(
        newton_direction(H.numpy(), g.numpy(), rcond=rcond, damping=damping)
    )

    alpha = step_scale
    for _ in range(backtrack + 1):
        c_next = v + alpha * dc
        loss_next = loss_one(c_next).item()
        if not backtrack or (np.isfinite(loss_next) and loss_next < loss_here):
            return c_next, loss_here, alpha
        alpha /= 2
    return v, loss_here, 0.0


def run_newton(
    get_loss, c_init, n_steps=60, tol=1e-8, log_every=5,
    rcond=NEWTON_RCOND, damping=0.0, step_scale=1.0, backtrack=0,
    hess_every=None, hess_index=0, hess_log=None, rank_every=None,
    rank_log=None,
):
    """Newton-Raphson every restart; drop-in run_restart for grad_descent.

    c_init is (K, 12); returns (c_best, loss_best, loss_history (n_steps, K)).
    Restarts that converge or stall are frozen with their loss carried forward.
    hess_every / hess_log sample restart hess_index's Hessian, rank_every /
    rank_log every active restart's Hessian rank. Both are sampled before the
    step, at the point loss_history[step] describes, so the Hessian at step t
    is the one the step from t was taken with.
    """
    c = c_init.clone().detach()
    n_restarts = c.shape[0]

    c_best = c.clone()
    loss_best = tr.full((n_restarts,), np.inf)
    loss_history = np.empty((n_steps, n_restarts))
    done = np.zeros(n_restarts, dtype=bool)

    for step in range(n_steps):
        if rank_every and step % rank_every == 0:
            log_ranks(rank_log, step, get_loss, c, ~done)

        if (hess_every and not done[hess_index]
                and (step % hess_every == 0 or step == n_steps - 1)):
            log_hessian(hess_log, step, print_hessian(
                get_loss, c[hess_index],
                label=f"  step {step:5d} Hessian: ",
                loss=get_loss(c[hess_index].reshape(1, -1))[0].item(),
            ))

        for k in range(n_restarts):
            if done[k]:
                loss_history[step, k] = loss_best[k].item()
                continue

            c_k, loss_k, alpha = newton_iterate(
                get_loss, c[k], rcond=rcond, damping=damping,
                step_scale=step_scale, backtrack=backtrack,
            )
            loss_history[step, k] = loss_k
            if loss_k < loss_best[k].item():
                loss_best[k] = loss_k
                c_best[k] = c[k]
            c[k] = c_k

            if loss_k < tol:
                done[k] = True
            elif not np.isfinite(alpha):
                print(f"  restart {k}: derivatives not finite, frozen")
                done[k] = True
            elif alpha == 0.0:
                print(f"  restart {k}: no improving step found, frozen")
                done[k] = True

        if step % log_every == 0 or step == n_steps - 1:
            print(
                f"step {step:5d}: min {loss_best.min().item():.3e}"
                f"  median {np.median(loss_history[step]):.3e}"
                f"  below tol {(loss_best < tol).sum().item()}/{n_restarts}"
                f"  active {int((~done).sum())}/{n_restarts}"
            )

        if done.all():
            loss_history[step + 1 :] = loss_history[step]
            print(f"all restarts settled after {step + 1} steps")
            break

    if hess_every:
        print_hessian(
            get_loss, c_best[hess_index],
            label="  best point Hessian: ", loss=loss_best[hess_index].item(),
        )

    return c_best, loss_best, loss_history


def dual_report(label, C_dual, c_targ, get_loss, deg_tol):
    """Print spectrum, degeneracy, duality and rank read-outs for one group.

    Ranks come from both the loss Hessian and the spectrum Jacobian; where they
    disagree trust the Jacobian (the Hessian's rank cut is far looser). Returns
    the data main's figures need, or None when C_dual is empty.
    """
    print(f"\n{label}: {C_dual.shape[0]} isospectral points")
    if C_dual.shape[0] == 0:
        return None

    print("coefficients:")
    print(C_dual)

    spec_targ_t = get_spectrum(c_targ)
    spec_dual_t = get_spectrum(tr.tensor(C_dual))
    print("max spectrum deviation per point:")
    print((spec_dual_t - spec_targ_t).abs().max(dim=1).values.numpy())
    spec_targ = spec_targ_t[0].numpy()
    spec_dual = spec_dual_t.numpy()

    def patterns(spec):
        return (degeneracy_pattern(spec, deg_tol),
                degeneracy_pattern(np.abs(spec), deg_tol))

    targ_patterns = patterns(spec_targ)
    print(
        f"degeneracy per point (target {targ_patterns[0]}, "
        f"|target| {targ_patterns[1]}):"
    )
    for i, spec in enumerate(spec_dual):
        pats = patterns(spec)
        flag = "" if pats == targ_patterns else "   <- differs from target"
        print(
            f"  dual {i}: {pats[0]}   |.| {pats[1]}"
            f"   reflection {reflection_residual(spec):.2e}{flag}"
        )

    # zero loss is only isospectrality; this rules out the trivial maps
    print_duality(C_dual, c_targ.numpy())

    print("rank at each dual:")
    ranks_h, ranks_j, notes = [], [], []
    for i, c_dual in enumerate(C_dual):
        report = print_hessian(get_loss, c_dual, label=f"  dual {i}: ")
        rank_h = None if report is None else report[1]
        rank_j, _ = jacobian_rank(c_dual)
        if rank_j is None:
            note_j = "degenerate spectrum, Jacobian undefined"
        else:
            note_j = (
                f"fiber dimension {N_PARAMS - rank_j}"
                + ("" if rank_h == rank_j else "   <- Hessian disagrees")
            )
        print(f"    Jacobian rank {rank_j}/{spec_targ.size}   {note_j}")
        ranks_h.append(rank_h)
        ranks_j.append(rank_j)
        notes.append(
            "rank $\\nabla^2 L$: " + ("n/a" if rank_h is None else str(rank_h))
            + "\nrank $\\partial\\lambda/\\partial c$: "
            + ("n/a" if rank_j is None else str(rank_j))
        )

    return dict(
        spec_targ=spec_targ, spec_dual=spec_dual,
        ranks_h=ranks_h, ranks_j=ranks_j, notes=notes,
    )


def search_group(
    label, tag, C_init, get_loss, c_targ, run_restart, do_search=True,
    step_through=False, n_steps=60, log_every=5, tol=1e-12, deg_tol=DEGEN_TOL,
    hess_every=10, hess_restart=0, rank_every=None,
):
    """Run (or reload, via results/newton_*_{tag}.pkl) one group and report it.

    Draws nothing except a step-through run's live figures. Returns a dict of
    everything main's figures read; hess_log and rank_log are only filled by a
    batch search, not by step-through or a reload.
    """
    output_file = f"results/newton_n{n_qb}_seed{seed}_{tag}.pkl"
    print("\n" + "=" * 72)
    print(f"{label}  ->  {output_file}")
    print("=" * 72)

    hess_log, rank_log = [], []

    if do_search:
        print(f"initial loss: min {get_loss(C_init).min().item():.3e}")
        if step_through:
            C_final, losses, loss_history = run_grad_descent_sequential(
                get_loss, C_init, n_steps=n_steps, tol=tol, log_every=log_every,
                c_targ=c_targ.numpy(), deg_tol=deg_tol,
                run_restart=run_restart,
                hess_every=hess_every, hess_restart=hess_restart,
            )
        else:
            C_final, losses, loss_history = run_restart(
                get_loss, C_init, n_steps=n_steps, tol=tol, log_every=log_every,
                hess_every=hess_every, hess_index=hess_restart,
                hess_log=hess_log, rank_every=rank_every, rank_log=rank_log,
            )
        with open(output_file, "wb") as file:
            pk.dump((n_qb, c_targ, C_init.numpy(), C_final.numpy(),
                     losses.numpy(), loss_history), file)

    with open(output_file, "rb") as file:
        n_qb_saved, c_targ, C_init, C_final, losses, loss_history = pk.load(file)

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

    keep = losses < tol
    print(f"\n{keep.sum()}/{len(losses)} restarts reached loss < {tol:.0e}")
    report = dual_report(label, C_final[keep], c_targ, get_loss, deg_tol)

    # rank at each restart's best point, closing its rank-map row
    final_ranks = (
        np.array([hessian_rank(get_loss, c) for c in C_final]) if rank_log
        else None
    )

    return dict(
        label=label, c_targ=c_targ, C_dual=C_final[keep], losses=losses,
        loss_history=loss_history, hess_log=hess_log,
        hess_restart=hess_restart, rank_log=rank_log, final_ranks=final_ranks,
        report=report, drew_live=do_search and step_through, tol=tol,
        deg_tol=deg_tol,
    )


def main():
    # ---- figures: set any of these to False to skip that figure ----------
    plots = dict(
        loss=True,           # loss of every restart over its search
        loss_strips=True,    # loss decrease + Hessian spectrum strips under it
        rank_map=True,       # Hessian rank of every restart (track_ranks groups)
        dual_ranks=False,    # Hessian vs Jacobian rank at each point below tol
        null_space=False,    # Jacobian null space at target vs points below tol
        spectrum=False,      # target spectrum against every point below tol
        coefficients=False,  # coefficients of every point below tol
    )

    do_search = True
    step_through = False  # one restart at a time, drawing each as it lands
    n_restarts = 12
    n_steps = 60
    log_every = 5
    tol = 1e-12
    deg_tol = DEGEN_TOL  # eigenvalues closer than this count as one level
    hess_every = 1       # Hessian read-out period along restart hess_restart
    rank_every = 1       # rank-map sampling period (a Hessian per restart)
    hess_restart = 0

    perturb_scale = 0.1  # width of the restart cloud around the KW point
    random_scale = 1.0   # spread of the arbitrary restarts

    # backtrack = 0, step_scale = 1 is the plain iteration; damping adds
    # lambda * I to the Newton system (Levenberg)
    step_scale = 1.0
    backtrack = 0
    damping = 0.0

    h = 1.5
    J = 1
    c_targ = tr.tensor([[0, 0, h, J, 0, 0, 0, 0, h, -J, 0, 0]])

    get_loss = get_loss_factory(c_targ)
    run_restart = partial(
        run_newton, step_scale=step_scale, backtrack=backtrack, damping=damping,
    )

    spec_targ = get_spectrum(c_targ)[0].numpy()
    print_degeneracy(spec_targ, label="target", tol=deg_tol)
    print_symmetry(spec_targ, label="target", tol=deg_tol)
    rank_targ, _ = jacobian_rank(c_targ.numpy().reshape(-1))
    print(
        f"target Jacobian rank {rank_targ}/{2**n_qb}, so the duals through the "
        f"target itself form a {N_PARAMS - rank_targ}-dimensional family"
    )

    # (label, artifact tag, initial points, track_ranks)
    groups = (
        ("KW neighbourhood", f"local{perturb_scale:g}",
         lambda: perturbed_inits(c_targ, n_restarts, scale=perturb_scale), True),
        ("random restarts", f"random{random_scale:g}",
         lambda: random_inits(n_restarts, scale=random_scale), True),
    )

    results = [
        search_group(
            label, tag, make_init(), get_loss, c_targ, run_restart,
            do_search=do_search, step_through=step_through, n_steps=n_steps,
            log_every=log_every, tol=tol, deg_tol=deg_tol,
            hess_every=hess_every, hess_restart=hess_restart,
            rank_every=rank_every if (plots["rank_map"] and track_ranks) else None,
        )
        for label, tag, make_init, track_ranks in groups
    ]

    # ---- figures, toggled by plots at the top of main ----------------------
    for r in results:
        label, report = r["label"], r["report"]
        live = r["drew_live"]  # step-through already drew loss/spectrum/coefs

        if plots["loss"] and not live:
            ax = plot_loss_history(
                r["loss_history"],
                r["hess_log"] if plots["loss_strips"] else None,
                r["hess_restart"],
            )
            ax.set_title(f"{label}: {ax.get_title()}", color=INK)

        if plots["rank_map"] and r["rank_log"]:
            plot_rank_map(r["rank_log"], r["final_ranks"], r["losses"],
                          tol=r["tol"], label=label)

        if report is None:
            continue

        if plots["dual_ranks"]:
            plot_dual_ranks(report["ranks_h"], report["ranks_j"], label=label,
                            loss_tol=r["tol"])

        if plots["null_space"]:
            plot_null_space_comparison(r["C_dual"], r["c_targ"].numpy(),
                                       label=label, loss_tol=r["tol"])

        if plots["spectrum"] and not live:
            # per-column notes only fit while each dual has its own column
            plot_spectrum_views(
                report["spec_targ"], report["spec_dual"], tol=r["deg_tol"],
                col_notes=(report["notes"] if r["C_dual"].shape[0] <= MAX_SERIES
                           else None),
            )

        if plots["coefficients"] and not live:
            ax = plot_dual_coefficients(r["C_dual"], r["c_targ"].numpy(),
                                        loss_tol=r["tol"])
            ax.set_title(f"{label}: {ax.get_title()}", color=INK, fontsize=10)

    plt.show()


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()
