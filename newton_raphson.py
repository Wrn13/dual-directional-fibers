"""Newton-Raphson search for dual tensor product structures.

The same search as grad_descent.py - restarts over the 12-dimensional Ising
coefficient space, looking for points that are isospectral with a target
Hamiltonian - stepped with Newton-Raphson instead of Adam. Everything around
the optimizer (the live figures, the degeneracy read-out, the Hessian
read-out, the saved artifact) is imported from grad_descent, so the two
scripts differ only in how a restart moves.

Newton-Raphson here is the root finder applied to the stationarity condition
grad L = 0: solve H dc = -grad L and step. H is the loss Hessian, which is
singular at any solution - the duals come in continuous families, and the
loss stops curving along them - so the step comes from a least-squares solve
rather than an inverse. That picks the minimum-norm step and leaves the flat
subspace where it is, instead of throwing the iterate to infinity along it.

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
    jacobian_rank,
    log_hessian,
    perturbed_inits,
    plot_dual_coefficients,
    plot_dual_ranks,
    plot_loss_history,
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

# singular values below this fraction of the largest are treated as zero when
# the Newton system is solved. The Hessian is exactly rank deficient at a dual,
# so this is what keeps the step finite there rather than an accident of
# conditioning.
NEWTON_RCOND = 1e-10


def newton_direction(H, g, rcond=NEWTON_RCOND, damping=0.0):
    """Minimum-norm solution dc of (H + damping I) dc = -g.

    H is (12, 12) and g is (12,), both numpy. Least squares rather than solve:
    H is singular wherever the loss has flat directions, and lstsq drops those
    directions from the step instead of dividing by their zero curvature.
    """
    A = 0.5 * (np.asarray(H) + np.asarray(H).T)
    if damping:
        A = A + damping * np.eye(A.shape[0])
    dc, *_ = np.linalg.lstsq(A, -np.asarray(g), rcond=rcond)
    return dc


def newton_iterate(
    get_loss, c, rcond=NEWTON_RCOND, damping=0.0, step_scale=1.0, backtrack=0
):
    """One Newton-Raphson step from a single point c (12,).

    Returns (c_next, loss_here, alpha), where alpha is the fraction of the
    Newton step actually taken. With backtrack = 0 that is always step_scale,
    the plain iteration; with backtrack = m the step is halved up to m times
    while it would raise the loss, which keeps a restart from being thrown
    across the space by the indefinite Hessian that sits between the basins.
    alpha comes back 0 where no tried step improved on where it stands, and
    nan where the derivatives were not finite.
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
    hess_every=None, hess_index=0, hess_log=None,
):
    """Newton-Raphson every restart, one row at a time per step.

    Signature and return values match run_grad_descent, so this drops straight
    into run_grad_descent_sequential as its run_restart: c_init is (K, 12) and
    it returns (c_best, loss_best, loss_history) with loss_history (n_steps, K).

    Rows are stepped in a loop rather than as a batch because each one needs
    its own 12x12 Hessian factorization; at these sizes that costs a few
    spectrum evaluations per row per step. A row that reaches tol, or whose
    derivatives stop being finite, or that cannot improve on where it stands,
    is frozen and its loss carried forward through the rest of the history, so
    the curves stay the same shape as the gradient-descent ones.

    With hess_every set, the Hessian eigenvalues and numerical rank of restart
    hess_index are printed every hess_every steps and again at the point that
    restart finished on. Pass a list as hess_log to also collect those samples
    as (step, rank, eigenvalues, tol), which is what plot_hessian_log draws
    under the loss. Each is taken at the point the step starts from, the same
    point loss_history[step] is, so the Hessian at step t is the one the step
    from t was taken with.
    """
    c = c_init.clone().detach()
    n_restarts = c.shape[0]

    c_best = c.clone()
    loss_best = tr.full((n_restarts,), np.inf)
    loss_history = np.empty((n_steps, n_restarts))
    done = np.zeros(n_restarts, dtype=bool)

    for step in range(n_steps):
        # a frozen restart is not stepping, so there is no step to read against
        if (hess_every and not done[hess_index]
                and (step % hess_every == 0 or step == n_steps - 1)):
            log_hessian(hess_log, step, print_hessian(
                get_loss, c[hess_index],
                label=f"  step {step:5d} Hessian: ",
                loss=get_loss(c[hess_index].reshape(1, -1))[0].item(),
            ))

        for k in range(n_restarts):
            if done[k]:
                # frozen: carry its last loss forward so the curve stays flat
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
            # nothing left to move: hold the final losses for the rest of the
            # history so the saved array keeps its (n_steps, K) shape
            loss_history[step + 1 :] = loss_history[step]
            print(f"all restarts settled after {step + 1} steps")
            break

    if hess_every:
        print_hessian(
            get_loss, c_best[hess_index],
            label="  best point Hessian: ", loss=loss_best[hess_index].item(),
        )

    return c_best, loss_best, loss_history


def dual_report(
    label, C_dual, c_targ, get_loss, deg_tol, draw_spectrum=True, loss_tol=None
):
    """Print and draw the spectrum and the rank read-out for one group's duals.

    Every dual gets its spectrum checked against the target's - deviation,
    degeneracy pattern, reflection residual - and its rank measured two ways:
    from the loss Hessian, and from the spectrum Jacobian. The Hessian is
    2 J^T J at a zero-residual point, but its rank cut is far looser on the
    singular values of J, so where they disagree the Jacobian is the one to
    believe.

    draw_spectrum is off when a step-through run already built the spectrum
    and coefficient figures live; the rank figure is drawn either way, being
    the one thing the live run does not show. loss_tol is the cut C_dual was
    kept under, for the figure titles.
    """
    n_duals = C_dual.shape[0]
    print(f"\n{label}: {n_duals} isospectral points")
    if n_duals == 0:
        return

    print("coefficients:")
    print(C_dual)

    spec_targ_t = get_spectrum(c_targ)
    spec_dual_t = get_spectrum(tr.tensor(C_dual))
    print("max spectrum deviation per point:")
    print((spec_dual_t - spec_targ_t).abs().max(dim=1).values.numpy())

    spec_targ = spec_targ_t[0].numpy()
    spec_dual = spec_dual_t.numpy()

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

    # the nullity counts the directions the loss does not curve along, which
    # are the directions a fiber of duals through that point can run
    print("rank at each dual:")
    ranks_h, ranks_j, notes = [], [], []
    for i, c_dual in enumerate(C_dual):
        report = print_hessian(get_loss, c_dual, label=f"  dual {i}: ")
        rank_h = None if report is None else report[1]
        rank_j, sv = jacobian_rank(c_dual)
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

    plot_dual_ranks(ranks_h, ranks_j, label=label, loss_tol=loss_tol)
    if draw_spectrum:
        # the per-column notes only fit while the duals still have their own
        # columns; past that the spectrum figure is one merged population
        plot_spectrum_views(
            spec_targ, spec_dual, tol=deg_tol,
            col_notes=notes if n_duals <= MAX_SERIES else None,
        )
        ax = plot_dual_coefficients(C_dual, c_targ.numpy(), loss_tol=loss_tol)
        ax.set_title(f"{label}: {ax.get_title()}", color=INK, fontsize=10)


def search_group(
    label, tag, C_init, get_loss, c_targ, run_restart, do_search=True,
    step_through=False, n_steps=60, log_every=5, tol=1e-12, deg_tol=DEGEN_TOL,
    hess_every=10, hess_restart=0,
):
    """Run one group of restarts end to end and report what it found.

    tag names the group's own artifact, so a random-restart run and a
    neighbourhood run do not overwrite each other.
    """
    output_file = f"results/newton_n{n_qb}_seed{seed}_{tag}.pkl"
    print("\n" + "=" * 72)
    print(f"{label}  ->  {output_file}")
    print("=" * 72)

    # the step-through run fills and draws its own; this one is for the batch
    # path, and stays empty on a reload since it is not part of the artifact
    hess_log = []

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

    keep = losses < tol
    print(f"\n{keep.sum()}/{len(losses)} restarts reached loss < {tol:.0e}")
    dual_report(
        label, C_final[keep], c_targ, get_loss, deg_tol,
        draw_spectrum=not drew_live, loss_tol=tol,
    )

    if not drew_live:
        ax = plot_loss_history(loss_history, hess_log, hess_restart)
        ax.set_title(f"{label}: {ax.get_title()}", color=INK)

    return C_final[keep]


def main():
    do_search = True
    # one restart at a time, adding each curve as it finishes. Off by default
    # now that a run covers two groups, since stepping through both of them is
    # 2 * n_restarts pauses; turn it on to watch a single group land.
    step_through = False

    n_restarts = 12
    # Newton converges quadratically once it is near a solution, so this is a
    # far shorter run than the gradient-descent one
    n_steps = 60
    log_every = 5
    tol = 1e-12
    # how close two eigenvalues must sit to be read as one degenerate level
    deg_tol = DEGEN_TOL
    # print the Hessian eigenvalues and rank this often along one restart's
    # search; None turns the read-out off. Every step, since a Newton run is
    # only tens of steps long and the loss-decrease strip needs a Hessian
    # sample under each of them to be read against.
    hess_every = 1
    # which restart is the one followed, when stepping through them in turn
    hess_restart = 0

    # width of the restart cloud around the KW point
    perturb_scale = 0.1
    # spread of the arbitrary restarts over the whole coefficient space
    random_scale = 1.0

    # Newton knobs. backtrack = 0 with step_scale = 1 is the plain iteration,
    # which is what this runs: on this loss it reaches 1e-13 in about twenty
    # steps. Raising backtrack halves any step that would climb, which stops a
    # restart being thrown across the space by the indefinite Hessian between
    # the basins, at the cost of freezing restarts that have to climb over a
    # ridge to reach a dual. damping adds lambda*I to the Newton system, the
    # Levenberg dial from Newton towards gradient descent.
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

    # the level structure every dual has to reproduce, printed before the
    # search so the duals arriving below can be read against it
    print_degeneracy(get_spectrum(c_targ)[0].numpy(), label="target", tol=deg_tol)
    print_symmetry(get_spectrum(c_targ)[0].numpy(), label="target", tol=deg_tol)
    rank_targ, _ = jacobian_rank(c_targ.numpy().reshape(-1))
    print(
        f"target Jacobian rank {rank_targ}/{2**n_qb}, so the duals through the "
        f"target itself form a {N_PARAMS - rank_targ}-dimensional family"
    )

    # Two groups against the same target: restarts drawn from the whole
    # coefficient space, and restarts clustered on the KW point. Both descend
    # to the same spectrum, so the difference is which part of the isospectral
    # set they land on - which is what the rank read-out is there to show.
    groups = (
        (
            "KW neighbourhood",
            f"local{perturb_scale:g}",
            lambda: perturbed_inits(c_targ, n_restarts, scale=perturb_scale),
        ),
        (
            "random restarts",
            f"random{random_scale:g}",
            lambda: random_inits(n_restarts, scale=random_scale),
        ),
    )

    for label, tag, make_init in groups:
        search_group(
            label, tag, make_init(), get_loss, c_targ, run_restart,
            do_search=do_search, step_through=step_through, n_steps=n_steps,
            log_every=log_every, tol=tol, deg_tol=deg_tol,
            hess_every=hess_every, hess_restart=hess_restart,
        )

    plt.show()


if __name__ == "__main__":
    mp.rcParams["font.family"] = "serif"
    mp.rcParams["text.usetex"] = False

    main()
