"""Is an isospectral point a true dual, in the sense of Cotler et al.?

Two Hamiltonians are dual when they are isospectral but not related by local
(single-qubit) unitaries, qubit permutations or transposition, in any
composition. A mismatch in the local invariants certifies duality; otherwise
the equivalence is searched for numerically, which is evidence, not proof.
"""

import itertools as it
from dataclasses import dataclass

import numpy as np
import torch as tr

from ising_finder import get_Hamiltonian, n_qb

_PAULI = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


@dataclass
class DualCheck:
    """check_dual's result. verdict is one of "not isospectral",
    "trivially equivalent" (see perm, transposed), "dual (certified)"
    (invariants differ) or "dual (none found)" (numerical evidence only).
    """

    verdict: str
    spectrum_error: float
    invariant_gap: float
    equivalence_residual: float | None = None
    perm: tuple | None = None
    transposed: bool | None = None

    @property
    def is_dual(self):
        return self.verdict.startswith("dual")


def _hamiltonian(c):
    """(2**n, 2**n) complex numpy Hamiltonian at one coefficient point."""
    c = tr.as_tensor(np.asarray(c), dtype=tr.float64).reshape(1, -1)
    return get_Hamiltonian(c)[0].numpy()


def pauli_coefficients(H, n=n_qb):
    """(4,)*n real tensor h[s] = Tr(P_s H) / 2**n; index 0 = I, 1..3 = X, Y, Z."""
    h = np.empty((4,) * n)
    for s in it.product(range(4), repeat=n):
        P = _PAULI[s[0]]
        for k in s[1:]:
            P = np.kron(P, _PAULI[k])
        h[s] = np.real(np.trace(P @ H)) / 2**n
    return h


def local_invariants(H, n=n_qb):
    """Dict keyed by support size k: sorted norms of every support-k Pauli block,
    plus "sv2", the sorted singular values of every two-site block.
    """
    h = pauli_coefficients(H, n)
    out = {}
    for k in range(1, n + 1):
        norms, svals = [], []
        for S in it.combinations(range(n), k):
            block = h[tuple(slice(1, 4) if q in S else 0 for q in range(n))]
            norms.append(np.linalg.norm(block))
            if k == 2:
                svals.extend(np.linalg.svd(block, compute_uv=False))
        out[k] = np.sort(norms)
        if k == 2:
            out["sv2"] = np.sort(svals)
    return out


def invariant_gap(H, H_targ, n=n_qb):
    """Largest difference between the local invariants of two Hamiltonians."""
    a, b = local_invariants(H, n), local_invariants(H_targ, n)
    return max(np.abs(a[k] - b[k]).max() for k in a)


def _permute_qubits(H, perm, n=n_qb):
    """H with qubit q moved to position perm[q]."""
    axes = list(perm) + [n + p for p in perm]
    return H.reshape((2,) * (2 * n)).transpose(axes).reshape(2**n, 2**n)


def _local_unitary(theta, n=n_qb):
    """Product of single-qubit SU(2)s, exp(-i theta_q . sigma / 2) on qubit q."""
    sig = tr.tensor(np.stack(_PAULI[1:]))
    U = None
    for t in theta[:n]:
        a = tr.sqrt((t**2).sum() + 1e-30)
        u = (tr.cos(a / 2) * tr.eye(2, dtype=tr.complex128)
             - 1j * tr.sin(a / 2) * tr.einsum("k,kab->ab", (t / a).to(tr.complex128), sig))
        U = u if U is None else tr.kron(U, u)
    return U


def _best_local_fit(G, H_targ, n=n_qb, n_restarts=16, rng=None):
    """min over local U of ||U G U^dag - H_targ||_F / ||H_targ||_F."""
    rng = np.random.default_rng(0) if rng is None else rng
    G_t, T_t = tr.tensor(G), tr.tensor(H_targ)
    scale = tr.linalg.norm(T_t)

    def residual(theta):
        U = _local_unitary(theta, n)
        return tr.linalg.norm(U @ G_t @ U.conj().T - T_t) / scale

    best = np.inf
    for _ in range(n_restarts):
        theta = tr.tensor(rng.uniform(-np.pi, np.pi, size=(n, 3)), requires_grad=True)
        opt = tr.optim.LBFGS([theta], max_iter=200, tolerance_grad=1e-14,
                             tolerance_change=1e-16, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            r = residual(theta)
            r.backward()
            return r

        opt.step(closure)
        with tr.no_grad():
            best = min(best, residual(theta).item())
    return best


def check_dual(c, c_targ, spec_tol=1e-5, equiv_tol=1e-5, n_restarts=16, seed=0):
    """Decide whether point c is a true dual of c_targ; returns a DualCheck.

    spec_tol bounds the eigenvalue difference; equiv_tol bounds both the
    invariant gap and the relative fit residual for "trivially equivalent".
    """
    H, H_targ = _hamiltonian(c), _hamiltonian(c_targ)
    spec_err = float(np.abs(np.linalg.eigvalsh(H) - np.linalg.eigvalsh(H_targ)).max())
    gap = float(invariant_gap(H, H_targ))

    if spec_err > spec_tol:
        return DualCheck("not isospectral", spec_err, gap)
    if gap > equiv_tol:
        return DualCheck("dual (certified)", spec_err, gap)

    # matching invariants are only necessary, so search for the equivalence
    rng = np.random.default_rng(seed)
    best = np.inf
    for perm in it.permutations(range(n_qb)):
        for transposed in (False, True):
            G = _permute_qubits(H.T if transposed else H, perm)
            r = _best_local_fit(G, H_targ, n_restarts=n_restarts, rng=rng)
            if r < equiv_tol:
                return DualCheck("trivially equivalent", spec_err, gap,
                                 r, perm, transposed)
            best = min(best, r)
    return DualCheck("dual (none found)", spec_err, gap, best)


def print_duality(C, c_targ, **kwargs):
    """Print check_dual's verdict for every row of C; returns the DualChecks."""
    print("duality against the target (Cotler et al. trivial equivalences):")
    checks = []
    for i, c in enumerate(C):
        r = check_dual(c, c_targ, **kwargs)
        checks.append(r)
        extra = ""
        if r.verdict == "trivially equivalent":
            extra = f"   perm {r.perm}, transposed {r.transposed}"
        elif r.equivalence_residual is not None:
            extra = f"   best fit {r.equivalence_residual:.1e}"
        print(
            f"  point {i}: {r.verdict:<21}  spectrum err {r.spectrum_error:.1e}"
            f"   invariant gap {r.invariant_gap:.1e}{extra}"
        )
    return checks
