"""Is an isospectral point a true dual, in the sense of Cotler et al.?

Locality from the Spectrum (Cotler, Penington, Ranard) counts two Hamiltonians
as dual when their spectra agree but they are not related by the trivial
equivalences: a product of single-qubit unitaries, a permutation of the
qubits, and transposition, in any composition. A search endpoint with zero
loss has only passed the first test; check_dual runs the second.

The second test is done two ways, strongest first:

  1. Local invariants. Write H in the Pauli basis and group the coefficients
     by support S, the set of qubits a string acts non-trivially on. A local
     unitary rotates each qubit's (X, Y, Z) index by an SO(3), transposition
     flips the sign of every Y, and a permutation relabels the supports - so
     the norm of each support block, and the singular values of each
     two-site block, are unchanged up to that relabelling. If those multisets
     differ, no trivial equivalence exists and the point is a certified dual.
  2. Search. Matching invariants are necessary, not sufficient, so the rest
     is numerical: for every qubit permutation, with and without
     transposition, minimise ||U G U^dag - H_targ|| over products U of
     single-qubit unitaries. Finding a residual below tolerance exhibits the
     equivalence; not finding one is evidence, not proof, and the verdict
     says so.
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
    """What check_dual found for one point.

    verdict is one of
      "not isospectral"      - the spectra differ by more than spec_tol
      "trivially equivalent" - a local unitary, permutation and transpose
                               maps it onto the target; see perm, transposed
      "dual (certified)"     - local invariants differ, so no such map exists
      "dual (none found)"    - invariants agree and the search found no map;
                               numerical evidence only
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
    """(4,)*n real tensor h[s_1..s_n] = Tr(P_s H) / 2**n, site 0 first.

    Index 0 is the identity on that site and 1, 2, 3 are X, Y, Z, in the same
    kron order get_Hamiltonian builds H with.
    """
    h = np.empty((4,) * n)
    for s in it.product(range(4), repeat=n):
        P = _PAULI[s[0]]
        for k in s[1:]:
            P = np.kron(P, _PAULI[k])
        h[s] = np.real(np.trace(P @ H)) / 2**n
    return h


def local_invariants(H, n=n_qb):
    """Sorted invariants of H under local unitaries, permutations and transpose.

    Returns a dict keyed by support size k: the sorted norms of every
    support-k block, and for k == 2 also the sorted singular values of every
    two-site block. Both are exact invariants of the whole trivial group, so
    any mismatch rules out a trivial equivalence.
    """
    h = pauli_coefficients(H, n)
    out = {}
    for k in range(1, n + 1):
        norms, svals = [], []
        for S in it.combinations(range(n), k):
            # identity off S, non-identity (1..3) on S
            idx = tuple(slice(1, 4) if q in S else 0 for q in range(n))
            block = h[idx]
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
    T = H.reshape((2,) * (2 * n))
    axes = list(perm) + [n + p for p in perm]
    return T.transpose(axes).reshape(2**n, 2**n)


def _local_unitary(theta, n=n_qb):
    """Product of single-qubit SU(2)s, exp(-i theta_q . sigma / 2) on qubit q."""
    sig = tr.tensor(np.stack(_PAULI[1:]))
    U = None
    for q in range(n):
        t = theta[q]
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
    best = np.inf
    for _ in range(n_restarts):
        theta = tr.tensor(rng.uniform(-np.pi, np.pi, size=(n, 3)), requires_grad=True)
        opt = tr.optim.LBFGS([theta], max_iter=200, tolerance_grad=1e-14,
                             tolerance_change=1e-16, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            U = _local_unitary(theta, n)
            r = tr.linalg.norm(U @ G_t @ U.conj().T - T_t) / scale
            r.backward()
            return r

        opt.step(closure)
        with tr.no_grad():
            U = _local_unitary(theta, n)
            r = (tr.linalg.norm(U @ G_t @ U.conj().T - T_t) / scale).item()
        best = min(best, r)
    return best


def check_dual(c, c_targ, spec_tol=1e-5, equiv_tol=1e-5, n_restarts=16, seed=0):
    """Decide whether point c is a true dual of c_targ; returns a DualCheck.

    spec_tol bounds the largest eigenvalue difference for "isospectral".
    equiv_tol is how close (relative Frobenius distance, and absolute gap in
    the local invariants) c must come to a trivial image of c_targ to count as
    equivalent to it. A point within equiv_tol of an equivalent one is called
    equivalent, even if it sits on a genuine dual family through it.
    """
    H, H_targ = _hamiltonian(c), _hamiltonian(c_targ)
    lam, lam_t = np.linalg.eigvalsh(H), np.linalg.eigvalsh(H_targ)
    spec_err = float(np.abs(lam - lam_t).max())
    gap = float(invariant_gap(H, H_targ))

    if spec_err > spec_tol:
        return DualCheck("not isospectral", spec_err, gap)
    if gap > equiv_tol:
        return DualCheck("dual (certified)", spec_err, gap)

    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for perm in it.permutations(range(n_qb)):
        for transposed in (False, True):
            G = _permute_qubits(H.T if transposed else H, perm)
            r = _best_local_fit(G, H_targ, n_restarts=n_restarts, rng=rng)
            if r < best[0]:
                best = (r, perm, transposed)
            if r < equiv_tol:
                return DualCheck("trivially equivalent", spec_err, gap,
                                 r, perm, transposed)
    return DualCheck("dual (none found)", spec_err, gap, best[0])


def print_duality(C, c_targ, **kwargs):
    """Print check_dual's verdict for every row of C; returns the DualChecks."""
    print("duality against the target (Cotler et al. trivial equivalences):")
    checks = []
    for i, c in enumerate(C):
        r = check_dual(c, c_targ, **kwargs)
        extra = ""
        if r.verdict == "trivially equivalent":
            extra = f"   perm {r.perm}, transposed {r.transposed}"
        elif r.equivalence_residual is not None:
            extra = f"   best fit {r.equivalence_residual:.1e}"
        print(
            f"  point {i}: {r.verdict:<21}  spectrum err {r.spectrum_error:.1e}"
            f"   invariant gap {r.invariant_gap:.1e}{extra}"
        )
        checks.append(r)
    return checks
