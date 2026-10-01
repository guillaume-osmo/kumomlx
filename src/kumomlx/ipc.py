"""An Ipc that does not break: exact integer characteristic polynomial, entropy in log space.

RDKit's ``Ipc`` and ``AvgIpc`` both fail on large molecules, for two different reasons:

* ``Ipc = S * H`` is extensive and ``S`` grows super-exponentially, so it leaves float32 range
  near 170 atoms;
* ``AvgIpc = H`` is bounded by ``log2(n+1)`` in theory, but RDKit computes the characteristic
  polynomial in float64, where the alternating-sign Faddeev-LeVerrier recursion suffers
  catastrophic cancellation. Measured against exact arithmetic, RDKit's ``AvgIpc`` stops being
  monotone in ``n`` from about 110 atoms (3.775 at 105, then 3.577, 2.891, 1.401 ...).

Fixing the *representation* -- using the intensive ``H`` and the log-space extensive term -- does
not help while the coefficients themselves are wrong. The real fix is to stop using floating
point for them at all.

The adjacency matrix is integer, so the characteristic polynomial has INTEGER coefficients. They
are computed here exactly, by running Faddeev-LeVerrier modulo several small primes and
reconstructing with the Chinese remainder theorem. The primes are chosen small enough that
``n * (p-1)**2 < 2**53``, which keeps every modular matrix product exactly representable in
float64 -- so the inner loop is a BLAS matmul rather than Python big-integer arithmetic, and the
whole thing stays fast while remaining exact.

The entropy is then evaluated in log space (log-sum-exp), so ``S`` is never formed and nothing
overflows at any molecular size.

    from kumomlx.ipc import ipc_terms
    H, log2_ipc = ipc_terms(mol)        # exact, monotone, overflow-free
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

__all__ = ["charpoly_exact", "ipc_terms", "avg_ipc", "log2_ipc"]


def _primes_below(limit: int, count: int) -> list[int]:
    """The `count` largest primes below `limit`, by trial division (limit is ~4e6, so this is
    cheap and avoids a sieve of that size)."""
    out: list[int] = []
    n = limit | 1
    while len(out) < count and n > 2:
        if all(n % d for d in range(3, int(n ** 0.5) + 1, 2)) and n % 2:
            out.append(n)
        n -= 2
    return out


def _coeff_bits(n: int) -> float:
    """An upper bound on log2 |c_i|, over all i.

    ``c_i`` is a signed sum of the ``C(n, i)`` principal ``i x i`` minors, and Hadamard bounds
    each 0/1 minor by ``i**(i/2)``. The bound is loose but cheap, and only sets how many primes
    are needed.
    """
    best = 1.0
    for i in range(1, n + 1):
        bits = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        bits = bits / math.log(2) + 0.5 * i * math.log2(i if i > 1 else 2)
        best = max(best, bits)
    return best


def _charpoly_mod(A: np.ndarray, p: int) -> list[int]:
    """Faddeev-LeVerrier modulo p.

    Entries stay below p and the products are accumulated in float64, which is exact because the
    prime was chosen so that ``n * (p-1)**2 < 2**53``.
    """
    n = A.shape[0]
    Ap = np.mod(A, p).astype(np.float64)
    M = np.eye(n, dtype=np.float64)
    coeffs = [1]
    for k in range(1, n + 1):
        AM = np.mod(Ap @ M, p)
        c = int((-int(round(AM.trace())) % p) * pow(k, p - 2, p) % p)
        coeffs.append(c)
        M = np.mod(AM + c * np.eye(n), p)
    return coeffs


def _crt(residues: list[int], moduli: list[int]) -> int:
    """Chinese remainder reconstruction, returned on the symmetric range so signs survive."""
    x, m = 0, 1
    for r, p in zip(residues, moduli):
        # solve x + m*t == r (mod p)
        t = ((r - x) % p) * pow(m % p, p - 2, p) % p
        x += m * t
        m *= p
    return x - m if x > m // 2 else x


def charpoly_exact(A: np.ndarray) -> list[int]:
    """Exact integer coefficients of ``det(xI - A)``, highest power first.

    Args:
        A: a square integer matrix (here, a molecular adjacency matrix).

    Returns:
        ``n + 1`` Python ints, exactly. No floating point error at any size.
    """
    n = A.shape[0]
    if n == 0:
        return [1]
    # n*(p-1)^2 < 2^53 keeps every modular matmul exact in float64.
    limit = min(1 << 22, int(math.sqrt(2 ** 53 / max(n, 1))))
    need = int(_coeff_bits(n) / math.log2(limit)) + 2
    primes = _primes_below(limit, need)
    if len(primes) < need:                       # pathologically large n; fall back to exactness
        raise ValueError(f"matrix too large for the modular scheme (n={n})")
    per_prime = [_charpoly_mod(A, p) for p in primes]
    return [_crt([r[i] for r in per_prime], primes) for i in range(n + 1)]


def _entropy_log_space(coeffs) -> tuple[float, float]:
    """``(H, log2 S)`` from exact integer coefficients, without ever forming ``S``."""
    a = [abs(int(c)) for c in coeffs if c]
    if not a:
        return math.nan, math.nan
    # math.log2 of a Python int is exact-ish for arbitrarily large ints, so this cannot overflow.
    L = [math.log2(v) for v in a]
    m = max(L)
    log2_S = m + math.log2(sum(2.0 ** (li - m) for li in L))
    H = sum(2.0 ** (li - log2_S) * (log2_S - li) for li in L)
    return H, log2_S


def ipc_terms(mol) -> tuple[float, float]:
    """``(AvgIpc, log2(Ipc))`` for an RDKit molecule, exactly and at any size.

    ``AvgIpc`` is the intensive information content ``H``, bounded by ``log2(n_atoms + 1)``.
    ``log2(Ipc) = log2(S) + log2(H)`` is the extensive term, which grows roughly linearly in the
    atom count instead of super-exponentially.
    """
    from rdkit import Chem                        # noqa: PLC0415

    A = Chem.GetAdjacencyMatrix(mol)
    H, log2_S = _entropy_log_space(charpoly_exact(np.asarray(A, dtype=np.int64)))
    if not (H > 0 and math.isfinite(H) and math.isfinite(log2_S)):
        return math.nan, math.nan
    return H, log2_S + math.log2(H)


def avg_ipc(mol) -> float:
    """The intensive half: Shannon entropy of the coefficient distribution, in bits."""
    return ipc_terms(mol)[0]


def log2_ipc(mol) -> float:
    """The extensive half, in log space."""
    return ipc_terms(mol)[1]
