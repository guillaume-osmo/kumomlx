"""Exact Z for a chain of K4,4 blocks linked WITHOUT violating the degree bound.

Previous attempt was invalid: K4,4 is already 4-regular, so adding a link edge pushed the
connectors to degree 5. The tell was that the "construction" exceeded the exhaustive maximum at
n=16 (15.58 vs 15.37), which is impossible.

Valid version: delete one edge (l, r) inside each block first, which drops both endpoints to
degree 3, then use l and r as the connectors. Every vertex stays at degree <= 4.
"""
import functools, math
import networkx as nx


def block_counts(G, l, r):
    edges = tuple(sorted(tuple(sorted(e)) for e in G.edges()))
    @functools.lru_cache(maxsize=None)
    def rec(es, banned):
        if not es: return 1
        (u, v), rest = es[0], es[1:]
        tot = rec(rest, banned)
        if u not in banned and v not in banned:
            tot += rec(tuple(e for e in rest if u not in e and v not in e), banned)
        return tot
    return (rec(edges, frozenset()), rec(edges, frozenset([l])),
            rec(edges, frozenset([r])), rec(edges, frozenset([l, r])))


def chain_Z(blocks):
    N, N_l, N_r, N_lr = blocks[0]
    u, m = N_r, N - N_r
    for (N, N_l, N_r, N_lr) in blocks[1:]:
        u, m = (u * N_lr + (u + m) * N_r,
                u * (N_l - N_lr) + (u + m) * (N - N_r))
    return u + m


# K4,4 minus the edge (0,4): vertices 0 and 4 drop to degree 3 and become the connectors.
G = nx.complete_bipartite_graph(4, 4)
G.remove_edge(0, 4)
c44 = block_counts(G, 0, 4)
print(f"K4,4 minus one edge, connectors at degree 3: (N, N_l, N_r, N_lr) = {c44}")
print(f"  (intact K4,4 has N=209; removing an edge costs {209 - c44[0]} matchings)")

FIT_C, FIT_B = 0.96092, -0.02313
EXACT16, SEARCH24 = 15.3731, 23.0459

print(f"\nSANITY: the construction must not exceed the exhaustive / searched maxima")
for q, n, ref, lab in ((2, 16, EXACT16, "exhaustive"), (3, 24, SEARCH24, "search")):
    lz = math.log2(chain_Z([c44] * q))
    ok = "OK" if lz <= ref + 1e-9 else "IMPOSSIBLE"
    print(f"  n={n}: chain {lz:.4f}  vs {lab} {ref:.4f}  -> {ok}")

_f = [0, 1]
def log2fib(k):
    while len(_f) <= k: _f.append(_f[-1] + _f[-2])
    return math.log2(_f[k])

print(f"\n{'n':>5} {'chain (valid LB)':>17} {'fit':>9} {'fit - chain':>12} {'per atom':>9}")
for q in (2, 3, 4, 5, 6, 8, 12, 31):
    n = 8 * q
    lz = math.log2(chain_Z([c44] * q))
    print(f"{n:5d} {lz:17.4f} {FIT_C*n+FIT_B:9.4f} {FIT_C*n+FIT_B-lz:+12.4f} {lz/n:9.5f}")

print(f"\nThe two sizes asked for (chain rounded to the nearest multiple of 8 below):")
for n in (30, 50):
    q = n // 8
    lz = math.log2(chain_Z([c44] * q))
    fit = FIT_C * n + FIT_B
    print(f"  n={n:3d}  fit {fit:8.4f} bits -> Z_max ~ {2**fit:.4e}")
    print(f"        (a {q}-block chain on {8*q} atoms already reaches {lz:.4f}, "
          f"so the fit is not obviously low)")
    print(f"        alkane {log2fib(n+1):.4f} -> dIpc ceiling {fit - log2fib(n+1):.3f} bits")
