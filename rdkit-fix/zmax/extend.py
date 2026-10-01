"""Combine exhaustive (n<=16) and searched (17<=n<=24) Z_max, fit, and extrapolate to n=40."""
import math
import numpy as np

EXACT = {5:26,6:51,7:100,8:209,9:388,10:780,11:1482,12:2921,13:5600,14:11032,15:21482,16:42439}
SEARCH = {17:81725,18:159310,19:306152,20:598516,21:1154562,22:2268536,23:4445701,24:8659688}

BLK = math.log2(209)          # K4,4
print(f"{'n':>4} {'Z_max':>12} {'source':>10} {'log2 Z':>9} {'/n':>8} {'n mod 8':>8}")
allz = {**EXACT, **SEARCH}
for n in sorted(allz):
    z = allz[n]; src = "exhaustive" if n in EXACT else "search"
    print(f"{n:4d} {z:12,d} {src:>10} {math.log2(z):9.4f} {math.log2(z)/n:8.5f} {n%8:8d}")

# block-and-link model: q = n/8 blocks, q-1 links
d = 2*BLK - math.log2(42439)
print(f"\nblock-and-link model (fitted on n=8,16 only): link cost delta = {d:.5f} bits")
print(f"{'n':>4} {'actual':>9} {'model':>9} {'residual':>9}")
for n in sorted(allz):
    q = n / 8.0
    model = q * BLK - (q - 1) * d
    print(f"{n:4d} {math.log2(allz[n]):9.4f} {model:9.4f} {model - math.log2(allz[n]):+9.4f}")

ns = np.array(sorted(allz)); lz = np.array([math.log2(allz[n]) for n in ns])
tail = ns >= 12
c, b = np.polyfit(ns[tail], lz[tail], 1)
res = lz[tail] - (c*ns[tail] + b)
print(f"\nlinear fit on n>=12:  log2 Zmax(n) = {c:.5f} n {b:+.5f}")
print(f"   max |residual| = {np.abs(res).max():.4f} bits over n=12..24")

_f=[0,1]
def log2fib(k):
    while len(_f)<=k: _f.append(_f[-1]+_f[-2])
    return math.log2(_f[k])

print(f"\nEXTRAPOLATION to n=40 (and the dIpc ceiling = Z_max over the n-alkane)")
print(f"{'n':>4} {'log2 Zmax':>11} {'Z_max':>14} {'log2 Z_alkane':>14} {'dIpc ceiling':>13}")
for n in (24, 28, 30, 32, 36, 40):
    v = math.log2(allz[n]) if n in allz else c*n + b
    zz = 2**v
    print(f"{n:4d} {v:11.4f} {zz:14.4e} {log2fib(n+1):14.4f} {v-log2fib(n+1):13.4f}")
print(f"\n   asymptotic density: fit {c:.5f}, K4,4 block bound {BLK/8:.5f} bits/atom")
print(f"   (the fit approaches the block bound from below, as linking cost amortises)")
