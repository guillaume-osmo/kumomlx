"""Separate the two mechanisms properly: split (exact products) vs compensation (exact sums)."""
import sys, time
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from double_single import matmul48

def chunked(a, b, chunk=128, split=False):
    mm = matmul48 if split else (lambda p, q: p @ q)
    s = c = None
    for i in range(0, a.shape[-1], chunk):
        p = mm(a[..., i:i + chunk], b[..., i:i + chunk, :])
        if s is None:
            s, c = p, mx.zeros_like(p)
        else:
            t = s + p; bb = t - s
            c = c + ((s - (t - bb)) + (p - bb))
            s = t
    return s + c

rng = np.random.default_rng(0)
M, Kd, N = 320, 1024, 2048
A = rng.standard_normal((M, Kd)).astype(np.float32); B = rng.standard_normal((Kd, N)).astype(np.float32)
truth = A.astype(np.float64) @ B.astype(np.float64)
Am, Bm = mx.array(A), mx.array(B)

def bench(name, fn):
    o = fn(); mx.eval(o)
    t0 = time.perf_counter()
    for _ in range(5): o = fn(); mx.eval(o)
    dt = (time.perf_counter() - t0) / 5
    e = np.abs(np.array(o).astype(np.float64) - truth).max()
    print(f"  {name:40s} {e:.3e}   {dt*1000:6.1f} ms")
    return e, dt

print(f"{M}x{Kd} @ {Kd}x{N} vs float64 truth:")
e0, t0 = bench("plain float32", lambda: Am @ Bm)
eS, _ = bench("split only (48-bit products)", lambda: matmul48(Am, Bm))
print()
for ch in (256, 128, 64):
    eC, tC = bench(f"compensated, PLAIN products, chunk={ch}", lambda ch=ch: chunked(Am, Bm, ch, False))
    eB, tB = bench(f"compensated + split,         chunk={ch}", lambda ch=ch: chunked(Am, Bm, ch, True))
    print(f"      -> plain {e0/eC:4.2f}x at {tC/t0:4.2f}x cost | +split {e0/eB:4.2f}x at {tB/t0:4.2f}x cost")
