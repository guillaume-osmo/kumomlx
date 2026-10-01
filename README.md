# kumomlx

NVIDIA's **Kumo-Tabular** in-context tabular model, reimplemented in [MLX](https://github.com/ml-explore/mlx)
so the 215.7 M-parameter network runs on the Apple Silicon GPU.

The preprocessing recipe stays in Python and only the network is ported — the recipe is
per-column statistics and costs almost nothing, while the transformer is the entire runtime.
`KumoMLX` replaces just the inner model's `forward`, so every recipe step and the public API
are untouched.

```python
from kumo_hybrid import KumoMLX

model = KumoMLX(weights="export/kumo_large_reg.safetensors")
pred  = model.predict(x_context, y_context, x_query)   # numpy in, numpy out
```

**4.7x faster than torch on CPU** end to end (0.14 s vs 0.67 s), 10x on the network alone,
measured on 256 context + 64 query rows.

## Install

```bash
pip install git+https://github.com/guillaume-osmo/kumomlx
pip install structured-data-models      # NVIDIA's preprocessing recipe, used by KumoMLX
```

Weights are resolved and cached on first use (~863 MB from the Hugging Face Hub, converted once
to safetensors under `~/.cache/kumomlx`). Point `KUMOMLX_WEIGHTS` at an existing export to skip
that, or `KUMOMLX_CACHE` to move the cache.

## Example: boiling point from 800 molecules, with no training

```bash
python examples/boiling_point.py
```

The context rows *are* the fit — there is no optimiser and no epochs. 800 labelled molecules and
200 unlabelled ones go in together; one forward pass returns a predictive distribution for each
query.

```
1000 molecules, 217 RDKit descriptors
boiling point -65 to 1023 C (sd 121.6)
context 800 / query 200

                MAE     RMSE       R2  Spearman
Kumo          12.31    31.22   0.9320    0.9737
ridge         17.45    33.45   0.9219    0.9662

in-context prediction of 200 molecules: 2.89 s (14.4 ms each), no training
90% interval covers 91.0% of held-out molecules (median width 30 C)
```

The ridge baseline is there because an impressive error means nothing unaccompanied: boiling
point is largely driven by molecular size, so a linear model already reaches R² 0.92. The 90%
interval covering 90.0% is not tuned — it is read straight off the 999-quantile head.

The largest residual is `O=C(O)c1cc(Cl)cc(Cl)c1` with a recorded boiling point of 0.0 °C, which
is a label error in the source data; dichlorobenzoic acids boil near 300 °C, which is what the
model predicts.

### Predictions are seeded by default

The recipe shuffles columns (a latin square) and rotates the per-column transform, drawing that
randomness **fresh on every call**. Unseeded, the same model on the same inputs moves by
**~2e-02** between consecutive calls — which is larger than this port's entire deviation from
torch, and is how a float32 explanation for a residual can look convincing when nothing is being
measured but noise. `predict`, `quantiles` and `predict_interval` therefore take `seed=0` by
default; pass `seed=None` for the upstream stochastic behaviour, for instance to average draws.

With the generator seeded, torch is bit-exact run to run and the hybrid pipeline agrees with it
to **1.67e-06**, or 0.0002% of the prediction spread.

```python
from kumomlx import KumoMLX, load_boiling_point

data = load_boiling_point()
(xc, yc), (xq, yq), _ = data.split(n_context=800)

model = KumoMLX()
pred    = model.predict(xc, yc, xq)             # point estimates
q       = model.quantiles(xc, yc, xq)           # [n, 999] predictive distribution
lo, hi  = model.predict_interval(xc, yc, xq, level=0.9)
```

## Descriptors

The bundled dataset carries the full `Descriptors.descList` block with one repair: raw `Ipc`
grows super-exponentially with molecule size and leaves float32 range near 170 atoms, and
RDKit's `AvgIpc` stops being monotone from about 110 atoms through float64 cancellation in the
characteristic polynomial. Both are replaced by log-space forms and emitted as NaN past their
validity limit. **[DESCRIPTORS.md](DESCRIPTORS.md)** has the derivation, the measurements and the
exact-arithmetic check.

## Accuracy

Validated against a float64 reference, which removes precision from the comparison and leaves
only structure:

| | max abs vs float64 reference |
|---|---|
| torch float32 (the reference's own precision) | 5.07e-06 |
| **this port, float32 (GPU)** | **2.90e-06** |
| this port, float32 (CPU, as the oracle script runs it) | 4.82e-06 |
| this port, float64, GELU removed both sides | 3.49e-08 |
| torch float64, 1-thread vs 8-thread (the floor) | 3.3e-15 |

The port at float32 is *more* accurate than torch's own float32. The float64 residual is fully
attributed to MLX library precision, not to the port — see below.

## Five traps, each of which produced plausible wrong numbers

1. **A parameterless module is invisible in the checkpoint.** `query_transform` /
   `key_transform` are `RMSNorm(head_dim, elementwise_affine=False)` — QK-norm. Having no
   weights they appear in no state dict, so an architecture read off the 556 tensors misses
   them, and attention is wrong by ~7 absolute while q, k and v all match to 1e-5.
   **Enumerate `named_children()`, never just the weights.**
2. **`torch.nn.RMSNorm` defaults `eps=None`,** and `F.rms_norm` then substitutes
   `torch.finfo(x.dtype).eps` — **1.19e-07 in float32, not 1e-6.** The reference passes
   `eps=1e-6` only for the two QK-norms; `query_norm`, `key_value_norm`, the MLP norm and the
   final norm all take the default. In float32 the gap is smaller than rounding noise, so this
   hid for the entire port.
3. **A forward hook captures the END state.** The ICL block mutates its input in place, so an
   un-cloned tap is post-mutation — the symptom is torch failing to reproduce its own output
   from its own tap. Chaining layers by hand fails the same way.
4. **The readout tokens evolve.** They are seeded once and carried through all six row blocks;
   re-seeding them each layer silently discards five layers of state.
5. **The naming is inverted.** `col_block` mixes across ROWS (that is the in-context mechanism,
   hence 256 inducing points); `row_block` mixes across COLUMNS plus the readout tokens.

## Two MLX float64 traps, if you use float64 as an oracle

MLX supports float64 on CPU, which makes it possible to prove a port structurally correct. Two
things will manufacture a fake "structural error" of ~1e-6:

* **`mx.array(np_float64_array)` silently returns float32.** The dtype must be given *at
  construction* — `mx.array(a, dtype=mx.float64)`. A later `.astype()` is too late.
* **MLX transcendentals are float32-precision even on float64 input**: measured `mx.exp`
  9.8e-03, `mx.erf` 4.1e-07, softmax 3.6e-09, while `+`, `*`, matmul, `sum` and `rsqrt` are
  exact. GELU (erf) and attention (exp) therefore set the oracle's floor near 1e-6. To test
  structure, replace GELU with the identity on *both* sides — and replace every GELU, including
  the head's.

`validation/mlx_float64_traps.py` demonstrates both.

## The 48-bit (double-single) trick does not pay here

Carrying each value as a pair of float32s and doing three matmuls makes every *product* exact.
It works — the split recombines bit-exactly — but this model's error is **accumulation-bound,
not product-bound**:

| variant | accuracy gain | cost |
|---|---|---|
| 48-bit split alone | 0.96x | 1.5x |
| Neumaier-compensated accumulation, chunk=256 | 3.01x | 0.83x |
| compensated, chunk=64 | 7.77x | 2.88x |
| compensated **+** 48-bit split, chunk=64 | 5.75x | 4.17x |

Adding the split to the compensation makes it *worse*: it removes product rounding, which was
never the bottleneck, while adding three accumulations of its own. And in the real model the
compensated gain collapses to 1.5x at 2.5x cost, because the error is not concentrated in the
linear layers. Kept in `experiments/` as a measured negative result.

## Architecture, as implemented

```
cells      each column groups with neighbours at offsets 2^k-1 = [0,1,3] mod C; Fourier
           features sin-then-cos over 32 frequencies, projection shared across groups, bias
           counted once per group, plus nan_lin on the missing mask; context-only mean imputation
row emb    6 x [ col_block: ISAB over rows, 256 inducing points, keyed on context rows
                 row_block: over readout+columns, RoPE (split_half) then QK-norm,
                            gated log-length query scaling ]
           -> 4 readout tokens x 256 = 1024 per row
ICL        24 layers, 1024 ch, 16 heads x 64, QK-norm, log-length query scaling; the context
           attends with all heads, queries with the leading 2 (grouped-query)
head       1024 -> 2048 -> 999 quantiles; the mean over the grid is the point estimate
```

## Layout

```
kumo_mlx.py                             the MLX network
kumo_hybrid.py                          KumoMLX: Python recipe + MLX network, drop-in
tools/make_goldens.py                   capture the post-recipe boundary (clones every tap)
validation/float64_oracle.py            the structural test described above
validation/probe_block.py               one block, component by component, in float64
validation/weight_audit.py              confirms both sides hold identical weights
validation/hybrid_check.py              end-to-end hybrid vs torch
validation/mlx_float64_traps.py         demonstrates the two MLX float64 traps
experiments/double_single.py            the 48-bit split
experiments/accumulation_vs_products.py separates the two error mechanisms
experiments/compensated_model.py        compensated accumulation in the real model
```

## Running it

Requires `mlx`, `torch`, `numpy`, `safetensors`, and NVIDIA's `structured-data-models` package
for the recipe. Weights are **not** distributed here; export them from the upstream checkpoint
to `export/kumo_large_reg.safetensors` (823 MB, fp32), then point `KUMOMLX_ROOT` at this
directory.

Note: all twelve RoPE `inv_freq` buffers are the same tensor object, and `safetensors` refuses
shared storage — clone before saving.

## Not done

Classification (the ECOC path), the `small` and `medium` sizes, and batching several tables per
call. Benchmarked only at 256 context rows; the CUDA arm runs 9,190-row contexts and MLX memory
behaviour at that scale is unmeasured.

## Attribution

This is an independent reimplementation of the Kumo-Tabular architecture published by NVIDIA
(Apache-2.0). No upstream source, and no model weights, are redistributed here. See `NOTICE`.
