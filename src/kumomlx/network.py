"""MLX port of NVIDIA Kumo-Tabular (regression head), for Apple Silicon.

Ported from the Apache-2.0 reference in `sdm.models.kumo.tabular`. Built stage by stage against
activations captured from the reference model, because every component here is shape-compatible
with a wrong implementation: a mis-split qkv or a missing query scale produces plausible numbers,
not an error.

Architecture, as read from the checkpoint and the reference forward passes (size "large"):

    cell embedding   Fourier features per (row, column), 256 channels
    row embedding    6 col_blocks (set transformer over columns, 256 inducing points)
                     + 6 row_blocks (attention over rows, RoPE, gated query scaling)
                     -> 4 readout tokens x 256 = 1024 per row
    ICL block        label embedding added to context rows, then 24 layers of
                     cross-attention from all rows to the context rows, 1024 channels,
                     16 query heads of 64; the query pass uses only the first 2 KV heads
    head             1024 -> 2048 -> 999 quantiles

Two details that are easy to get wrong and are load-bearing:

* the block is pre-norm with SEPARATE norms for the query and key/value streams, and the residual
  adds the UN-normalised query: `tmp = attn(qn(q), kvn(kv)) + q; out = tmp + mlp(tmp)`;
* queries are scaled by `head_scale[h] * log(max(n_context, 1))` before attention, so the result
  depends on the context length. Dropping it leaves the shapes intact and the answers wrong.
"""

from __future__ import annotations

import math

import numpy as np

import mlx.core as mx


# torch.nn.RMSNorm defaults `eps` to None, and F.rms_norm then substitutes
# `torch.finfo(x.dtype).eps` -- NOT 1e-6. The reference passes eps=1e-6 ONLY for the two QK
# norms; `query_norm`, `key_value_norm`, the MLP norm and the final norm all take the default.
# In float32 the two values (1.19e-07 vs 1e-6) differ by less than float32 rounding noise, so
# this hid for the whole port and looked like reduction-order error.
_FINFO_EPS = {
    mx.float32: 1.1920928955078125e-07,
    mx.float64: 2.220446049250313e-16,
    mx.float16: 0.0009765625,
    mx.bfloat16: 0.0078125,
}


def _rms_norm(x: mx.array, weight: mx.array, eps: float | None = None) -> mx.array:
    if eps is None:
        eps = _FINFO_EPS[x.dtype]
    if x.dtype == mx.float32:
        return mx.fast.rms_norm(x, weight, eps)
    # mx.fast.rms_norm is float32-only; this is the same arithmetic, written out, so the
    # float64 reference path and the double-single path can reuse the whole model.
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps) * weight


def _sdpa(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
    if q.dtype == mx.float32:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    attn = (q * scale) @ mx.swapaxes(k, -1, -2)
    return mx.softmax(attn, axis=-1, precise=True) @ v


def _qk_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """RMSNorm over the head dimension, with NO learnable weight.

    `query_transform` / `key_transform` are `RMSNorm(head_dim, elementwise_affine=False)`. Having
    no parameters, they appear nowhere in the checkpoint -- so an architecture inferred from the
    weights alone misses them entirely, and the attention is then quietly wrong.
    """
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


def _linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    y = x @ weight.T
    return y if bias is None else y + bias


def _gelu(x: mx.array) -> mx.array:
    # The reference uses torch.nn.GELU() with its default, which is the exact erf form.
    return x * 0.5 * (1.0 + mx.erf(x / math.sqrt(2.0)))


class Attention:
    """Multi-head attention with a separate key/value stream and log-length query scaling."""

    def __init__(self, w: dict, prefix: str, num_heads: int, head_dim: int):
        self.qw = w[f"{prefix}.qkv_lin.weight"]
        self.qb = w.get(f"{prefix}.qkv_lin.bias")
        self.ow = w[f"{prefix}.out_lin.weight"]
        self.ob = w.get(f"{prefix}.out_lin.bias")
        self.head_scale = w.get(f"{prefix}.sdpa.query_scaling.head_scale")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_dim = num_heads * head_dim

    def __call__(self, query: mx.array, key_value: mx.array, kv_heads: int | None = None):
        q = _linear(query, self.qw[: self.q_dim], None if self.qb is None else self.qb[: self.q_dim])
        kv = _linear(key_value, self.qw[self.q_dim:], None if self.qb is None else self.qb[self.q_dim:])
        k, v = mx.split(kv, 2, axis=-1)

        n_q, n_kv = query.shape[-2], key_value.shape[-2]
        q = q.reshape(n_q, self.num_heads, self.head_dim)
        k = k.reshape(n_kv, self.num_heads, self.head_dim)
        v = v.reshape(n_kv, self.num_heads, self.head_dim)

        # QK-norm, before the query scaling that sdpa applies.
        q = _qk_norm(q)
        k = _qk_norm(k)

        if self.head_scale is not None:
            # LogScale: q_hi *= a_h * log(max(n_kv, 1)). Depends on the CONTEXT length.
            q = q * (self.head_scale * math.log(max(n_kv, 1.0)))[None, :, None]

        if kv_heads is not None and kv_heads < self.num_heads:
            # The query pass reuses only the leading KV heads, grouped-query style.
            k, v = k[:, :kv_heads], v[:, :kv_heads]
            repeat = self.num_heads // kv_heads
            k = mx.repeat(k, repeat, axis=1)
            v = mx.repeat(v, repeat, axis=1)

        q = q.transpose(1, 0, 2)[None]   # [1, H, Q, Dh]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        # the reference passes scale=None, which is torch's default 1/sqrt(head_dim)
        o = _sdpa(q, k, v, self.head_dim ** -0.5)
        o = o[0].transpose(1, 0, 2).reshape(n_q, self.q_dim)
        return _linear(o, self.ow, self.ob)


class Block:
    """Pre-norm block: attn(query_norm(q), kv_norm(kv)) + q, then + mlp(that)."""

    def __init__(self, w: dict, prefix: str, num_heads: int, head_dim: int):
        self.qn = w.get(f"{prefix}.query_norm.weight")
        self.kn = w.get(f"{prefix}.key_value_norm.weight")
        self.attn = Attention(w, f"{prefix}.attn", num_heads, head_dim)
        # mlp = [RMSNorm, Linear(d, 2d), GELU, Linear(2d, d)] -> indices 0, 1, 3
        self.mlp_norm = w[f"{prefix}.mlp.0.weight"]
        self.mlp_w1, self.mlp_b1 = w[f"{prefix}.mlp.1.weight"], w[f"{prefix}.mlp.1.bias"]
        self.mlp_w2, self.mlp_b2 = w[f"{prefix}.mlp.3.weight"], w[f"{prefix}.mlp.3.bias"]

    def __call__(self, query: mx.array, key_value: mx.array, kv_heads: int | None = None):
        q_in = query if self.qn is None else _rms_norm(query, self.qn)
        kv_in = key_value if self.kn is None else _rms_norm(key_value, self.kn)
        tmp = self.attn(q_in, kv_in, kv_heads=kv_heads) + query
        h = _rms_norm(tmp, self.mlp_norm)
        h = _linear(_gelu(_linear(h, self.mlp_w1, self.mlp_b1)), self.mlp_w2, self.mlp_b2)
        return tmp + h


class ICLBlock:
    """The in-context stack: context rows are keys/values, every row is a query."""

    def __init__(self, w: dict, prefix: str, num_layers: int = 24, num_heads: int = 16,
                 head_dim: int = 64, kv_heads_for_query: int | None = 2):
        self.y_lin = w[f"{prefix}.y_lin.weight"]
        self.y_bias = w.get(f"{prefix}.y_lin.bias")
        self.layers = [Block(w, f"{prefix}.layers.{i}", num_heads, head_dim)
                       for i in range(num_layers)]
        self.norm = w[f"{prefix}.norm.weight"]
        self.hw1, self.hb1 = w[f"{prefix}.head.0.weight"], w[f"{prefix}.head.0.bias"]
        self.hw2, self.hb2 = w[f"{prefix}.head.2.weight"], w[f"{prefix}.head.2.bias"]
        self.kv_heads = kv_heads_for_query

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        n_train = y.shape[-1]
        y_emb = _linear(y[:, None], self.y_lin, self.y_bias)
        x = mx.concatenate([x[:n_train] + y_emb, x[n_train:]], axis=0)

        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            context = x[:n_train]
            # Queries see only the leading KV heads; the context pass uses all of them.
            q_out = layer(x[n_train:], context, kv_heads=self.kv_heads)
            if last:
                x = q_out
            else:
                x = mx.concatenate([layer(context, context), q_out], axis=0)

        h = _rms_norm(x, self.norm)
        return _linear(_gelu(_linear(h, self.hw1, self.hb1)), self.hw2, self.hb2)


def load_weights(path, prefix: str | None = None) -> dict:
    """Load a checkpoint, dropping any task prefix so the keys match the module names.

    Two key layouts occur in the wild: the upstream ``regressor.pt`` holds the inner model's
    state dict with bare names, while a dump of the outer `KumoTabular` wrapper prefixes every
    key with ``models.regression.``. With ``prefix=None`` the layout is detected rather than
    assumed, so either file loads.
    """
    raw = mx.load(str(path))
    if prefix is None:
        for cand in ("models.regression.", "models.classification."):
            if any(k.startswith(cand) for k in raw):
                prefix = cand
                break
        else:
            prefix = ""
    return {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}


# ---------------------------------------------------------------- row embedding

def _rope_split_half(x: mx.array, inv_freq: mx.array) -> mx.array:
    """Rotary embedding, `split_half` layout, over the sequence axis of [S, H, C]."""
    s = x.shape[-3]
    freq = mx.arange(s, dtype=inv_freq.dtype)[:, None] * inv_freq[None, :]   # [S, C/2]
    sin, cos = mx.sin(freq)[:, None, :], mx.cos(freq)[:, None, :]
    half = inv_freq.shape[-1]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class RowAttention:
    """Attention with QK-norm, optional RoPE on the query/key, and optional gated scaling."""

    def __init__(self, w: dict, prefix: str, num_heads: int, head_dim: int,
                 rope: bool = False, gated: bool = False):
        self.qw, self.qb = w[f"{prefix}.qkv_lin.weight"], w.get(f"{prefix}.qkv_lin.bias")
        self.ow, self.ob = w[f"{prefix}.out_lin.weight"], w.get(f"{prefix}.out_lin.bias")
        self.head_scale = w.get(f"{prefix}.sdpa.query_scaling.head_scale")
        self.inv_freq = w.get(f"{prefix}.query_transform.0.inv_freq") if rope else None
        self.gated = gated
        if gated:
            self.g0w, self.g0b = w[f"{prefix}.sdpa.query_scaling.gate.0.weight"], w[f"{prefix}.sdpa.query_scaling.gate.0.bias"]
            self.g2w, self.g2b = w[f"{prefix}.sdpa.query_scaling.gate.2.weight"], w[f"{prefix}.sdpa.query_scaling.gate.2.bias"]
        self.h, self.d = num_heads, head_dim
        self.q_dim = num_heads * head_dim

    def __call__(self, query: mx.array, key_value: mx.array) -> mx.array:
        *lead, n_q, _ = query.shape
        n_kv = key_value.shape[-2]
        q = _linear(query, self.qw[: self.q_dim], None if self.qb is None else self.qb[: self.q_dim])
        kv = _linear(key_value, self.qw[self.q_dim:], None if self.qb is None else self.qb[self.q_dim:])
        k, v = mx.split(kv, 2, axis=-1)
        q = q.reshape(*lead, n_q, self.h, self.d)
        k = k.reshape(*lead, n_kv, self.h, self.d)
        v = v.reshape(*lead, n_kv, self.h, self.d)

        if self.inv_freq is not None:            # RoPE is applied BEFORE the QK-norm
            q, k = _rope_split_half(q, self.inv_freq), _rope_split_half(k, self.inv_freq)
        q, k = _qk_norm(q), _qk_norm(k)

        if self.head_scale is not None:
            scaled = q * (self.head_scale * math.log(max(n_kv, 1.0)))[None, :, None]
            if self.gated:
                # the gate reads the UNSCALED query
                g = _linear(_gelu(_linear(q, self.g0w, self.g0b)), self.g2w, self.g2b)
                scaled = scaled * (1.0 + mx.tanh(g))
            q = scaled

        qs = mx.swapaxes(q, -3, -2)
        ks = mx.swapaxes(k, -3, -2)
        vs = mx.swapaxes(v, -3, -2)
        lead_n = int(np.prod(lead)) if lead else 1
        o = _sdpa(
            qs.reshape(lead_n, self.h, n_q, self.d),
            ks.reshape(lead_n, self.h, n_kv, self.d),
            vs.reshape(lead_n, self.h, n_kv, self.d), scale=self.d ** -0.5)
        o = mx.swapaxes(o.reshape(*lead, self.h, n_q, self.d), -3, -2).reshape(*lead, n_q, self.q_dim)
        return _linear(o, self.ow, self.ob)


class RowBlock:
    """Pre-norm block around a RowAttention."""

    def __init__(self, w: dict, prefix: str, num_heads: int, head_dim: int,
                 rope: bool = False, gated: bool = False):
        self.qn, self.kn = w.get(f"{prefix}.query_norm.weight"), w.get(f"{prefix}.key_value_norm.weight")
        self.attn = RowAttention(w, f"{prefix}.attn", num_heads, head_dim, rope, gated)
        self.mn = w[f"{prefix}.mlp.0.weight"]
        self.w1, self.b1 = w[f"{prefix}.mlp.1.weight"], w[f"{prefix}.mlp.1.bias"]
        self.w2, self.b2 = w[f"{prefix}.mlp.3.weight"], w[f"{prefix}.mlp.3.bias"]

    def __call__(self, query: mx.array, key_value: mx.array) -> mx.array:
        q_in = query if self.qn is None else _rms_norm(query, self.qn)
        kv_in = key_value if self.kn is None else _rms_norm(key_value, self.kn)
        tmp = self.attn(q_in, kv_in) + query
        h = _rms_norm(tmp, self.mn)
        return tmp + _linear(_gelu(_linear(h, self.w1, self.b1)), self.w2, self.b2)


class ColBlock:
    """Induced set block: inducing points attend to the context, then the query attends to them."""

    def __init__(self, w: dict, prefix: str, num_heads: int = 4, head_dim: int = 64):
        self.points = w[f"{prefix}.inducing_points"]
        self.inducing = RowBlock(w, f"{prefix}.inducing_block", num_heads, head_dim)
        self.output = RowBlock(w, f"{prefix}.output_block", num_heads, head_dim)

    def __call__(self, query: mx.array, key_value: mx.array) -> mx.array:
        lead = query.shape[:-2]
        pts = mx.broadcast_to(self.points, (*lead, *self.points.shape))
        return self.output(query, self.inducing(pts, key_value))


class RowEmbedding:
    """SMILES-free: recipe-processed cells in, one 1024-vector per row out."""

    def __init__(self, w: dict, prefix: str = "row_embedding", group_size: int = 3,
                 channels: int = 256, num_layers: int = 6):
        self.readout = w[f"{prefix}.readout_token"]
        self.num_freq, self.cat_freq = w[f"{prefix}.cell_embedding.num_freq"], w[f"{prefix}.cell_embedding.cat_freq"]
        self.nw, self.nb = w[f"{prefix}.cell_embedding.num_lin.weight"], w[f"{prefix}.cell_embedding.num_lin.bias"]
        self.cw, self.cb = w[f"{prefix}.cell_embedding.cat_lin.weight"], w[f"{prefix}.cell_embedding.cat_lin.bias"]
        self.nan_w = w[f"{prefix}.cell_embedding.nan_lin.weight"]
        self.y_lin = w[f"{prefix}.y_lin.weight"]
        self.y_bias = w.get(f"{prefix}.y_lin.bias")
        self.col = [ColBlock(w, f"{prefix}.col_blocks.{i}") for i in range(num_layers)]
        self.row = [RowBlock(w, f"{prefix}.row_blocks.{i}", 4, 64, rope=True, gated=True)
                    for i in range(num_layers)]
        self.norm = w[f"{prefix}.norm.weight"]
        self.group_size, self.channels = group_size, channels

    def cells(self, x: mx.array, categorical: mx.array, n_train: int) -> mx.array:
        r, c = x.shape
        missing = mx.isnan(x)
        # impute from the CONTEXT rows only
        safe = mx.where(missing, mx.zeros_like(x), x)
        counts = mx.sum((~missing[:n_train]).astype(x.dtype), axis=0, keepdims=True)
        mean = mx.sum(safe[:n_train], axis=0, keepdims=True) / mx.maximum(counts, 1.0)
        mean = mx.where(counts > 0, mean, mx.zeros_like(mean))
        x = mx.where(missing, mx.broadcast_to(mean, x.shape), x)

        # each column groups with its neighbours at offsets 2^k - 1
        shift = (2 ** mx.arange(self.group_size) - 1)
        idx = (mx.arange(c)[:, None] + shift[None, :]) % c           # [C, G]
        xg = x[:, idx]                                                # [R, C, G]
        mg = missing[:, idx].astype(x.dtype)
        freq = mx.where(categorical[None, :, None, None], self.cat_freq[None, None], self.num_freq[None, None])
        xf = xg[..., None] * freq                                     # [R, C, G, F]
        fourier = mx.concatenate([mx.sin(xf), mx.cos(xf)], axis=-1)   # sin first, then cos
        # the projection is shared across groups, so the flattened contraction is a sum over G
        out = mx.einsum("rcgf,df->rcd", fourier, self.nw)
        out = out + (self.nb * self.group_size)[None, None, :]
        return out + mg @ self.nan_w.T

    def __call__(self, x: mx.array, y: mx.array, categorical: mx.array) -> mx.array:
        r, c = x.shape
        n_train = y.shape[-1]
        k = self.readout.shape[-2]
        cells = self.cells(x, categorical, n_train)                   # [R, C, D]
        y_emb = _linear(y[:, None], self.y_lin, self.y_bias)[:, None, :]
        cells = mx.concatenate([cells[:n_train] + y_emb, cells[n_train:]], axis=0)

        h = mx.swapaxes(cells, -3, -2)                                # [C, R, D]
        # The readout tokens are seeded once and then EVOLVE through the row blocks; the
        # reference recovers them by splitting them back off at the top of each iteration.
        # Re-seeding them every layer silently discards five layers of readout state.
        readout = mx.broadcast_to(self.readout, (r, k, self.channels))
        for i, (cb, rb) in enumerate(zip(self.col, self.row)):
            # col_block mixes across ROWS, keyed on the context rows only
            h = cb(h, h[:, :n_train])                                 # [C, R, D]
            # row_block mixes across readout tokens + COLUMNS
            t = mx.concatenate([readout, mx.swapaxes(h, -3, -2)], axis=-2)   # [R, K+C, D]
            last = i == len(self.col) - 1
            t = rb(t[:, :k] if last else t, t)
            if last:
                h = t
            else:
                readout = t[:, :k]
                h = mx.swapaxes(t[:, k:], -3, -2)
        return _rms_norm(h, self.norm).reshape(r, k * self.channels)
