"""Fetching, converting and caching the Kumo-Tabular checkpoint.

The upstream weights live on the Hugging Face Hub as a torch ``.pt`` state dict. MLX cannot read
that, so the first use converts it once to safetensors in a cache directory and every later call
is a straight memory-mapped load.

Conversion has one trap: all twelve RoPE ``inv_freq`` buffers are the *same tensor object*, and
``safetensors`` refuses to write shared storage. Each tensor is therefore cloned.

    from kumomlx import weights
    path = weights.resolve()            # downloads + converts on first call, then cached
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["resolve", "convert", "cache_dir", "HF_REPO", "HF_REVISION"]

HF_REPO = "nvidia/Kumo-Tabular"
HF_REVISION = "v1.0.0"

#: Only ``large`` is implemented by :mod:`kumomlx.network`; the other sizes change the layer
#: counts and channel widths, and the port does not yet read them from the checkpoint.
SUPPORTED_SIZES = ("large",)


def cache_dir() -> Path:
    """Where converted weights are kept. Override with ``KUMOMLX_CACHE``."""
    d = os.environ.get("KUMOMLX_CACHE")
    base = Path(d) if d else Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "kumomlx"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _filename(size: str, task: str) -> str:
    stem = "regressor" if task == "regression" else "classifier"
    return f"{size}/{stem}.pt"


def convert(size: str = "large", task: str = "regression",
            out: str | Path | None = None, *, local_files_only: bool = False) -> Path:
    """Download the upstream checkpoint and rewrite it as safetensors.

    Imports torch and huggingface_hub lazily: they are needed for this one-time step, not to run
    the model afterwards.
    """
    import torch                                    # noqa: PLC0415
    from huggingface_hub import hf_hub_download     # noqa: PLC0415
    from safetensors.torch import save_file         # noqa: PLC0415

    src = hf_hub_download(repo_id=HF_REPO, filename=_filename(size, task),
                          revision=HF_REVISION, local_files_only=local_files_only)
    ckpt = torch.load(src, map_location="cpu", weights_only=True)

    out = Path(out) if out is not None else cache_dir() / f"kumo_{size}_{task}.safetensors"
    # .clone() is not optional: the RoPE inv_freq buffers alias one another and safetensors
    # rejects shared storage. .contiguous() guards against any non-dense view in the checkpoint.
    save_file({k: v.detach().clone().contiguous() for k, v in ckpt.items()}, str(out))
    return out


def resolve(size: str = "large", task: str = "regression", *,
            local_files_only: bool = False) -> Path:
    """Path to MLX-ready weights, converting on first use.

    ``KUMOMLX_WEIGHTS`` short-circuits everything and is used as-is, which is the escape hatch
    for an already-exported file.
    """
    override = os.environ.get("KUMOMLX_WEIGHTS")
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"KUMOMLX_WEIGHTS points at a missing file: {p}")
        return p

    if size not in SUPPORTED_SIZES:
        raise ValueError(
            f"size={size!r} is not implemented; the port covers {SUPPORTED_SIZES}. "
            "The other checkpoints convert fine via convert(), but the network's layer counts "
            "and widths are fixed for 'large'.")

    cached = cache_dir() / f"kumo_{size}_{task}.safetensors"
    if cached.exists():
        return cached
    return convert(size, task, cached, local_files_only=local_files_only)
