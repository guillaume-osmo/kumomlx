"""Tests that need neither the 823 MB checkpoint nor a network connection."""

import numpy as np
import pytest

from kumomlx import load_boiling_point, weights


def test_bundled_dataset_shape():
    d = load_boiling_point()
    assert len(d) == 1000
    assert d.x.shape == (1000, len(d.feature_names)) and d.x.dtype == np.float32
    assert d.y.shape == (1000,) and np.isfinite(d.y).all()
    assert len(d.smiles) == 1000


def test_split_is_a_partition():
    d = load_boiling_point()
    (xc, yc), (xq, yq), (ci, qi) = d.split(n_context=800, seed=0)
    assert len(yc) == 800 and len(yq) == 200
    assert set(ci).isdisjoint(qi), "context and query must not overlap"
    assert len(set(ci) | set(qi)) == 1000


def test_split_is_deterministic_and_seed_sensitive():
    d = load_boiling_point()
    a = d.split(seed=0)[2][0]
    assert (a == d.split(seed=0)[2][0]).all()
    assert not (a == d.split(seed=1)[2][0]).all()


def test_load_weights_detects_key_layout(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from kumomlx import network

    for prefix in ("", "models.regression."):
        p = tmp_path / f"w{len(prefix)}.safetensors"
        mx.save_safetensors(str(p), {f"{prefix}icl_block.norm.weight": mx.ones((4,))})
        assert list(network.load_weights(p)) == ["icl_block.norm.weight"]


def test_unsupported_size_is_refused(monkeypatch):
    monkeypatch.delenv("KUMOMLX_WEIGHTS", raising=False)
    with pytest.raises(ValueError, match="not implemented"):
        weights.resolve(size="small")


def test_weights_override_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("KUMOMLX_WEIGHTS", str(tmp_path / "nope.safetensors"))
    with pytest.raises(FileNotFoundError):
        weights.resolve()


def test_descriptor_block_is_the_full_rdkit_list():
    d = load_boiling_point()
    assert len(d.feature_names) == 217, "the bundled block is RDKit 2025.09.x descList"


def test_ipc_is_the_repaired_column_not_the_raw_one():
    """Raw `Ipc` grows super-exponentially and leaves float32 range; the shipped column is the
    log-space form. See DESCRIPTORS.md."""
    names = load_boiling_point().feature_names
    assert "Ipc_log2" in names
    assert "Ipc" not in names, "raw Ipc must not ship -- it overflows float32 near 170 atoms"
    assert "AvgIpc" in names, "the intensive half is kept"


def test_no_infinities_anywhere():
    """NaN is allowed (Kumo imputes from context); Inf is not, because a float32 cast plus a
    nan_to_num fill turns it into 0 and silently reverses its meaning."""
    x = load_boiling_point().x
    assert not np.isinf(x).any()


def test_ipc_log2_is_in_a_sane_range():
    d = load_boiling_point()
    v = d.x[:, d.feature_names.index("Ipc_log2")]
    v = v[~np.isnan(v)]
    assert v.size and np.abs(v).max() < 1e3, "log-space Ipc should be O(n_atoms), not O(2**n)"
