"""Unit tests for the NIVA Gaussian export format (no GPU needed)."""

from __future__ import annotations

import numpy as np
import pytest

from niva_offline.export.gaussian_format import GaussianAvatar, load, save


def _make_avatar(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return GaussianAvatar(
        positions=rng.standard_normal((n, 3)).astype(np.float32),
        rotations=q,
        scales=(rng.random((n, 3)).astype(np.float32) * 0.05),
        opacities=rng.random((n, 1)).astype(np.float32),
        colors=rng.random((n, 3)).astype(np.float32),
        meta={"source": "unit-test", "subject": "niva-female"},
    )


def test_validate_ok():
    _make_avatar().validate()


def test_roundtrip_bitexact(tmp_path):
    a = _make_avatar(n=2048, seed=1)
    paths = save(a, tmp_path, name="avatar")
    assert paths["bin"].exists() and paths["meta"].exists()
    b = load(tmp_path, name="avatar")
    assert b.num_gaussians == a.num_gaussians
    for fld in ("positions", "rotations", "scales", "opacities", "colors"):
        np.testing.assert_array_equal(getattr(a, fld), getattr(b, fld))
    # provenance meta survives
    assert b.meta.get("subject") == "niva-female"


def test_meta_records_conventions(tmp_path):
    import json
    a = _make_avatar(n=16)
    save(a, tmp_path, name="av")
    meta = json.loads((tmp_path / "av.json").read_text())
    assert meta["conventions"]["quaternion_order"] == "wxyz"
    assert meta["field_order"] == ["positions", "rotations", "scales", "opacities", "colors"]
    assert meta["stride_floats"] == 14


def test_rejects_bad_quaternion():
    a = _make_avatar(n=10)
    a.rotations = (a.rotations * 5.0).astype(np.float32)  # non-unit
    with pytest.raises(ValueError, match="unit quaternions"):
        a.validate()


def test_rejects_out_of_range_color():
    a = _make_avatar(n=10)
    a.colors = (a.colors + 2.0).astype(np.float32)
    with pytest.raises(ValueError, match="colors"):
        a.validate()


def test_opacity_1d_coerced_to_column():
    a = _make_avatar(n=10)
    a.opacities = np.ones(10, dtype=np.float32)  # 1-D
    a.__post_init__()
    assert a.opacities.shape == (10, 1)
