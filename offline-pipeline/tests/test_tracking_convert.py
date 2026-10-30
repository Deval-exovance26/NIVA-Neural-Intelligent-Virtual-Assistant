"""Unit tests for the Task 4 tracking converter (no GPU / no tracker needed)."""

from __future__ import annotations

import numpy as np

from niva_offline.schema.flame_params import load_frame_npz, load_manifest_json
from niva_offline.tracking.track_flame import (
    _frame_sort_key,
    _pad_or_trim,
    _to_bbox,
    _to_camera,
    convert_results,
)


def _fake_frame():
    return {
        "shapecode": np.random.randn(300).astype(np.float32),
        "expcode": np.random.randn(100).astype(np.float32),
        "posecode": np.random.randn(6).astype(np.float32),
        "eyecode": np.random.randn(6).astype(np.float32),
        "transform_matrix": np.eye(4, dtype=np.float32)[:3, :4],
        "bbox": np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32),  # normalized
    }


def test_frame_sort_key():
    names = ["clip_10", "clip_2", "clip_1"]
    assert sorted(names, key=_frame_sort_key) == ["clip_1", "clip_2", "clip_10"]


def test_pad_or_trim():
    assert _pad_or_trim(np.ones(50), 300).shape == (300,)
    assert _pad_or_trim(np.ones(400), 300).shape == (300,)
    assert np.allclose(_pad_or_trim(np.ones(300), 300), 1.0)


def test_to_camera_variants():
    assert _to_camera(np.eye(4)).shape == (3, 4)
    assert _to_camera(np.eye(3)).shape == (3, 4)
    assert _to_camera(np.zeros((3, 4))).shape == (3, 4)


def test_to_bbox_denormalizes():
    b = _to_bbox({"bbox": np.array([0.1, 0.1, 0.9, 0.9])})
    assert b.shape == (4,)
    # normalized values should be scaled up toward pixel space
    assert b[2] > 1.5


def test_convert_results_roundtrip(tmp_path):
    results = {f"clip_{i}": _fake_frame() for i in range(5)}
    manifest = convert_results(
        results,
        out_root=tmp_path,
        subject_id="niva",
        source_clip="NIVA_PSR.mp4",
        fps=30.0,
        width=2160,
        height=2160,
    )
    assert manifest.num_frames == 5
    # manifest + every frame NPZ must round-trip through the schema loaders
    loaded_manifest = load_manifest_json(tmp_path / "manifest.json")
    assert loaded_manifest.subject_id == "niva"
    assert loaded_manifest.tracker == "GAGAvatar_track"
    for rel in loaded_manifest.frames:
        fp = load_frame_npz(tmp_path / rel)
        fp.validate()
        assert fp.shape.shape == (300,)
        assert fp.conditioning_vector.shape == (112,)
