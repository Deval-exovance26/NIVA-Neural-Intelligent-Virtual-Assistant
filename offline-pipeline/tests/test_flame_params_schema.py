"""Round-trip tests for the FLAME-param on-disk schema."""

import numpy as np
import pytest

from niva_offline.schema import (
    EXPRESSION_DIM,
    EYE_DIM,
    POSE_DIM,
    SHAPE_DIM,
    FlameFrameParams,
    FramesManifest,
    load_frame_npz,
    load_manifest_json,
    save_frame_npz,
    save_manifest_json,
)
from niva_offline.schema.flame_params import CONDITIONING_DIM


def _make_frame(frame_index: int = 7, num_landmarks: int = 68) -> FlameFrameParams:
    rng = np.random.default_rng(frame_index)
    return FlameFrameParams(
        frame_index=frame_index,
        shape=rng.standard_normal(SHAPE_DIM).astype(np.float32),
        expression=rng.standard_normal(EXPRESSION_DIM).astype(np.float32),
        pose=rng.standard_normal(POSE_DIM).astype(np.float32),
        eye=rng.standard_normal(EYE_DIM).astype(np.float32),
        camera=rng.standard_normal((3, 4)).astype(np.float32),
        bbox=np.array([10, 20, 110, 140], dtype=np.float32),
        landmarks=rng.standard_normal((num_landmarks, 2)).astype(np.float32),
        source_name="NIVA_PSR.mp4",
    )


def test_dimensions_locked():
    assert (SHAPE_DIM, EXPRESSION_DIM, POSE_DIM, EYE_DIM) == (300, 100, 6, 6)
    assert CONDITIONING_DIM == EXPRESSION_DIM + POSE_DIM + EYE_DIM == 112


def test_conditioning_vector_layout():
    frame = _make_frame()
    cond = frame.conditioning_vector
    assert cond.shape == (CONDITIONING_DIM,)
    np.testing.assert_array_equal(cond[:EXPRESSION_DIM], frame.expression)
    np.testing.assert_array_equal(cond[EXPRESSION_DIM : EXPRESSION_DIM + POSE_DIM], frame.pose)
    np.testing.assert_array_equal(cond[EXPRESSION_DIM + POSE_DIM :], frame.eye)


def test_frame_npz_roundtrip(tmp_path):
    frame = _make_frame()
    path = tmp_path / "frame_0007.npz"
    save_frame_npz(frame, path)
    loaded = load_frame_npz(path)

    assert loaded.frame_index == frame.frame_index
    assert loaded.source_name == frame.source_name
    assert loaded.schema_version == frame.schema_version
    for field_name in ("shape", "expression", "pose", "eye", "camera", "bbox", "landmarks"):
        np.testing.assert_array_equal(getattr(loaded, field_name), getattr(frame, field_name))


def test_manifest_json_roundtrip(tmp_path):
    manifest = FramesManifest(
        subject_id="niva",
        source_clip="NIVA_PSR.mp4",
        fps=30.0,
        width=2160,
        height=2160,
        frames=["frame_0000.npz", "frame_0001.npz", "frame_0002.npz"],
        notes="tracked by GAGAvatar_track (EMICA)",
    )
    path = tmp_path / "manifest.json"
    save_manifest_json(manifest, path)
    loaded = load_manifest_json(path)

    assert loaded == manifest
    assert loaded.num_frames == 3


def test_validate_rejects_wrong_shape():
    bad = _make_frame()
    bad.shape = np.zeros(299, dtype=np.float32)  # wrong dim
    with pytest.raises(ValueError, match="shape"):
        bad.validate()
