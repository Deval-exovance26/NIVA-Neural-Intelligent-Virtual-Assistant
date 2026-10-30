"""Unit tests for the CPU-side comparison helpers (no GPU required)."""

from __future__ import annotations

import numpy as np
import pytest

from niva_offline.compare.comparison_common import (
    ComparisonConfig,
    PANEL_LABELS,
    PANEL_ORDER,
    build_ffmpeg_cmd,
    composite_panels,
    ensure_rgb_uint8,
    frame_index_to_timestamp,
    select_frame_keys,
    sort_frame_keys,
    timestamp_to_selection_index,
)


def test_sort_frame_keys_numeric_order():
    keys = ["frame_000010", "frame_000002", "frame_000100", "frame_000001"]
    assert sort_frame_keys(keys) == [
        "frame_000001", "frame_000002", "frame_000010", "frame_000100",
    ]


def test_select_frame_keys_stride_and_cap():
    keys = [f"frame_{i:06d}" for i in range(20)]
    sel = select_frame_keys(keys, stride=5, max_frames=3)
    assert sel == ["frame_000000", "frame_000005", "frame_000010"]


def test_select_frame_keys_sync_is_deterministic():
    # The SAME selected list must drive all three panels -> deterministic.
    keys = [f"frame_{i:06d}" for i in range(100)]
    a = select_frame_keys(keys, stride=3)
    b = select_frame_keys(list(reversed(keys)), stride=3)
    assert a == b  # order independent of input ordering


def test_select_frame_keys_rejects_bad_stride():
    with pytest.raises(ValueError):
        select_frame_keys(["frame_000000"], stride=0)


def test_frame_index_to_timestamp():
    assert frame_index_to_timestamp(0, 25) == 0.0
    assert frame_index_to_timestamp(25, 25) == 1.0
    assert frame_index_to_timestamp(50, 25) == 2.0


def test_timestamp_to_selection_index_clamps():
    assert timestamp_to_selection_index(0.0, 25, 100, 1) == 0
    assert timestamp_to_selection_index(1.0, 25, 100, 1) == 25
    # beyond range -> clamp to last
    assert timestamp_to_selection_index(999.0, 25, 100, 1) == 99
    # negative -> clamp to 0
    assert timestamp_to_selection_index(-5.0, 25, 100, 1) == 0


def test_ensure_rgb_uint8_from_float_hwc():
    img = np.ones((8, 8, 3), dtype=np.float32) * 0.5
    out = ensure_rgb_uint8(img)
    assert out.dtype == np.uint8 and out.shape == (8, 8, 3)
    assert 120 <= int(out.mean()) <= 135


def test_ensure_rgb_uint8_from_chw_is_transposed():
    # (C,H,W) float -> (H,W,C) uint8  (guards the historical channel-order bug)
    img = np.zeros((3, 8, 8), dtype=np.float32)
    img[0] = 1.0  # red plane
    out = ensure_rgb_uint8(img)
    assert out.shape == (8, 8, 3)
    assert out[..., 0].mean() > 200 and out[..., 1].mean() < 50


def test_ensure_rgb_uint8_truncates_extra_channels():
    # 32-channel rasterizer buffer must NOT leak through as "color".
    img = np.zeros((8, 8, 32), dtype=np.float32)
    img[..., :3] = 0.25
    out = ensure_rgb_uint8(img)
    assert out.shape == (8, 8, 3)


def test_composite_panels_shape_and_order():
    size, band = 32, 10
    imgs = [
        np.zeros((64, 64, 3), np.uint8),           # original (black)
        np.full((64, 64, 3), 128, np.uint8),       # base (gray)
        np.full((64, 64, 3), 255, np.uint8),       # refined (white)
    ]
    labels = [PANEL_LABELS[p] for p in PANEL_ORDER]
    out = composite_panels(imgs, labels, panel_size=size, label_band_h=band)
    # 3 panels wide, each size px + band on top
    assert out.shape == (size + band, size * 3, 3)
    # left third darker than right third (order preserved: black ... white)
    left = out[band:, :size].mean()
    right = out[band:, 2 * size:].mean()
    assert left < right


def test_composite_panels_length_mismatch_raises():
    with pytest.raises(ValueError):
        composite_panels([np.zeros((4, 4, 3), np.uint8)], ["a", "b"], 4, 2)


def test_build_ffmpeg_cmd_is_argv_list_no_shell():
    cmd = build_ffmpeg_cmd("/tmp/frames", 25, "/tmp/out.mp4")
    assert isinstance(cmd, list)
    assert cmd[0] == "ffmpeg"
    assert "25" in cmd
    assert cmd[-1] == "/tmp/out.mp4"
    # frame pattern present
    assert any("frame_%06d.png" in c for c in cmd)


def test_config_paths_coerced_to_path():
    cfg = ComparisonConfig(
        base_export_dir="/a/b", refined_export_dir="/c/d",
        driver_dir="/e/f", out_dir="/g/h",
    )
    assert str(cfg.base_export_dir) == "/a/b"
    assert cfg.panel_size == 512 and cfg.fps == 25
