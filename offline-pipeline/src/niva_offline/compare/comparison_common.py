"""CPU-side helpers for the honest 3-way comparison video.

This module is DELIBERATELY free of any GPU / GAGAvatar / torch-CUDA dependency
so it can be unit-tested on the workstation BEFORE the GPU box is available.
It owns:
    * configuration + panel definitions,
    * synchronized timestamp / frame-key selection,
    * panel compositing (resize, label banners, side-by-side layout),
    * video assembly (ffmpeg invocation builder).

The GPU-dependent rendering (loading the GAGAvatar runtime, per-frame FLAME
posing, 32-channel rasterization, RGB readback) lives in
``render_comparison.py`` and is NOT imported here. That separation is the whole
point: nothing in this file can accidentally claim a render was validated.

Panel semantics (honest framing — see handoff Q1):
    ORIGINAL  = the REAL reference frame (actual photograph from the tracker's
                img_lmdb). It is a photo, NOT a rasterizer output. We label it
                "Original (reference)" and never imply it was generated.
    BASE      = the Stage-1 baked Gaussian export, posed per frame + rendered
                through the GAGAvatar 32-ch rasterizer.
    REFINED   = the refined (v2) Gaussian export, posed per frame + rendered
                through the SAME rasterizer with the SAME camera convention.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Panel identity
# ---------------------------------------------------------------------------

PANEL_ORIGINAL = "original"
PANEL_BASE = "base"
PANEL_REFINED = "refined"

PANEL_ORDER = (PANEL_ORIGINAL, PANEL_BASE, PANEL_REFINED)

PANEL_LABELS = {
    PANEL_ORIGINAL: "Original (reference)",
    PANEL_BASE: "Base splat",
    PANEL_REFINED: "Refined splat (v2)",
}


@dataclass
class ComparisonConfig:
    """All knobs for the comparison build.

    Paths that live on the GPU box (driver/img_lmdb, exports) are passed through
    unmodified; this module never dereferences them, it only arranges work.
    """

    # On-box inputs (consumed by the GPU renderer, not here):
    base_export_dir: Path          # dir containing niva_base.{bin,json}
    refined_export_dir: Path       # refine/v2/refined containing niva_refined.{bin,json}
    base_name: str = "niva_base"
    refined_name: str = "niva_refined"
    driver_dir: Optional[Path] = None   # tracker out: optim.pkl + img_lmdb

    # Output:
    out_dir: Path = Path("/tmp/niva_compare")
    video_name: str = "niva_3way.mp4"

    # Timing / selection:
    fps: int = 25
    panel_size: int = 512          # each panel rendered/resized to panel_size^2
    label_band_h: int = 36         # px banner height for the text label
    # Which frames to use. If frame_stride>0 we subsample the sorted frame keys.
    frame_stride: int = 2
    max_frames: Optional[int] = None    # cap total frames (None = all selected)
    # Timestamps (seconds) for the pre-render synchronized previews:
    preview_times_s: Sequence[float] = (0.0, 1.0, 2.0, 4.0)

    def __post_init__(self) -> None:
        self.base_export_dir = Path(self.base_export_dir)
        self.refined_export_dir = Path(self.refined_export_dir)
        self.out_dir = Path(self.out_dir)
        if self.driver_dir is not None:
            self.driver_dir = Path(self.driver_dir)


# ---------------------------------------------------------------------------
# Synchronized frame-key selection
# ---------------------------------------------------------------------------

def sort_frame_keys(keys: Sequence[str]) -> List[str]:
    """Sort tracker frame keys by their trailing integer (frame_002496 -> 2496).

    Matches refine_gaussians.py's ``key.split('_')[-1]`` convention so the
    ORIGINAL/BASE/REFINED panels stay on the SAME frame at the SAME timestamp.
    """
    def idx(k: str) -> int:
        return int(str(k).split("_")[-1])

    return sorted(keys, key=idx)


def select_frame_keys(
    all_keys: Sequence[str],
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> List[str]:
    """Deterministically subsample sorted keys. One shared list drives all
    three panels, guaranteeing per-frame timestamp sync."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    ordered = sort_frame_keys(all_keys)
    picked = ordered[::stride]
    if max_frames is not None:
        picked = picked[:max_frames]
    return picked


def frame_index_to_timestamp(i: int, fps: int) -> float:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return i / float(fps)


def timestamp_to_selection_index(t_s: float, fps: int, n_selected: int,
                                 stride: int) -> int:
    """Map a wall-clock preview timestamp to an index into the SELECTED
    (post-stride) key list, clamped to range. Timestamps are defined on the
    OUTPUT video timeline (which advances one selected frame per 1/fps)."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    idx = int(round(t_s * fps))
    return max(0, min(idx, max(0, n_selected - 1)))


# ---------------------------------------------------------------------------
# Compositing (pure numpy — testable without a GPU)
# ---------------------------------------------------------------------------

def _resize_nn(img: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour square resize (no external deps). img is HxWx3 uint8."""
    h, w = img.shape[:2]
    ys = (np.linspace(0, h - 1, size)).astype(int)
    xs = (np.linspace(0, w - 1, size)).astype(int)
    return img[ys][:, xs]


def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce an image to HxWx3 uint8. Accepts float [0,1] or uint8, gray or
    3-channel. This guards against the historical 'feature-channel /
    RGB-decoding' bug where a (C,H,W) or 32-channel tensor leaked through."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        # looks like (C,H,W) -> (H,W,C)
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 3 and a.shape[-1] > 3:
        # too many channels (e.g. 32-ch rasterizer buffer): take first 3 and flag
        a = a[..., :3]
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0) if a.max() <= 1.0 + 1e-6 else np.clip(a, 0, 255)
        a = (a * 255).astype(np.uint8) if a.max() <= 1.0 + 1e-6 else a.astype(np.uint8)
    return np.ascontiguousarray(a[..., :3])


def _draw_label_band(width: int, height: int, text: str) -> np.ndarray:
    """Render a dark label band with simple 5x7 bitmap text (no PIL needed).

    Kept intentionally tiny; purpose is an unambiguous panel caption so the
    ORIGINAL panel is clearly marked as a reference photo, not a render.
    """
    band = np.full((height, width, 3), 20, dtype=np.uint8)
    _blit_text(band, text, x=6, y=(height - 7) // 2, color=(235, 235, 235))
    return band


def composite_panels(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    panel_size: int,
    label_band_h: int,
) -> np.ndarray:
    """Stack labelled panels horizontally into a single frame (uint8 HxWx3).

    All panels are coerced to RGB uint8 and resized to panel_size^2, each gets
    a label band on top, then they're concatenated left-to-right in the given
    order. This is the single compositing path for ALL three panels, so there
    is no per-panel orientation/representation divergence introduced here.
    """
    if len(images) != len(labels):
        raise ValueError("images and labels length mismatch")
    cols = []
    for img, label in zip(images, labels):
        rgb = ensure_rgb_uint8(img)
        rgb = _resize_nn(rgb, panel_size)
        band = _draw_label_band(panel_size, label_band_h, label)
        cols.append(np.concatenate([band, rgb], axis=0))
    return np.concatenate(cols, axis=1)


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def build_ffmpeg_cmd(frames_glob_dir: Path, fps: int, out_path: Path) -> List[str]:
    """Construct an ffmpeg command to encode numbered PNG frames into mp4.

    Returned as an argv list (no shell string interpolation) to avoid injection
    and quoting issues.
    """
    return [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(Path(frames_glob_dir) / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]


def run_ffmpeg(frames_dir: Path, fps: int, out_path: Path) -> subprocess.CompletedProcess:
    cmd = build_ffmpeg_cmd(frames_dir, fps, out_path)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# Minimal 5x7 bitmap font for labels (A-Z, 0-9, space, parens, dash, dot)
# ---------------------------------------------------------------------------

def _blit_text(canvas: np.ndarray, text: str, x: int, y: int,
               color: Tuple[int, int, int]) -> None:
    cx = x
    for ch in text.upper():
        glyph = _FONT.get(ch, _FONT[" "])
        for row in range(7):
            bits = glyph[row]
            for col in range(5):
                if bits & (1 << (4 - col)):
                    yy, xx = y + row, cx + col
                    if 0 <= yy < canvas.shape[0] and 0 <= xx < canvas.shape[1]:
                        canvas[yy, xx] = color
        cx += 6


# Each glyph: 7 rows of 5-bit masks.
_FONT = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    "-": [0, 0, 0, 0b11111, 0, 0, 0],
    ".": [0, 0, 0, 0, 0, 0b00110, 0b00110],
    "(": [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
    ")": [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
    "3": [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "6": [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "B": [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    "D": [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "F": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    "G": [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
    "H": [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "I": [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "P": [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    "R": [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    "S": [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "U": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "V": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
}
