"""NIVA Gaussian avatar export format (v1).

A single, reusable on-disk representation of a 3D Gaussian Splatting head
avatar, shared by:

* the **initial demo** (Stage 1): geometry extracted from GAGAvatar + per-Gaussian
  RGB sampled from the real rendered female avatar;
* the **refined** output (Task 8): same fields, better RGB/geometry after
  per-subject optimization;
* the **browser WebGPU renderer** (Tasks 10-11), which loads this format directly.

Design goals:
    * Browser-friendly: plain float32 arrays, little-endian, no pickling, so a
      ``.bin`` payload can be fetched and uploaded to GPU buffers directly.
    * Self-describing: a JSON sidecar records counts, dtypes, field order,
      coordinate conventions, and provenance.
    * Round-trippable: ``save`` then ``load`` reproduces arrays bit-for-bit.

Per-Gaussian fields (N = number of Gaussians, ~58K for GAGAvatar):
    positions : (N, 3) float32  world-space xyz (GAGAvatar/FLAME convention)
    rotations : (N, 4) float32  unit quaternion, (w, x, y, z) order
    scales    : (N, 3) float32  per-axis scale (linear, NOT log) — matches the
                                 rasterizer which consumes linear scales
    opacities : (N, 1) float32  in [0, 1] (already activated)
    colors    : (N, 3) float32  linear RGB in [0, 1]

Coordinate conventions (documented so the browser matches exactly):
    * World space is GAGAvatar's: +X right, +Y up, +Z toward camera is handled
      by the view matrix (see utils_renderer.build_camera_matrices). The export
      stores raw world positions; the renderer applies the same view/projection.
    * Quaternion order is (w, x, y, z), unit-normalized.
    * Scales are linear (the rasterizer multiplies them directly); the browser
      must NOT re-exp them.

Known limitations (Stage 1 RGB):
    * Colors are sampled by projecting each Gaussian center into ONE rendered
      frame (near-frontal). Gaussians occluded in that view, or facing away,
      receive approximate colors (nearest visible / fallback). Side/back regions
      are therefore less accurate until Task 8 refinement optimizes RGB against
      the full reference video. This is recorded in meta['limitations'].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

FORMAT_VERSION = "1.0.0"
PathLike = Union[str, Path]

#: Ordered field layout -> (columns, )
_FIELDS = {
    "positions": 3,
    "rotations": 4,
    "scales": 3,
    "opacities": 1,
    "colors": 3,
}


@dataclass
class GaussianAvatar:
    """In-memory container for an exportable Gaussian avatar."""

    positions: np.ndarray      # (N, 3)
    rotations: np.ndarray      # (N, 4) wxyz
    scales: np.ndarray         # (N, 3) linear
    opacities: np.ndarray      # (N, 1) in [0,1]
    colors: np.ndarray         # (N, 3) linear RGB [0,1]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = _f32(self.positions, "positions")
        self.rotations = _f32(self.rotations, "rotations")
        self.scales = _f32(self.scales, "scales")
        self.opacities = _f32(self.opacities, "opacities")
        self.colors = _f32(self.colors, "colors")

    @property
    def num_gaussians(self) -> int:
        return int(self.positions.shape[0])

    def validate(self) -> "GaussianAvatar":
        n = self.num_gaussians
        for name, cols in _FIELDS.items():
            arr = getattr(self, name)
            if arr.shape != (n, cols):
                raise ValueError(
                    f"field '{name}' must be ({n}, {cols}); got {arr.shape}"
                )
        # quaternions ~unit
        qnorm = np.linalg.norm(self.rotations, axis=1)
        if not np.allclose(qnorm, 1.0, atol=1e-2):
            raise ValueError(
                f"rotations must be unit quaternions; |q| range "
                f"[{qnorm.min():.3f}, {qnorm.max():.3f}]"
            )
        if float(self.opacities.min()) < -1e-4 or float(self.opacities.max()) > 1 + 1e-4:
            raise ValueError("opacities must lie in [0, 1]")
        if float(self.colors.min()) < -1e-4 or float(self.colors.max()) > 1 + 1e-4:
            raise ValueError("colors must lie in [0, 1]")
        return self


def save(avatar: GaussianAvatar, out_dir: PathLike, name: str = "avatar") -> Dict[str, Path]:
    """Write ``avatar`` as a browser-friendly ``.bin`` + JSON meta sidecar.

    Layout of ``<name>.bin``: all fields concatenated column-wise into a single
    (N, 14) float32 matrix in the order of ``_FIELDS``, row-major, little-endian.
    The browser can parse it with one ``Float32Array`` + a stride of 14.
    """
    avatar.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = sum(_FIELDS.values())  # 14
    n = avatar.num_gaussians
    matrix = np.empty((n, stride), dtype="<f4")
    col = 0
    for name_, cols in _FIELDS.items():
        matrix[:, col:col + cols] = getattr(avatar, name_)
        col += cols

    bin_path = out_dir / f"{name}.bin"
    matrix.tofile(bin_path)

    meta = {
        "format": "niva-gaussian-avatar",
        "version": FORMAT_VERSION,
        "num_gaussians": n,
        "stride_floats": stride,
        "dtype": "float32",
        "byte_order": "little-endian",
        "field_order": list(_FIELDS.keys()),
        "field_widths": _FIELDS,
        "conventions": {
            "quaternion_order": "wxyz",
            "scales": "linear (do not exponentiate)",
            "colors": "linear RGB in [0,1]",
            "world_space": "GAGAvatar/FLAME; renderer applies view+proj",
        },
        **avatar.meta,
    }
    meta_path = out_dir / f"{name}.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return {"bin": bin_path, "meta": meta_path}


def load(out_dir: PathLike, name: str = "avatar") -> GaussianAvatar:
    """Load a GaussianAvatar written by :func:`save`."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / f"{name}.json").read_text())
    n = int(meta["num_gaussians"])
    stride = int(meta["stride_floats"])
    matrix = np.fromfile(out_dir / f"{name}.bin", dtype="<f4").reshape(n, stride)

    fields: Dict[str, np.ndarray] = {}
    col = 0
    for name_, cols in _FIELDS.items():
        fields[name_] = np.ascontiguousarray(matrix[:, col:col + cols], dtype=np.float32)
        col += cols

    extra = {k: v for k, v in meta.items() if k not in {
        "format", "version", "num_gaussians", "stride_floats", "dtype",
        "byte_order", "field_order", "field_widths", "conventions",
    }}
    return GaussianAvatar(meta=extra, **fields).validate()


def _f32(arr: Any, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return np.ascontiguousarray(a)
