"""FLAME-parameter on-disk schema — the GAGAvatar_track output contract.

This module defines the stable, versioned on-disk representation of the
per-frame FLAME tracking output that the ``GAGAvatar_track`` stage (Task 4 in
docs/plan.md) will produce from the reference video / portraits. Downstream
stages (feed-forward base, per-subject refinement, runtime driver) consume this
contract, so it is frozen here before any tracking is run.

Parameter dimensions (citing research/AVATAR.md, SpatialAvatar-0 App. A.1 and
§3.2 / §3.3):

* ``shape`` beta  in R^300  — FLAME identity shape coefficients. Per App. A.1,
  shape beta is *intentionally excluded* from the feed-forward residual head
  because "it feeds mesh geometry directly through FLAME's blendshape basis and
  is the load-bearing axis of cross-subject generalization" (§3.2). It is still
  tracked and stored here (identity is fixed per subject).
* ``expression`` psi in R^100 — FLAME expression coefficients.
* ``pose`` theta in R^6 — head + jaw pose (two 3-vectors, axis-angle:
  ``pose[0:3]`` = global/head rotation, ``pose[3:6]`` = jaw rotation).
* ``eye`` in R^6 — eye pose (two 3-vectors: ``eye[0:3]`` = left, ``eye[3:6]`` =
  right, axis-angle).

The FiLM-modulated residual head in §3.2 uses a 112-d conditioning vector =
expression(100) + pose(6) + eye(6); this schema stores exactly those code
blocks so that conditioning vector can be reconstructed as
``concat(expression, pose, eye)``.

Per-frame geometry recovered by the EMICA-based tracker [10] (App. A.1):

* ``camera`` in R^{3x4} — world->camera extrinsic (rotation|translation).
* ``bbox`` in R^4 — face bounding box ``[x_min, y_min, x_max, y_max]`` in pixels
  (used by the face-region loss L_box in Eq. 2).
* ``landmarks`` in R^{Lx2} — 2D facial landmarks in pixels (L tracker-dependent,
  e.g. 68 or mediapipe set).

A :class:`FramesManifest` ties an ordered sequence of per-frame files to the
source clip (fps, resolution, subject id). Frames are stored one NPZ per frame
(numeric arrays); the manifest is JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# --- Canonical dimensions (locked; see module docstring + App. A.1) --------
SHAPE_DIM: int = 300
EXPRESSION_DIM: int = 100
POSE_DIM: int = 6  # head(3) + jaw(3)
EYE_DIM: int = 6  # left(3) + right(3)
CAMERA_SHAPE = (3, 4)
BBOX_DIM: int = 4

#: Dimension of the FiLM conditioning vector in §3.2 (expression+pose+eye).
CONDITIONING_DIM: int = EXPRESSION_DIM + POSE_DIM + EYE_DIM  # 112

#: Schema version written into every file; bump on breaking changes.
SCHEMA_VERSION: str = "1.0.0"

PathLike = Union[str, Path]


@dataclass
class FlameFrameParams:
    """Per-frame FLAME tracking output (one frame of GAGAvatar_track).

    All array fields are :class:`numpy.ndarray` of dtype float32 (landmarks may
    be float32 pixel coordinates). ``frame_index`` is the 0-based index within
    the source clip; ``source_name`` optionally records the originating file.
    """

    frame_index: int
    shape: np.ndarray  # (300,)
    expression: np.ndarray  # (100,)
    pose: np.ndarray  # (6,) head(3)+jaw(3)
    eye: np.ndarray  # (6,) left(3)+right(3)
    camera: np.ndarray  # (3, 4)
    bbox: np.ndarray  # (4,) [x_min, y_min, x_max, y_max]
    landmarks: np.ndarray  # (L, 2)
    source_name: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.shape = _as_f32(self.shape, "shape")
        self.expression = _as_f32(self.expression, "expression")
        self.pose = _as_f32(self.pose, "pose")
        self.eye = _as_f32(self.eye, "eye")
        self.camera = _as_f32(self.camera, "camera")
        self.bbox = _as_f32(self.bbox, "bbox")
        self.landmarks = _as_f32(self.landmarks, "landmarks")

    def validate(self) -> "FlameFrameParams":
        """Assert every array has its canonical shape. Returns self for chaining."""
        _check_shape(self.shape, (SHAPE_DIM,), "shape")
        _check_shape(self.expression, (EXPRESSION_DIM,), "expression")
        _check_shape(self.pose, (POSE_DIM,), "pose")
        _check_shape(self.eye, (EYE_DIM,), "eye")
        _check_shape(self.camera, CAMERA_SHAPE, "camera")
        _check_shape(self.bbox, (BBOX_DIM,), "bbox")
        if self.landmarks.ndim != 2 or self.landmarks.shape[1] != 2:
            raise ValueError(
                f"landmarks must have shape (L, 2); got {self.landmarks.shape}"
            )
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be >= 0; got {self.frame_index}")
        return self

    @property
    def conditioning_vector(self) -> np.ndarray:
        """Return the 112-d FiLM conditioning vector = [expression | pose | eye]."""
        return np.concatenate([self.expression, self.pose, self.eye]).astype(np.float32)


@dataclass
class FramesManifest:
    """Manifest tying an ordered set of per-frame NPZ files to the source clip."""

    subject_id: str
    source_clip: str
    fps: float
    width: int
    height: int
    frames: List[str] = field(default_factory=list)  # relative NPZ paths, ordered
    tracker: str = "GAGAvatar_track"  # EMICA-based tracker [10]
    schema_version: str = SCHEMA_VERSION
    notes: str = ""

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FramesManifest":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


# --- (de)serialization -----------------------------------------------------
#: Ordered field layout used by the NPZ (de)serializer.
_FRAME_ARRAY_FIELDS = ("shape", "expression", "pose", "eye", "camera", "bbox", "landmarks")


def save_frame_npz(params: FlameFrameParams, path: PathLike) -> Path:
    """Serialize one :class:`FlameFrameParams` to a compressed ``.npz`` file."""
    params.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {name: getattr(params, name) for name in _FRAME_ARRAY_FIELDS}
    # Scalars/metadata ride along as 0-d object/int arrays.
    payload["frame_index"] = np.asarray(params.frame_index, dtype=np.int64)
    payload["source_name"] = np.asarray(params.source_name if params.source_name is not None else "")
    payload["schema_version"] = np.asarray(params.schema_version)
    np.savez_compressed(path, **payload)
    return path


def load_frame_npz(path: PathLike) -> FlameFrameParams:
    """Deserialize a :class:`FlameFrameParams` from a ``.npz`` file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        source_name = str(data["source_name"])
        params = FlameFrameParams(
            frame_index=int(data["frame_index"]),
            shape=data["shape"],
            expression=data["expression"],
            pose=data["pose"],
            eye=data["eye"],
            camera=data["camera"],
            bbox=data["bbox"],
            landmarks=data["landmarks"],
            source_name=source_name if source_name != "" else None,
            schema_version=str(data["schema_version"]),
        )
    return params.validate()


def save_manifest_json(manifest: FramesManifest, path: PathLike) -> Path:
    """Serialize a :class:`FramesManifest` to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return path


def load_manifest_json(path: PathLike) -> FramesManifest:
    """Deserialize a :class:`FramesManifest` from JSON."""
    data = json.loads(Path(path).read_text())
    return FramesManifest.from_dict(data)


# --- internal helpers ------------------------------------------------------
def _as_f32(arr: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(arr, dtype=np.float32)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"field '{name}' is not array-like: {exc}") from exc


def _check_shape(arr: np.ndarray, expected: tuple, name: str) -> None:
    if arr.shape != expected:
        raise ValueError(f"field '{name}' must have shape {expected}; got {arr.shape}")
