"""Task 4 — FLAME tracking driver (GAGAvatar_track wrapper).

Runs the vendored ``GAGAvatar_track`` monocular FLAME tracker on a reference
video and converts its per-frame output into the stable on-disk contract
defined in :mod:`niva_offline.schema.flame_params`.

The tracker itself lives in its own conda env (``GAGAvatar_track``: python 3.12,
torch 2.4.1, pytorch3d) because of the pytorch3d dependency. This module does
NOT import the tracker; it invokes ``track_video.py`` as a subprocess under that
env and then reads the resulting ``*.pkl`` back in the (lightweight) current
interpreter to emit our schema. Keeping the heavy tracker out-of-process is what
lets the rest of the pipeline stay on the pinned ``niva-offline`` env.

GAGAvatar_track per-frame result keys (see engines/engine_core.py):
    shapecode (300,), expcode (100,), posecode (6,), eyecode (6,),
    transform_matrix (3x4 world->cam affine), bbox (4, normalized /512),
    optionally landmarks (L,2).

Mapping to :class:`niva_offline.schema.flame_params.FlameFrameParams`:
    shape       <- shapecode
    expression  <- expcode
    pose        <- posecode
    eye         <- eyecode
    camera      <- transform_matrix          (3x4)
    bbox        <- bbox * 512 (de-normalized to pixels) if present else frame box
    landmarks   <- landmarks if present else empty (0,2)

Usage (on the GPU box):
    python -m niva_offline.tracking.track_flame \
        --video /opt/dlami/nvme/data/raw-capture/v1/video/NIVA_PSR.mp4 \
        --tracker-repo offline-pipeline/checkpoints/repos/GAGAvatar_track \
        --out /opt/dlami/nvme/data/tracking/v1 \
        --subject-id niva \
        --conda-env GAGAvatar_track
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from niva_offline.schema.flame_params import (
    BBOX_DIM,
    EXPRESSION_DIM,
    EYE_DIM,
    POSE_DIM,
    SHAPE_DIM,
    FlameFrameParams,
    FramesManifest,
    save_frame_npz,
    save_manifest_json,
)


def _frame_sort_key(name: str) -> int:
    """GAGAvatar_track keys frames as '<something>_<index>'; sort by index."""
    try:
        return int(name.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def run_tracker(
    video_path: Path,
    tracker_repo: Path,
    conda_env: str,
    no_vis: bool = False,
) -> Path:
    """Invoke GAGAvatar_track's track_video.py under its conda env.

    Returns the tracker's output directory (``<repo>/outputs/<video_stem>``).
    Idempotent: if a ``smoothed.pkl`` / ``optim.pkl`` already exists there, the
    tracker will reuse its cached ``base.pkl`` / ``optim.pkl`` stages.
    """
    tracker_repo = tracker_repo.resolve()
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not (tracker_repo / "track_video.py").is_file():
        raise FileNotFoundError(
            f"track_video.py not found under tracker repo: {tracker_repo}"
        )

    stem = video_path.name.split(".")[0]
    out_dir = tracker_repo / "outputs" / stem

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "track_video.py", "-v", str(video_path),
    ]
    if no_vis:
        cmd.append("--no_vis")

    print(f"[track] running: {' '.join(cmd)}", flush=True)
    print(f"[track] cwd    : {tracker_repo}", flush=True)
    # Stream output live so long tracking runs are observable.
    proc = subprocess.run(cmd, cwd=str(tracker_repo))
    if proc.returncode != 0:
        raise RuntimeError(
            f"GAGAvatar_track track_video.py exited {proc.returncode}"
        )
    if not out_dir.is_dir():
        raise RuntimeError(
            f"tracker finished but output dir missing: {out_dir}"
        )
    return out_dir


def _load_results(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load the best available per-frame result pkl (smoothed > optim)."""
    for name in ("smoothed.pkl", "optim.pkl"):
        pkl = out_dir / name
        if pkl.is_file():
            print(f"[convert] loading {pkl}", flush=True)
            with open(pkl, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(
        f"no smoothed.pkl or optim.pkl in {out_dir} — did tracking produce output?"
    )


def _pad_or_trim(arr: np.ndarray, dim: int) -> np.ndarray:
    """Coerce a 1-D code vector to exactly ``dim`` (pad with zeros / trim)."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] == dim:
        return arr
    out = np.zeros((dim,), dtype=np.float32)
    n = min(dim, arr.shape[0])
    out[:n] = arr[:n]
    return out


def _to_camera(transform_matrix: Any) -> np.ndarray:
    """Coerce the tracker transform_matrix to a (3,4) float32 extrinsic."""
    m = np.asarray(transform_matrix, dtype=np.float32)
    if m.shape == (3, 4):
        return m
    if m.shape == (4, 4):
        return m[:3, :4]
    if m.shape == (3, 3):  # rotation only -> append zero translation
        return np.concatenate([m, np.zeros((3, 1), np.float32)], axis=1)
    raise ValueError(f"unexpected transform_matrix shape: {m.shape}")


def _to_bbox(frame: Dict[str, Any]) -> np.ndarray:
    """Recover a pixel-space bbox. Tracker stores bbox normalized by /512."""
    bbox = frame.get("bbox")
    if bbox is not None:
        b = np.asarray(bbox, dtype=np.float32).reshape(-1)[:BBOX_DIM]
        # Stored normalized (/512) in engine_core; de-normalize to pixels.
        if np.all(b <= 1.5):
            b = b * 512.0
        if b.shape[0] == BBOX_DIM:
            return b.astype(np.float32)
    return np.zeros((BBOX_DIM,), dtype=np.float32)


def _to_landmarks(frame: Dict[str, Any]) -> np.ndarray:
    for key in ("landmarks", "lmks_2d70", "lmks", "landmarks_2d"):
        lmk = frame.get(key)
        if lmk is not None:
            arr = np.asarray(lmk, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr
    return np.zeros((0, 2), dtype=np.float32)


def convert_results(
    results: Dict[str, Dict[str, Any]],
    out_root: Path,
    subject_id: str,
    source_clip: str,
    fps: float,
    width: int,
    height: int,
) -> FramesManifest:
    """Convert tracker results -> per-frame NPZ + manifest in our schema."""
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_names = sorted(results.keys(), key=_frame_sort_key)
    rel_paths = []
    for idx, name in enumerate(frame_names):
        fr = results[name]
        params = FlameFrameParams(
            frame_index=idx,
            shape=_pad_or_trim(fr["shapecode"], SHAPE_DIM),
            expression=_pad_or_trim(fr["expcode"], EXPRESSION_DIM),
            pose=_pad_or_trim(fr["posecode"], POSE_DIM),
            eye=_pad_or_trim(fr["eyecode"], EYE_DIM),
            camera=_to_camera(fr["transform_matrix"]),
            bbox=_to_bbox(fr),
            landmarks=_to_landmarks(fr),
            source_name=name,
        ).validate()
        rel = f"frames/frame_{idx:06d}.npz"
        save_frame_npz(params, out_root / rel)
        rel_paths.append(rel)

    manifest = FramesManifest(
        subject_id=subject_id,
        source_clip=source_clip,
        fps=fps,
        width=width,
        height=height,
        frames=rel_paths,
        tracker="GAGAvatar_track",
        notes="Task 4 output: GAGAvatar_track EMICA-based monocular FLAME tracking.",
    )
    save_manifest_json(manifest, out_root / "manifest.json")
    return manifest


def _probe_video(video_path: Path) -> Dict[str, float]:
    """Best-effort probe of fps/width/height via opencv (fallbacks if absent)."""
    info = {"fps": 30.0, "width": 0, "height": 0}
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            info["fps"] = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception as exc:  # pragma: no cover - probe is best-effort
        print(f"[warn] video probe failed ({exc}); using defaults", flush=True)
    return info


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="NIVA Task 4 — FLAME tracking")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--tracker-repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subject-id", default="niva", type=str)
    parser.add_argument("--conda-env", default="GAGAvatar_track", type=str)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument(
        "--skip-track", action="store_true",
        help="skip running the tracker; only convert existing pkl output",
    )
    args = parser.parse_args(argv)

    video_path = args.video.resolve()
    tracker_repo = args.tracker_repo.resolve()
    out_root = args.out.resolve()

    stem = video_path.name.split(".")[0]
    tracker_out = tracker_repo / "outputs" / stem

    if not args.skip_track:
        tracker_out = run_tracker(
            video_path, tracker_repo, args.conda_env, no_vis=args.no_vis
        )

    results = _load_results(tracker_out)
    probe = _probe_video(video_path)

    manifest = convert_results(
        results,
        out_root=out_root,
        subject_id=args.subject_id,
        source_clip=video_path.name,
        fps=probe["fps"],
        width=int(probe["width"]),
        height=int(probe["height"]),
    )

    print(
        f"[OK] Task 4 complete: {manifest.num_frames} frames -> {out_root}",
        flush=True,
    )
    print(f"[OK] manifest: {out_root / 'manifest.json'}", flush=True)
    # Surface the tracker's visualization if present.
    vis = tracker_out / "tracked.mp4"
    if vis.is_file():
        print(f"[OK] tracker visualization: {vis}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
