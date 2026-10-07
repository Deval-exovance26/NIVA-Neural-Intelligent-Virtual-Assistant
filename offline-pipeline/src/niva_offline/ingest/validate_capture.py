"""Capture / ingest validator for the NIVA avatar build (Task 5, docs/plan.md).

Validates the real NIVA subject capture assets before they feed the FLAME
tracker:

* **Portraits** (``NIVA/images/*.png``, 5 x 1176x1337): resolution floor,
  exactly one detectable face, and consistent-ish aspect ratio across the set.
* **Reference video** (``NIVA/video/NIVA_PSR.mp4``, 2160x2160, 30fps, ~104.5s):
  duration / fps / resolution sanity, plus a sampled-frame motion & expression
  coverage heuristic (frame-difference energy and mouth-region activity as a
  cheap proxy for expression variety).

Design: pure, unit-testable helpers (no global state), with OpenCV loaded
lazily so the module imports even when cv2 is absent (the pytest for
``read_image`` / ``detect_single_face`` requires cv2 at runtime).

Face detection uses OpenCV's bundled Haar cascade (``haarcascade_frontalface_default.xml``)
— lightweight, no extra model download, adequate for a presence/count check.
It is NOT an identity check; identity consistency across portraits+video was
confirmed by the manual audit (see this package's README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# --- Tunable thresholds ----------------------------------------------------
MIN_PORTRAIT_WIDTH = 512
MIN_PORTRAIT_HEIGHT = 512
#: Portrait aspect ratios (w/h) within this fraction of the set-median pass.
ASPECT_CONSISTENCY_TOL = 0.10
#: Reference-video sanity floors/targets.
MIN_VIDEO_DURATION_S = 10.0
MIN_VIDEO_FPS = 20.0
MIN_VIDEO_SIDE = 512
#: How many evenly-spaced frames to sample for the coverage heuristic.
VIDEO_SAMPLE_FRAMES = 24


@dataclass
class FaceCheckResult:
    """Result of a single-face detection pass."""

    num_faces: int
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)

    @property
    def is_single_face(self) -> bool:
        return self.num_faces == 1


@dataclass
class PortraitReport:
    path: str
    width: int
    height: int
    aspect: float
    face: FaceCheckResult
    passed: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class VideoReport:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    sampled_frames: int
    mean_motion_energy: float
    mean_mouth_activity: float
    passed: bool
    issues: List[str] = field(default_factory=list)


# --- OpenCV lazy loader ----------------------------------------------------
def _cv2():
    try:
        import cv2  # noqa: WPS433 (lazy import by design)
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "OpenCV (cv2) is required for capture validation. "
            "Install opencv-python-headless."
        ) from exc
    return cv2


def _face_cascade():
    cv2 = _cv2()
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():  # pragma: no cover - env-dependent
        raise RuntimeError(f"failed to load Haar cascade at {cascade_path}")
    return cascade


# --- Pure-ish helpers (unit-testable) --------------------------------------
def read_image(path: PathLike) -> np.ndarray:
    """Read an image as a BGR uint8 ndarray. Raises on failure."""
    cv2 = _cv2()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def detect_single_face(image: np.ndarray) -> FaceCheckResult:
    """Detect faces in a BGR image via Haar cascade; report count + boxes."""
    cv2 = _cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # minSize scales with image so tiny false positives are ignored.
    min_side = max(60, min(gray.shape[:2]) // 8)
    faces = _face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(min_side, min_side),
    )
    boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    return FaceCheckResult(num_faces=len(boxes), boxes=boxes)


def validate_portrait(
    image: np.ndarray,
    path: str = "<array>",
    reference_aspect: Optional[float] = None,
) -> PortraitReport:
    """Validate a single already-loaded portrait image.

    Checks resolution floor, exactly one detectable face, and (if a
    ``reference_aspect`` is given) aspect consistency within tolerance.
    """
    height, width = image.shape[:2]
    aspect = width / height if height else 0.0
    issues: List[str] = []

    if width < MIN_PORTRAIT_WIDTH or height < MIN_PORTRAIT_HEIGHT:
        issues.append(
            f"resolution {width}x{height} below floor "
            f"{MIN_PORTRAIT_WIDTH}x{MIN_PORTRAIT_HEIGHT}"
        )

    face = detect_single_face(image)
    if not face.is_single_face:
        issues.append(f"expected exactly 1 face, detected {face.num_faces}")

    if reference_aspect is not None and reference_aspect > 0:
        rel = abs(aspect - reference_aspect) / reference_aspect
        if rel > ASPECT_CONSISTENCY_TOL:
            issues.append(
                f"aspect {aspect:.3f} deviates {rel * 100:.1f}% from set median "
                f"{reference_aspect:.3f} (> {ASPECT_CONSISTENCY_TOL * 100:.0f}%)"
            )

    return PortraitReport(
        path=path,
        width=width,
        height=height,
        aspect=aspect,
        face=face,
        passed=not issues,
        issues=issues,
    )


def validate_portrait_file(
    path: PathLike, reference_aspect: Optional[float] = None
) -> PortraitReport:
    """Load + validate a portrait file."""
    image = read_image(path)
    return validate_portrait(image, path=str(path), reference_aspect=reference_aspect)


def validate_portrait_set(paths: List[PathLike]) -> List[PortraitReport]:
    """Validate a set of portraits, using the set-median aspect for consistency."""
    loaded = [(str(p), read_image(p)) for p in paths]
    aspects = [img.shape[1] / img.shape[0] for _, img in loaded if img.shape[0]]
    ref_aspect = float(np.median(aspects)) if aspects else None
    return [
        validate_portrait(img, path=name, reference_aspect=ref_aspect)
        for name, img in loaded
    ]


# --- Video heuristics ------------------------------------------------------
def _mouth_region(gray: np.ndarray) -> np.ndarray:
    """Crude lower-center crop as a mouth-region proxy (no landmark model)."""
    h, w = gray.shape[:2]
    return gray[int(h * 0.60) : int(h * 0.90), int(w * 0.30) : int(w * 0.70)]


def validate_video_file(
    path: PathLike, sample_frames: int = VIDEO_SAMPLE_FRAMES
) -> VideoReport:
    """Validate the reference video: metadata sanity + coverage heuristics.

    The coverage heuristic samples ``sample_frames`` evenly-spaced frames and
    computes (a) mean absolute inter-sample frame difference (overall motion
    energy) and (b) mean mouth-region temporal activity (expression/viseme
    proxy). These are reported, not hard-failed, since the manual audit already
    confirmed expression/viseme coverage is good.
    """
    cv2 = _cv2()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        issues: List[str] = []
        if duration < MIN_VIDEO_DURATION_S:
            issues.append(f"duration {duration:.1f}s below floor {MIN_VIDEO_DURATION_S}s")
        if fps < MIN_VIDEO_FPS:
            issues.append(f"fps {fps:.1f} below floor {MIN_VIDEO_FPS}")
        if min(width, height) < MIN_VIDEO_SIDE:
            issues.append(f"resolution {width}x{height} below floor {MIN_VIDEO_SIDE}px/side")

        motion, mouth, sampled = _sample_coverage(cap, cv2, frame_count, sample_frames)

        return VideoReport(
            path=str(path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=duration,
            sampled_frames=sampled,
            mean_motion_energy=motion,
            mean_mouth_activity=mouth,
            passed=not issues,
            issues=issues,
        )
    finally:
        cap.release()


def _sample_coverage(cap, cv2, frame_count: int, sample_frames: int):
    """Return (mean_motion_energy, mean_mouth_activity, num_sampled)."""
    if frame_count <= 0:
        return 0.0, 0.0, 0
    n = min(sample_frames, frame_count)
    indices = np.linspace(0, frame_count - 1, num=n, dtype=int)
    prev_gray = None
    prev_mouth = None
    motion_diffs: List[float] = []
    mouth_diffs: List[float] = []
    sampled = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        sampled += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        mouth = _mouth_region(gray).astype(np.float32)
        if prev_gray is not None:
            motion_diffs.append(float(np.mean(np.abs(gray.astype(np.float32) - prev_gray))))
            mouth_diffs.append(float(np.mean(np.abs(mouth - prev_mouth))))
        prev_gray = gray.astype(np.float32)
        prev_mouth = mouth
    motion = float(np.mean(motion_diffs)) if motion_diffs else 0.0
    mouth_activity = float(np.mean(mouth_diffs)) if mouth_diffs else 0.0
    return motion, mouth_activity, sampled


# --- CLI entrypoint --------------------------------------------------------
def _format_portrait(r: PortraitReport) -> str:
    status = "PASS" if r.passed else "FAIL"
    line = f"[{status}] {r.path}  {r.width}x{r.height} aspect={r.aspect:.3f} faces={r.face.num_faces}"
    for issue in r.issues:
        line += f"\n         - {issue}"
    return line


def _format_video(r: VideoReport) -> str:
    status = "PASS" if r.passed else "FAIL"
    line = (
        f"[{status}] {r.path}\n"
        f"         {r.width}x{r.height} @ {r.fps:.2f}fps, "
        f"{r.frame_count} frames, {r.duration_s:.1f}s\n"
        f"         sampled {r.sampled_frames} frames | "
        f"motion_energy={r.mean_motion_energy:.2f} "
        f"mouth_activity={r.mean_mouth_activity:.2f}"
    )
    for issue in r.issues:
        line += f"\n         - {issue}"
    return line


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="NIVA capture/ingest validator")
    parser.add_argument("--images", nargs="*", default=[], help="portrait image paths")
    parser.add_argument("--video", default=None, help="reference video path")
    args = parser.parse_args(argv)

    exit_code = 0
    if args.images:
        print("== Portraits ==")
        for report in validate_portrait_set(args.images):
            print(_format_portrait(report))
            if not report.passed:
                exit_code = 1
    if args.video:
        print("== Reference video ==")
        report = validate_video_file(args.video)
        print(_format_video(report))
        if not report.passed:
            exit_code = 1
    if not args.images and not args.video:
        parser.print_help()
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
