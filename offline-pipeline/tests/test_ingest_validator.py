"""Tests for the capture/ingest validator helpers.

Runs the unit-testable helpers on the real ``NIVA/images/front.png`` portrait.
Skips gracefully if OpenCV or the fixture image is unavailable.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python-headless required")

from niva_offline.ingest import (  # noqa: E402
    detect_single_face,
    read_image,
    validate_portrait,
    validate_portrait_file,
)
from niva_offline.ingest.validate_capture import (  # noqa: E402
    MIN_PORTRAIT_HEIGHT,
    MIN_PORTRAIT_WIDTH,
)


@pytest.fixture(scope="module")
def front_image(front_portrait):
    if not front_portrait.exists():
        pytest.skip(f"fixture image missing: {front_portrait}")
    return read_image(front_portrait)


def test_read_image_shape(front_image):
    assert front_image.ndim == 3
    assert front_image.shape[2] == 3
    h, w = front_image.shape[:2]
    # Known capture resolution of the NIVA portraits is 1176x1337.
    assert w >= MIN_PORTRAIT_WIDTH and h >= MIN_PORTRAIT_HEIGHT


def test_detect_single_face(front_image):
    result = detect_single_face(front_image)
    assert result.num_faces == 1, f"expected 1 face, got {result.num_faces}"
    assert result.is_single_face
    x, y, w, h = result.boxes[0]
    assert w > 0 and h > 0


def test_validate_portrait_passes(front_image, front_portrait):
    report = validate_portrait(front_image, path=str(front_portrait))
    assert report.passed, f"front.png failed validation: {report.issues}"
    assert report.face.is_single_face
    assert report.width == front_image.shape[1]
    assert report.height == front_image.shape[0]


def test_validate_portrait_file(front_portrait):
    report = validate_portrait_file(front_portrait)
    assert report.passed, report.issues


def test_resolution_floor_catches_tiny_image():
    tiny = np.zeros((64, 64, 3), dtype=np.uint8)
    report = validate_portrait(tiny, path="<tiny>")
    assert not report.passed
    assert any("resolution" in issue for issue in report.issues)
