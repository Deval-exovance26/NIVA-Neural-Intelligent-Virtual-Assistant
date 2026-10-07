"""Pytest configuration for the niva_offline test suite.

Adds ``offline-pipeline/src`` to ``sys.path`` so tests import the package
without an install step, and exposes repo-relative fixtures.
"""

import sys
from pathlib import Path

import pytest

# tests/ -> offline-pipeline/ -> repo root
OFFLINE_ROOT = Path(__file__).resolve().parents[1]
SRC = OFFLINE_ROOT / "src"
REPO_ROOT = OFFLINE_ROOT.parent

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def front_portrait(repo_root: Path) -> Path:
    return repo_root / "NIVA" / "images" / "front.png"
