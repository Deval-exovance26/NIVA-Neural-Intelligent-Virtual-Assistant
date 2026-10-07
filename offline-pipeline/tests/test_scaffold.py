"""Smoke tests for the offline-pipeline scaffold.

These do NOT require a GPU or model weights — they assert the package imports and the
locked FLAME/Gaussian representation constants (docs/architecture.md). The GPU/CUDA
smoke test lives in niva_offline.verify_env (run via `make verify-env` on EC2).
"""

import niva_offline as niva


def test_package_imports():
    assert niva.__version__ == "0.1.0"


def test_locked_representation_constants():
    # FLAME 2020 + patch_teeth.
    assert niva.FLAME_NUM_VERTS == 5143
    assert niva.FLAME_NUM_FACES == 10144
    # flame_uv.npz 256x256 UV unwrap; ~58,173 valid-UV Gaussians.
    assert niva.UV_GRID == 256
    assert niva.NUM_GAUSSIANS == 58173
    # ARTalk-style FLAME motioncode frame rate.
    assert niva.MOTION_FPS == 25


def test_stage_subpackages_present():
    import importlib

    for sub in ("flame", "tracking", "base", "binding", "refine", "export"):
        importlib.import_module(f"niva_offline.{sub}")
