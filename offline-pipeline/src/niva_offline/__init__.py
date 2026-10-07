"""NIVA offline avatar build pipeline.

See docs/plan.md (Arc A) and offline-pipeline/README.md for the authoritative design.

Pipeline stages (reconstruction):
    flame     — FLAME 2020 foundation + asset loading (Task 3)
    tracking  — GAGAvatar_track wrappers: video/image -> FLAME params (Task 4)
    base      — GAGAvatar feed-forward base avatar (Task 6)
    binding   — FLAME-bound Gaussian deformation, Eq. 1 (Task 7)
    refine    — per-subject refinement, 10K iters (Task 8)
    export    — runtime asset serialization + S3 upload (Task 9)

Local-prep modules (data contract + ingest QA):
    schema    — FLAME-param on-disk schema (tracker output contract)
    ingest    — capture/ingest validator for portraits + reference video

Scaffold stage: subpackages above are stubs; implementations land with their tasks.
"""

__version__ = "0.1.0"

# Locked representation constants (docs/architecture.md "Key Representations").
FLAME_NUM_VERTS = 5143      # FLAME 2020 + patch_teeth
FLAME_NUM_FACES = 10144     # FLAME 2020 + patch_teeth
UV_GRID = 256               # flame_uv.npz 256x256 UV unwrap
NUM_GAUSSIANS = 58173       # |Omega| ~= valid-UV mask -> one Gaussian per valid UV pixel
MOTION_FPS = 25             # FLAME motioncode frame rate (ARTalk-style)
