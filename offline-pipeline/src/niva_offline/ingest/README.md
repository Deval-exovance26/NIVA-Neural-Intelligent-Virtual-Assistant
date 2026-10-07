# NIVA Capture / Ingest Validator

Validates the real NIVA subject capture assets (Task 5 in `docs/plan.md`) before
they feed the FLAME tracker (`GAGAvatar_track`). Runnable locally.

## What it checks

**Portraits** (`NIVA/images/*.png`, 5 x 1176x1337):
- resolution floor (>= 512x512),
- exactly one detectable face (OpenCV Haar cascade — presence/count, not identity),
- aspect-ratio consistency across the set (within 10% of the set median).

**Reference video** (`NIVA/video/NIVA_PSR.mp4`, 2160x2160, 30fps, ~104.5s):
- duration / fps / resolution sanity floors,
- a sampled-frame coverage heuristic: mean inter-frame motion energy and
  mouth-region temporal activity (a cheap proxy for expression/viseme variety).
  These are **reported, not hard-failed**.

## Usage

```bash
# from offline-pipeline/ with the package importable (PYTHONPATH=src)
python -m niva_offline.ingest.validate_capture \
  --images ../NIVA/images/*.png \
  --video  ../NIVA/video/NIVA_PSR.mp4
```

Requires `opencv-python-headless` and `numpy`.

## Manual audit already performed (authoritative)

A human review of the capture set has already been completed. Findings:

- **Identity consistency**: identity matches across all 5 portraits and the
  reference video. (The automated face check here is presence/count only — it is
  **not** an identity verifier.)
- **Expression / viseme coverage: GOOD** — the reference video covers neutral,
  smiles, open-mouth, and varied visemes. Sufficient for Stage-2 per-subject
  refinement.
- **Not confirmed (v2 nice-to-haves)**: deliberate **blinks** and **large head
  rotation** were not confirmed present. These improve eye/pose coverage but are
  not blockers for the current build; capture them in a v2 session if eye/pose
  fidelity needs lifting.

The automated validator exists to catch gross regressions (wrong resolution,
missing/multiple faces, corrupt or wrong-length video) on re-captures, and to
produce reproducible coverage numbers — it does not replace the manual audit
above.
