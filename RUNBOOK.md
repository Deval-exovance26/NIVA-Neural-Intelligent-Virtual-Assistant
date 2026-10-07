# NIVA — EC2 Day-1 Runbook

Zero-friction startup for a **fresh EC2 g6e.4xlarge (NVIDIA L40S, 48GB, CUDA)** in
**us-east-2 (Ohio)**. The repo is code-only; weights and large assets come from
upstream / S3 (see `docs/architecture.md` → *Code vs Data Split*).

> **Nova Sonic lives in us-east-1** (`amazon.nova-2-sonic-v1:0` /
> `amazon.nova-2-5-sonic`). Build/data region is us-east-2 → the Arc B runtime makes
> a cross-region Bedrock call. Nova Sonic emits audio+text only (no visemes); FLAME
> motion is derived via the ARTalk-style audio→FLAME model.

## Prerequisites on the instance
- NVIDIA driver + CUDA 12.1-compatible runtime (Deep Learning AMI or driver installed).
- `conda` or `python3.10` + `pip`, `git`, `awscli` v2 configured (`aws configure` /
  instance role) with region `us-east-2`.
- Outbound internet for `pip` and upstream model repos.

## Steps

```bash
# 1. Clone the code (fast — no weights/datasets in git)
git clone <GITHUB_URL> && cd NIVA-Neural-Intelligent-Virtual-Assistant

# 2. Create env + install pinned CUDA 12.1 deps + editable package
make setup

# 3. Verify the GPU/CUDA stack (asserts torch.cuda.is_available(); prints driver/CUDA/GPU)
make verify-env

# 4. Fetch model weights into git-ignored offline-pipeline/checkpoints/
#    (GAGAvatar, GAGAvatar_track, ARTalk, DINOv3, gsplat).
#    FLAME 2020 is license-gated: supply your licensed copy via FLAME2020_SRC.
FLAME2020_SRC=/path/to/FLAME2020 ./scripts/fetch_models.sh

# 5. Stage capture assets to S3 (dry-run first, then confirm the upload)
./scripts/stage_assets_to_s3.sh                 # dry-run (prints what WOULD upload)
NIVA_CONFIRM=1 ./scripts/stage_assets_to_s3.sh  # actually upload to S3

# 6. Start Task 4 — FLAME tracking (GAGAvatar_track on NIVA/video/NIVA_PSR.mp4)
#    → per-frame flame_params.npz written via the niva_offline schema.
```

## What each step guarantees
| Step | Command | Guarantee |
|------|---------|-----------|
| 1 | `git clone` | Code, docs, env specs, scripts, 5 preview portraits. No large blobs. |
| 2 | `make setup` | Pinned `torch==2.3.1+cu121` / `torchvision==0.18.1+cu121` (L40S sm_89) + `pip install -e .`. |
| 3 | `make verify-env` | GPU visible, `torch.cuda.is_available()` true; prints driver/CUDA/GPU or exits non-zero. |
| 4 | `fetch_models.sh` | Weights in `offline-pipeline/checkpoints/`; FLAME 2020 only via `FLAME2020_SRC` (never auto-bypassed). |
| 5 | `stage_assets_to_s3.sh` | Capture assets in S3 (source of truth); dry-run unless `NIVA_CONFIRM=1`. |
| 6 | Task 4 tracking | First real `flame_params.npz` for the NIVA subject. |

## Validate ingest before/after tracking (optional)
```bash
cd offline-pipeline && python3 -m pytest tests/ -q   # schema + ingest + scaffold suites
```

## Pre-EC2 status (done locally)
Scaffold, pinned env, Makefile, fetch/stage scripts, FLAME param schema, capture
ingest validator, license audit, and docs are all committed. Only the live-GPU and
AWS steps (3–6) run on the instance. Full detail: `docs/plan.md` → *Local Prep
Status (pre-EC2)*.

## License flag (do not block, do gate launch)
GaussianAvatars is **Toyota non-commercial**. For the commercial product, use one of
the escape hatches before launch: reimplement the triangle-binding math (standard
FLAME math), or use FlashAvatar's permissive UV embedding. See `docs/LICENSES.md`.

## Cost control
On-demand GPU instance. Stop it when idle; restart and re-run `make verify-env`
before resuming. Co-locate Arc A build and Arc B runtime on the one instance.
