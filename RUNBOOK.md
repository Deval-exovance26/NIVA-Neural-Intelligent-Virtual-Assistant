# NIVA — EC2 Day-1 Runbook

Zero-friction startup for a **fresh EC2 g6e.4xlarge (NVIDIA L40S, 48GB, CUDA)** in
**ap-south-1 (Asia Pacific / Mumbai)**. The repo is code-only; weights and large assets come from
upstream / S3 (see `docs/architecture.md` → *Code vs Data Split*).

> **Nova Sonic lives in us-east-1** (`amazon.nova-2-sonic-v1:0` /
> `amazon.nova-2-5-sonic`). Arc A (avatar build) runs in **ap-south-1**. The
> Arc B runtime makes a **cross-region Bedrock call** from ap-south-1 to us-east-1
> (added RTT must stay within the <1.5s time-to-first-audio budget). Nova Sonic
> emits audio+text only (no visemes); FLAME motion is derived via the ARTalk-style
> audio→FLAME model. Nova Sonic is NOT invoked during Arc A.

## Prerequisites on the instance
- NVIDIA driver + CUDA 12.1-compatible runtime (Deep Learning AMI or driver installed).
- `conda` or `python3.10` + `pip`, `git`, `awscli` v2 configured (`aws configure` /
  instance role) with region `ap-south-1`.
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

## Cost control & overnight stop/start (resume where you left off)

The GPU is the dominant cost (~$3.00/hr for the g6e.4xlarge). **Stop the instance
every night and restart the next morning — all work persists.** This is designed
to be a zero-loss "pick up exactly where you left off" workflow.

### What survives a STOP (and what does NOT)
| Survives a `stop` ✅ | Wiped on `stop` ❌ |
|----------------------|---------------------|
| **Root EBS volume (100GB)** — conda env, NVIDIA driver, fetched weights, FLAME 2020 | **Instance-store / ephemeral NVMe** (~600GB on g6e) — scratch + S3 working copy |
| Everything in **S3** (source of truth: tracked params, checkpoints, exports) | **RAM / running processes** (a stop kills in-flight jobs) |
| | The **public IPv4** (changes on restart — no Elastic IP in use; grab the new IP each start) |

> ⚠️ **Never leave the only copy of durable work on NVMe.** It is erased on every
> stop. Durable artifacts go to **S3** (via `sync_outputs_to_s3.sh`); the env/weights
> live on **EBS**. NVMe is for regenerable scratch + the fetched S3 working copy.

### Durability model (hybrid: 100GB EBS env + S3 source-of-truth + NVMe scratch)
1. **EBS root (100GB, encrypted) = persistent env disk.** Holds the OS, NVIDIA
   driver, conda env, model weights, and FLAME files. Survives stop/start, so
   next-morning resume needs **no re-setup / no env rebuild**. ~$6/mo while stopped.
2. **S3 = source of truth for all data/artifacts.** After each stage, push outputs
   to `s3://niva-avatar-assets-508137168365-aps1/`. Bucket has **versioning** +
   **SSE-S3 encryption**. A lost volume costs only recompute time, never data.
3. **NVMe instance-store (~600GB, ephemeral) = fast scratch + S3 working copy.**
   Decoded frames, temp tensors, render dumps, and the S3 data copy live here.
   **Wiped on every stop** — regenerable/re-fetchable, so that's fine.

### Scripts (one command each way)
- `sudo bash scripts/setup_nvme_scratch.sh` — format (if blank) + mount the NVMe at
  `/mnt/nvme` with `data/ frames/ tmp/ renders/`. Run **after each start** (NVMe is
  blank on boot).
- `bash scripts/sync_from_s3.sh` — **start of day:** pull stage artifacts
  S3 → `/mnt/nvme/data/`.
- `NIVA_CONFIRM=1 bash scripts/sync_outputs_to_s3.sh` — **end of day:** push outputs
  → S3 before stopping (run without `NIVA_CONFIRM` first for a dry-run plan).

### S3 artifact layout (staged after each stage)
```
s3://niva-avatar-assets-508137168365-aps1/
  raw-capture/v1/            # DONE — source footage, portraits, audio (staged day 1)
  tracking/v1/               # Task 4  — flame_params.npz + overlays
  base/v1/                   # Task 6  — feed-forward base Gaussians
  refine/v1/checkpoints/     # Task 8  — periodic refinement checkpoints (resumable)
  refine/v1/final/           # Task 8  — refined Gaussians
  export/v1/                 # Task 9  — browser-loadable avatar asset
```

### Nightly STOP procedure
```bash
# 1. Checkpoint any in-flight long run (refinement auto-checkpoints; else finish a step).
# 2. Push the current stage's outputs to S3 (source of truth):
NIVA_CONFIRM=1 bash scripts/sync_outputs_to_s3.sh
# 3. Stop the instance (from your laptop or the console) — NOT terminate:
aws ec2 stop-instances --instance-ids <INSTANCE_ID> --region ap-south-1
```

### Morning START procedure
```bash
# 1. Start the instance:
aws ec2 start-instances --instance-ids <INSTANCE_ID> --region ap-south-1
# 2. Get the new public IP (changes each start — no Elastic IP in use):
aws ec2 describe-instances --instance-ids <INSTANCE_ID> --region ap-south-1 \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text
# 3. SSH in, then re-mount NVMe (blank on boot) and repopulate scratch from S3:
sudo bash scripts/setup_nvme_scratch.sh
bash scripts/sync_from_s3.sh
# 4. Re-assert the GPU stack (env already on EBS, so this is instant):
make verify-env
# 5. Resume from the last completed stage / last refinement checkpoint. The conda
#    env, weights, and FLAME files are still on EBS — nothing to reinstall.
```

> **STOP ≠ lose work. TERMINATE = lose the instance + its EBS (unless retained).**
> Only ever **stop** overnight. Reserve **terminate** for when the project is fully
> done and all artifacts are safely in S3.

**Note:** no Elastic IP is used (by choice), so the instance's public IP **changes on
every start**. Grab the new IP from the START procedure's `describe-instances` call
each morning and reconnect SSH to it. (If the daily IP churn becomes annoying, an
Elastic IP would pin it — small charge while stopped.)

## License flag (do not block, do gate launch)
GaussianAvatars is **Toyota non-commercial**. For the commercial product, use one of
the escape hatches before launch: reimplement the triangle-binding math (standard
FLAME math), or use FlashAvatar's permissive UV embedding. See `docs/LICENSES.md`.
