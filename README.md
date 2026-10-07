# NIVA — Neural Intelligent Virtual Assistant

NIVA is Calyza Tech LLP's real-time conversational **digital employee**: a
photorealistic, animatable FLAME-mesh-bound **3D Gaussian Splatting** head avatar,
rendered entirely **client-side in the browser (WebGPU)** and driven end-to-end by
**Amazon Bedrock Nova Sonic** speech-to-speech over **WebRTC**. The backend never
renders avatar video — the user's GPU performs all splat rendering, while the server
streams audio, text, and derived FLAME motion params down to the client.

---

## Architecture at a glance

**Offline avatar build (Arc A, on EC2 g6e.4xlarge):**

```
Portraits + Reference video
        │
        ▼ GAGAvatar_track ──► per-frame FLAME params (shape/expr/pose/eye), camera, bbox, landmarks
        │
        ▼ GAGAvatar feed-forward ──► ~58,173 FLAME-bound Gaussians (base avatar)
        │
        ▼ Per-subject refinement (GaussianAvatars-style, 10K iters, binding + count frozen, anti-spike reg)
        │
        ▼ Export: packed Gaussians + binding tables + FLAME metadata ──► S3
        │
        ▼ Browser WebGPU renderer loads the exported asset
```

**Live runtime (Arc B):**

```
Mic ─► WebRTC ─► Nova Sonic (InvokeModelWithBidirectionalStream)
                      │
                      ├─► response audio ──────────────► Speaker (client)
                      │        │
                      │        ▼ Audio→FLAME (ARTalk-style, 25fps: expr/jaw/pose/eye)
                      │        │
                      │        ▼ FLAME param frames ─(WebRTC data channel)─► browser
                      ▼
             Browser WebGPU splat renderer applies the FLAME binding transform per-frame on the GPU;
             an A/V sync buffer aligns lips to the audio.
```

See [`docs/architecture.md`](docs/architecture.md) for the full component lineage,
representations, deployment topology, and region split.

---

## Pipeline summary

NIVA uses a **hybrid** reconstruction approach over a single shared
**FLAME-mesh-bound Gaussian representation**:

1. **Feed-forward base** — `GAGAvatar` produces a ~58K FLAME-bound Gaussian avatar
   from the subject's portrait(s) in real time (Stage 1 stand-in).
2. **Per-subject refinement** — a `GaussianAvatars`-style optimization refines the
   Gaussian attributes against the reference video, with the triangle binding and
   Gaussian count frozen (Stage 2 lineage).

This two-stage design mirrors **SpatialAvatar-0**, which is the methodological
reference for the project. The SpatialAvatar-0 analysis and full research notes live
in [`research/AVATAR.md`](research/AVATAR.md); SpatialAvatar-0's own novel generator
has no public checkpoint, so GAGAvatar is used as the runnable feed-forward stand-in.

Key representations: FLAME 2020 (teeth-patched → 5143 verts / 10144 faces), a
256×256 UV layout with a valid-UV mask of ≈ 58,173 pixels (one Gaussian per valid UV
pixel), and FLAME motion at 25fps (expression ∈ ℝ¹⁰⁰, pose ∈ ℝ⁶, eye ∈ ℝ⁶).

---

## Repository layout

| Path | What it is |
|------|------------|
| [`offline-pipeline/`](offline-pipeline/) | Arc A avatar build (GPU, non-interactive). `src/niva_offline` package: `flame/ tracking/ base/ binding/ refine/ export/ schema/ ingest/`. Pinned CUDA 12.1 env, Dockerfile, tests. |
| [`web-renderer/`](web-renderer/) | Browser-side WebGPU Gaussian splat renderer (Three.js host). Loads the exported asset and applies the FLAME binding transform per-frame on the client GPU. *Scaffold only.* |
| [`runtime-server/`](runtime-server/) | Arc B live runtime: Nova Sonic session service + WebRTC transport + ARTalk-style audio→FLAME driver. *Scaffold only.* |
| [`scripts/`](scripts/) | `fetch_models.sh` (idempotent model/weight fetcher) and `stage_assets_to_s3.sh` (capture-asset upload to S3). |
| [`docs/`](docs/) | [`plan.md`](docs/plan.md) (full task breakdown), [`architecture.md`](docs/architecture.md), [`LICENSES.md`](docs/LICENSES.md). |
| [`research/`](research/) | Methodology references — [`AVATAR.md`](research/AVATAR.md) (SpatialAvatar-0 analysis) and the Calyza NIVA Avatar SOW. |
| `NIVA/` | Reference capture assets (portraits, reference video, test audio). **Large media are git-ignored and live in S3**; only the 5 small portrait PNGs are kept as a lightweight preview. |
| [`RUNBOOK.md`](RUNBOOK.md) | Authoritative EC2 day-1 startup runbook. |
| [`Makefile`](Makefile) | Top-level entry points (see below). |

---

## Prerequisites

- **Compute:** single EC2 **g6e.4xlarge** (NVIDIA **L40S, 48GB**), with CUDA **12.x**
  driver/runtime (Deep Learning AMI or equivalent).
- **Offline pipeline:** **Python 3.10** (conda env `niva-offline`), pinned
  `torch==2.3.1+cu121` / `torchvision==0.18.1+cu121` for L40S (sm_89).
- **Web renderer:** **Node** + a modern WebGPU-capable browser (desktop Chrome).
- **AWS account** with the CLI configured:
  - Build + data region: **us-east-2** (US East / Ohio) — EC2, S3, CloudWatch.
  - **Nova Sonic region: us-east-1** (N. Virginia), live as
    `amazon.nova-2-sonic-v1:0` / `amazon.nova-2-5-sonic`. The Arc B runtime makes a
    cross-region Bedrock call to us-east-1.

---

## Quick start

[`RUNBOOK.md`](RUNBOOK.md) is the authoritative EC2 day-1 guide. In short, on a fresh
g6e.4xlarge:

```bash
git clone <GITHUB_URL> && cd NIVA-Neural-Intelligent-Virtual-Assistant
make setup                        # create the conda env + install the package (editable)
make verify-env                   # assert GPU visible + torch.cuda.is_available(); print driver/CUDA/GPU
FLAME2020_SRC=/path/to/FLAME2020 ./scripts/fetch_models.sh   # fetch weights (FLAME 2020 is license-gated)
./scripts/stage_assets_to_s3.sh   # dry-run; then NIVA_CONFIRM=1 ./scripts/stage_assets_to_s3.sh to upload
# then start Task 4 — FLAME tracking (GAGAvatar_track on NIVA/video/NIVA_PSR.mp4)
```

See [`RUNBOOK.md`](RUNBOOK.md) for per-step guarantees and troubleshooting.

---

## Make targets

| Target | Description |
|--------|-------------|
| `make setup` | Create the `niva-offline` conda env (Python 3.10, CUDA 12.1 torch) and `pip install -e` the package. |
| `make verify-env` | Assert the GPU is visible and `torch.cuda.is_available()`; print driver/CUDA/GPU. |
| `make fetch-models` | Download FLAME/GAGAvatar/ARTalk weights via `scripts/fetch_models.sh`. |
| `make stage-assets` | Upload NIVA capture assets to S3 via `scripts/stage_assets_to_s3.sh`. |

Additional helpers: `make test` (offline-pipeline unit tests, CPU-only) and
`make docker-build` (CUDA 12.x offline-pipeline image).

---

## Status

**Local pre-EC2 preparation is complete.** Committed and ready in the repo:

- Repo scaffold (`offline-pipeline` / `web-renderer` / `runtime-server`, each with a role README).
- Pinned CUDA 12.1 environment (`requirements.txt` / `environment.yml` / `Dockerfile` / `pyproject.toml`) and the top-level `Makefile`.
- `scripts/fetch_models.sh` and `scripts/stage_assets_to_s3.sh`.
- Capture **ingest/QA validator** (`niva_offline.ingest.validate_capture`) and the stable **FLAME param schema** (`niva_offline.schema.flame_params`: `FlameFrameParams` + `FramesManifest`, NPZ/JSON round-trip).
- License audit ([`docs/LICENSES.md`](docs/LICENSES.md)) and docs (`plan.md`, `architecture.md`).
- Offline-pipeline unit tests (scaffold + schema + ingest suites).

**Next:** EC2 spin-up + model training — live `make verify-env` GPU/CUDA assertion,
S3/CloudWatch setup, model-weight fetches, the licensed FLAME 2020 download, and the
first FLAME tracking run. The full task breakdown (Arc A avatar build, Arc B Nova
Sonic runtime) is in [`docs/plan.md`](docs/plan.md).

---

## Licensing

A full dependency license audit lives in [`docs/LICENSES.md`](docs/LICENSES.md). The
single launch-gating flag is **GaussianAvatars**, which is released under a **Toyota
Motor Europe non-commercial** research license — it must **not** ship in the
commercial product. NIVA's dependency on it is narrow (the standard triangle-binding
math), and two documented mitigations exist: reimplement the binding transform from
standard FLAME mesh geometry, or use FlashAvatar's permissive UV embedding for
refinement. This is a tracked risk, **not a build blocker**.

---

## Credits / methodology references

NIVA builds on the following upstream work:

- [`xg-chu/GAGAvatar`](https://github.com/xg-chu/GAGAvatar) — feed-forward FLAME-bound Gaussian avatar (base stage).
- [`xg-chu/GAGAvatar_track`](https://github.com/xg-chu/GAGAvatar_track) — FLAME tracker/preprocessor.
- [`xg-chu/ARTalk`](https://github.com/xg-chu/ARTalk) — audio → FLAME motion (25fps) driver.
- [`ShenhanQian/GaussianAvatars`](https://github.com/ShenhanQian/GaussianAvatars) — per-subject FLAME-bound Gaussian refinement + triangle-binding math (*Toyota non-commercial*).
- [FLAME](https://flame.is.tue.mpg.de/) — parametric head model (the rig).
- [`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3) — frozen image feature extractor (reference component).
- [`nerfstudio-project/gsplat`](https://github.com/nerfstudio-project/gsplat) — Gaussian rasterization backend (offline/validation).

**SpatialAvatar-0** is the methodological reference for the overall two-stage design;
see [`research/AVATAR.md`](research/AVATAR.md).

---

*© Calyza Tech LLP. See [`docs/LICENSES.md`](docs/LICENSES.md) for third-party terms.*
