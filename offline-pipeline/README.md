# offline-pipeline/

The **avatar build** (Arc A) of NIVA. All GPU-heavy, non-interactive reconstruction
code lives here. Runs on the single EC2 **g6e.4xlarge (L40S 48GB, CUDA 12.x)** in
`us-east-2` (US East / Ohio). The backend never renders avatar video; this stage only
produces the exported, browser-loadable avatar asset.

## Role (per docs/architecture.md)

This package implements the offline data flow:

```
Portraits + Reference video
        │
        ▼ GAGAvatar_track ──► per-frame FLAME params (shape/expr/pose/eye), camera, bbox, landmarks
        │
        ▼ GAGAvatar feed-forward ──► ~58,173 FLAME-bound Gaussians (base avatar)
        │
        ▼ Per-subject refinement (10K iters, binding + count frozen, anti-spike reg)
        │
        ▼ Export: packed Gaussians + binding tables + FLAME metadata ──► S3
```

## Component lineage (what each dependency is)

| Component | Repo | Role |
|-----------|------|------|
| FLAME 2020 | `TimoBolkart/TF_FLAME` | Parametric head rig. Teeth-patched → 5143 verts / 10144 faces. |
| GAGAvatar_track | `xg-chu/GAGAvatar_track` | FLAME tracker: video/image → per-frame FLAME params + camera + bbox + landmarks. |
| GAGAvatar | `xg-chu/GAGAvatar` | Feed-forward FLAME-bound Gaussian avatar from one portrait (base / Stage 1). |
| GaussianAvatars | `ShenhanQian/GaussianAvatars` | Per-subject optimized avatar; source of triangle-binding math (Eq. 1). **Toyota non-commercial — flagged.** |
| gsplat | `nerfstudio-project/gsplat` | Gaussian rasterization backend (offline / validation). |
| ARTalk | `xg-chu/ARTalk` | Audio → FLAME motion (25fps). Bridge to Nova Sonic (consumed in runtime-server). |
| DINOv3 | `facebookresearch/dinov3` | Frozen feature extractor (not built here; reference only). |

## Key representations

- **FLAME mesh:** FLAME 2020 + `patch_teeth.py` → 5143 vertices / 10144 faces.
- **UV layout:** `flame_uv.npz`, 256×256 grid, valid-UV mask |Ω| ≈ 58,173 → one Gaussian per valid UV pixel.
- **Binding (Eq. 1):** each Gaussian rigidly bound to a parent FLAME triangle; binding + count frozen through refinement and runtime.
- **Motion signal:** FLAME expression (ψ∈R100), pose (R6: head+jaw), eye (R6), at 25fps.

## Layout

```
offline-pipeline/
├── pyproject.toml        # package metadata + build config (src/ layout)
├── requirements.txt      # PINNED pip deps (CUDA 12.1 torch for L40S/Ada)
├── environment.yml       # conda mirror of requirements (python 3.10)
├── Dockerfile            # CUDA 12.x devel image, reproducible build on EC2
└── src/niva_offline/     # the Python package
    ├── flame/            # FLAME foundation + asset loading (Task 3)
    ├── tracking/         # GAGAvatar_track wrappers (Task 4)
    ├── base/             # GAGAvatar feed-forward base avatar (Task 6)
    ├── binding/          # FLAME-bound Gaussian deformation, Eq. 1 (Task 7)
    ├── refine/           # per-subject refinement (Task 8)
    └── export/           # runtime asset serialization + S3 upload (Task 9)
```

## Setup on a fresh EC2 g6e.4xlarge

From the repo root (see the top-level `Makefile`):

```bash
make setup        # create the conda env (or: docker build this folder)
make verify-env   # assert GPU visible + torch.cuda.is_available() + print driver/CUDA
make fetch-models # download FLAME/GAGAvatar/ARTalk weights (scripts/fetch_models.sh)
make stage-assets # push NIVA capture assets to S3 (scripts/stage_assets_to_s3.sh)
```

Two reproducible paths are provided and kept in sync:
1. **conda/pip** — `make setup` builds `environment.yml` + `requirements.txt`.
2. **Docker** — `docker build -t niva-offline .` from this directory.

> Model weights and datasets are **never** committed to git. They are fetched on the
> instance via `scripts/fetch_models.sh` and staged to S3 via `scripts/stage_assets_to_s3.sh`.

## License note (non-blocking, tracked)

GaussianAvatars is **Toyota non-commercial**. NIVA is a commercial product. This is a
flagged, tracked risk — not a build blocker. Mitigations (per docs/plan.md Task 2):
reimplement the standard FLAME triangle-binding math from scratch, or use FlashAvatar's
permissive UV embedding for refinement. The full audit lives in `LICENSES.md` (other stage).
