 # NIVA 3DGS Avatar — Architecture
  
  ## System Overview
  NIVA = animatable FLAME-mesh-bound 3D Gaussian head avatar, rendered client-side in the
  browser, driven end-to-end by Amazon Bedrock Nova Sonic over WebRTC. The backend never renders
  avatar video; the client GPU performs all splat rendering.
  
  ## Component Lineage (what each repo is)
  - **FLAME** (`TimoBolkart/TF_FLAME`): parametric head model = the rig. Params: shape, expression,
    pose (head+jaw), eye. Control signal for the whole system.
  - **GAGAvatar_track** (`xg-chu/GAGAvatar_track`): FLAME tracker/preprocessor. Video/image →
    per-frame FLAME params + camera + bbox + landmarks. No rendering. (= paper's tracking stage.)
  - **GAGAvatar** (`xg-chu/GAGAvatar`): feed-forward FLAME-bound Gaussian avatar from one portrait,
    real-time, pretrained weights. (≈ SpatialAvatar-0 Stage 1; the runnable stand-in.)
  - **GaussianAvatars** (`ShenhanQian/GaussianAvatars`): per-subject optimized FLAME-bound Gaussian
    avatar from multi-view video; source of the triangle-binding math (Eq. 1). (≈ Stage 2 lineage.)
    License: Toyota non-commercial — flagged.
  - **DINOv3** (`facebookresearch/dinov3`): frozen image feature extractor; internal component of a
    from-scratch SpatialAvatar-0 generator (not built here).
  - **gsplat** (`nerfstudio-project/gsplat`): Gaussian rasterization backend (offline/validation).
  - **ARTalk** (`xg-chu/ARTalk`): audio → FLAME motion (25fps), real-time; bridge to Nova Sonic.
    Successor: `UniLS`.
  
  ## Data Flow — Offline Avatar Build (Arc A, on EC2 g6e.4xlarge)
  
  Portraits + Reference video       │       ▼ GAGAvatar_track ──► per-frame FLAME params
  (shape/expr/pose/eye), camera, bbox, landmarks       │       ▼ GAGAvatar feed-forward ──► ~58K
  FLAME-bound Gaussians (base avatar)       │       ▼ Per-subject refinement (10K iters, binding+count
  frozen, anti-spike reg)       │       ▼ Export: packed Gaussians + binding tables + FLAME metadata ──►
  S3
  
  
  ## Data Flow — Live Runtime (Arc B)
  
  Mic ─► WebRTC ─► Nova Sonic (InvokeModelWithBidirectionalStream)                       │
  ├─► response audio ─────────────► Speaker (client)                       │                 │
  │                 ▼                       │        Audio→FLAME (ARTalk-style, 25fps: expr/jaw/pose/eye)
  │                 │                       │                 ▼                       │        FLAME param
  frames ─(data channel)─► Browser                       ▼           Browser WebGPU splat renderer
  • loads exported avatar asset             • applies FLAME binding transform per-frame on GPU
  • A/V sync buffer aligns lips to audio                       │                       ▼
  NIVA avatar
  
  
  ## Key Representations
  - **FLAME mesh:** FLAME 2020 + patch_teeth → 5143 vertices / 10144 faces.
  - **UV layout:** `flame_uv.npz`, 256×256 grid, valid-UV mask |Ω| ≈ 58,173 → one Gaussian per
    valid UV pixel.
  - **Binding (Eq. 1):** each Gaussian rigidly bound to a parent FLAME triangle; world attributes
    from the triangle's local frame (center, rotation, scale). Binding + count frozen through
    refinement and runtime.
  - **Motion signal:** FLAME expression (ψ∈R100), pose (R6: head+jaw), eye (R6), at 25fps.
  
  ## Deployment Topology
  - **Single EC2 g6e.4xlarge (L40S 48GB, US East/Ohio):** avatar build (Arc A) AND runtime
    API/WebRTC + Nova Sonic session service (Arc B), co-located.
  - **S3:** source footage, tracked params, exported avatar assets.
  - **CloudWatch:** basic logging.
  - **Client browser:** all avatar rendering (WebGPU, Three.js host; WebGL2 fallback).
  - **Cost control:** on-demand instance; documented stop/start when idle.
  
  ## Region Topology (Nova Sonic cross-region)
  - **Build + data region = us-east-2 (Ohio, SOW):** EC2 g6e.4xlarge, S3 buckets,
    CloudWatch all live here.
  - **Nova Sonic region = us-east-1 (N. Virginia):** confirmed live as
    `amazon.nova-2-sonic-v1:0` / `amazon.nova-2-5-sonic`. The Arc B runtime server
    (in us-east-2) therefore makes a **cross-region Bedrock call to us-east-1** for
    the `InvokeModelWithBidirectionalStream` speech-to-speech session. Pin this
    endpoint region in config and account for the small added RTT in the
    time-to-first-audio budget (<1.5s).
  - **No visemes:** Nova Sonic returns audio + text only; FLAME lip/jaw/pose/eye
    motion is derived from the audio via the ARTalk-style audio→FLAME model (25fps).
  
  ## Code vs Data Split (GitHub + S3 workflow)
  The repo is deliberately a **code-only** artifact so a fresh EC2 clone is fast and
  reproducible; all heavy bytes live in S3 or are fetched on-instance.
  - **GitHub (git) holds:** source (`offline-pipeline/`, `web-renderer/`,
    `runtime-server/`), pinned env specs, `Makefile`, scripts, docs, tests, and the
    5 small portrait PNGs as a lightweight preview. No weights, datasets, or large
    media (enforced by `.gitignore`).
  - **S3 holds (source of truth for data/weights):** reference footage
    (`NIVA_PSR.mp4`, 534MB), capture portraits/audio, tracked `flame_params.npz`,
    and exported avatar assets. Capture assets are pushed via
    `scripts/stage_assets_to_s3.sh` (dry-run by default; `NIVA_CONFIRM=1` to
    upload); model weights are pulled on-instance via `scripts/fetch_models.sh`
    into git-ignored `offline-pipeline/checkpoints/`.
  - **EC2 workflow:** `git clone` (code) → `make setup`/`verify-env` →
    `fetch_models.sh` (weights + FLAME 2020 via `FLAME2020_SRC`) →
    `stage_assets_to_s3.sh` (capture assets ↔ S3) → build. See `RUNBOOK.md`.
  
  ## Performance Targets
  - Time-to-first-audio < 1.5s.
  - FPS tiers: High ~60 / Medium ~30 / Low 20–30. Stable 30 preferred over unstable higher.
  - Refinement sanity: max Gaussian aspect ratio bounded (~8) via anti-spike regularization.
  
  ## Known Risks / Flags
  - GaussianAvatars non-commercial license (mitigations: reimplement binding math; or FlashAvatar
    UV embedding).
  - SpatialAvatar-0 novel generator unavailable (mitigated by GAGAvatar stand-in).
  - Nova Sonic emits no visemes (confirmed): audio+text only — mitigated by the
    ARTalk-style audio→FLAME derivation (Task 12). Nova Sonic lives in us-east-1
    (`amazon.nova-2-sonic-v1:0` / `amazon.nova-2-5-sonic`) → cross-region call from
    the us-east-2 runtime; keep RTT inside the <1.5s TTFA budget.
  
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
  
  Tell me how you'd like to proceed:
  - "go" / "implement" → I hand off to execution; its first action is to create plan.md and
  architecture.md, then wait for your review before coding.
  - Or give me edits first (e.g. add the multi-view capture task, include UniLS evaluation), and I'll
  revise before handoff.
  
  And noted as a standing preference: whenever you ask me for a plan, produce a .md file for it.