 # NIVA 3DGS Avatar — Implementation Plan (End-to-End with Nova Sonic)
  
  ## Problem Statement
  Build Calyza's NIVA: a real-time conversational digital employee with a photorealistic,
  animatable FLAME-mesh-bound 3D Gaussian head avatar, rendered in the browser (client GPU),
  driven end-to-end by Amazon Bedrock Nova Sonic speech-to-speech over WebRTC. Scope spans the
  SOW's 4-week avatar build AND the full Nova Sonic + WebRTC runtime. Backend never renders
  avatar video.
  
  ## Locked Requirements
  - **Pipeline (1=c):** Hybrid — GAGAvatar feed-forward base + GaussianAvatars-style per-subject
    refinement against the reference video (mirrors SpatialAvatar-0's two stages).
  - **License (2=a):** Proceed now with GAGAvatar/FLAME; one tracked task documents commercial-use
    restrictions + escape hatches; not a build blocker.
  - **Audio layer (2=b):** Derive FLAME motion from Nova Sonic audio via an ARTalk-style
    audio→FLAME model (Nova Sonic does not emit phoneme visemes).
  - **Scope (3=c):** Full end-to-end including Nova Sonic + WebRTC.
  - **Hardware:** Single EC2 g6e.4xlarge (L40S, 48GB), US East (Ohio), S3 + basic CloudWatch;
    AWS CLI configured. Avatar rendering is client-side.
  - **Performance:** Time-to-first-audio < 1.5s; FPS tiers High ~60 / Medium ~30 / Low 20–30;
    stable 30 beats unstable higher.
  
  ## Background (paper + repo research)
  - SpatialAvatar-0's own infrastructure = GAGAvatar lineage: FLAME params via `GAGAvatar_track`
    (EMICA-style); FLAME 2020 + `patch_teeth.py` (5143 verts / 10144 faces); `flame_uv.npz`
    256×256 UV unwrap; |Ω| ≈ 58,173 Gaussians; binding math from GaussianAvatars §3.2 (Eq. 1).
  - SpatialAvatar-0's novel feed-forward generator has NO public checkpoint → reuse GAGAvatar
    as the runnable feed-forward stand-in.
  - Audio→FLAME: `xg-chu/ARTalk` outputs FLAME motion at 25fps, real-time, same FLAME
    representation as GAGAvatar. Successor: `xg-chu/UniLS`.
  - Nova Sonic: `InvokeModelWithBidirectionalStream`, structured event sequence
    (completionStart → audio output); treat as audio-out, derive FLAME from audio.
  - License flag: GaussianAvatars = Toyota non-commercial. Mitigations: reimplement the
    triangle-binding math (standard FLAME math), or use FlashAvatar's permissive UV embedding
    for refinement.
  
  ## Proposed Solution
  Two arcs, each ending in a demoable increment. Arc A delivers a browser-rendered,
  FLAME-animatable NIVA avatar. Arc B wires the live Nova Sonic + WebRTC runtime onto it.
  Test-driven, incremental, no orphaned code — each task builds on the last and ends by
  integrating.
  
  ## Task Breakdown
  
  ### Arc A — Avatar Build
  - [ ] **Task 1: Project scaffold + EC2 g6e.4xlarge environment.** Repo structure
    (offline-pipeline / web-renderer / runtime-server); CUDA/PyTorch env on L40S; S3 bucket +
    CloudWatch log group; AWS CLI identity/region verified (read-only).
    *Test:* smoke-test asserts GPU visible, torch CUDA available, S3 round-trip.
    *Demo:* `make verify-env` prints GPU/driver/CUDA, confirms S3 + CloudWatch.
  - [ ] **Task 2: License & dependency audit (tracked, non-blocking).** Document licenses for
    FLAME, GAGAvatar, GAGAvatar_track, GaussianAvatars, ARTalk/UniLS, gsplat, DINOv3; mark
    commercial restrictions; record escape hatches.
    *Test:* `LICENSES.md` + CI check listing non-commercial deps. *Demo:* audit table.
  - [ ] **Task 3: FLAME foundation + asset acquisition.** Obtain FLAME 2020, teeth-patched mesh,
    `flame_uv.npz`, ~58K valid-UV mask; load + visualize neutral mesh.
    *Test:* assert 5143/10144 counts and ~58,173 UV pixels; render neutral mesh.
    *Demo:* parameter sweeps rendered to PNG.
  - [ ] **Task 4: FLAME tracking stage (GAGAvatar_track).** Run on reference video → per-frame
    FLAME params (shape/expr/pose/eye), camera, bbox, landmarks; stable on-disk schema.
    *Test:* shapes/ranges; mesh overlay on frames. *Demo:* video → `flame_params.npz` + overlay.
  - [ ] **Task 5: Capture spec + ingest the real NIVA subject.** Capture guide (portraits:
    front/±yaw/±pitch; video: neutral, phonemes, smile, mouth-open, blink, gaze, head motion)
    + ingest/QA tool.
    *Test:* validator checks resolution, face detect, length/fps. *Demo:* subject `flame_params.npz`.
  - [ ] **Task 6: Feed-forward base avatar (GAGAvatar).** Portrait(s) → ~58K FLAME-bound Gaussians;
    render under tracked params.
    *Test:* Gaussian count ≈ expected; PSNR/SSIM vs source above threshold.
    *Demo:* one portrait → animatable base at several poses.
  - [ ] **Task 7: Animation core — FLAME-bound Gaussian deformation (Eq. 1).** Per-triangle binding
    transform (world pos/rot/scale from parent triangle); freeze binding + count.
    *Test:* unit tests on binding math; expression visibly moves correct Gaussians.
    *Demo:* drive expression/jaw/eye; avatar animates offline.
  - [ ] **Task 8: Per-subject refinement (10K-iter, layout-preserving).** Refine Gaussian attributes
    vs reference video with mesh/binding/count frozen; anti-spike reg (scale-freeze warmup,
    log-scale clamp, screen-space anti-anisotropy) + dual-side random-bg compositing.
    *Test:* held-out PSNR/SSIM up vs base; max aspect ratio bounded (~8).
    *Demo:* before/after on held-out frames.
  - [ ] **Task 9: Export runtime avatar asset.** Serialize refined Gaussians + binding tables +
    FLAME metadata to browser-loadable format; upload to S3.
    *Test:* round-trip loader matches offline renders; size within budget.
    *Demo:* versioned asset in S3.
  - [ ] **Task 10: Browser WebGPU splat renderer (static).** Load asset, render neutral pose on
    client GPU (WebGPU; Three.js host).
    *Test:* headless render matches offline neutral within tolerance; FPS counter.
    *Demo:* page renders NIVA's head interactively.
  - [ ] **Task 11: Browser-side FLAME animation.** Apply FLAME expr/pose/eye per-frame on client to
    deform Gaussians (GPU binding transform); play a tracked FLAME sequence.
    *Test:* in-browser frames match offline within tolerance; sustained FPS.
    *Demo:* browser plays tracked sequence — NIVA talks/blinks/moves client-side.
    **(End of Arc A.)**
  
  ### Arc B — Nova Sonic Live Runtime
  - [ ] **Task 12: Audio→FLAME driver (ARTalk-style).** Audio → 25fps FLAME motioncode
    (expr/jaw/pose/eye); resample/align to renderer frame clock.
    *Test:* known clip → correct shape/fps + plausible lipsync via Task 11.
    *Demo:* audio file → browser lip-sync.
  - [ ] **Task 13: Nova Sonic session service.** EC2 backend opens
    `InvokeModelWithBidirectionalStream`, handles event sequence, exposes clean audio+events stream.
    *Test:* scripted turn asserts audio bytes + event ordering; measure TTFA.
    *Demo:* prompt in → Nova Sonic audio out, TTFA logged (<1.5s).
  - [ ] **Task 14: WebRTC transport.** Single tester: mic up; Nova Sonic audio + derived FLAME
    params down; data channel for FLAME frames alongside audio.
    *Test:* loopback latency; FLAME frames time-aligned. *Demo:* speak; round-trip audio over WebRTC.
  - [ ] **Task 15: End-to-end integration + lip-sync alignment.**
    Mic→WebRTC→Nova Sonic→audio→ARTalk FLAME→data channel→browser→speaker; A/V sync buffer.
    *Test:* automated turn asserts synced audio + FLAME animation; measure sync offset + TTFA.
    *Demo:* developer talks to NIVA; synced speech, lip-sync, blinks, expressions, head/eye motion.
  - [ ] **Task 16: Performance tiering + runtime QA.** High/Medium/Low tiers (Gaussian LOD /
    resolution / frame pacing); WebGPU→WebGL2 fallback; stability under sustained conversation.
    *Test:* automated FPS/stability per tier meets SOW targets. *Demo:* tiers + graceful fallback.
  - [ ] **Task 17: Deploy, cost, and ops.** Deploy runtime to EC2 (API/WebRTC + Nova Sonic
    co-located); CloudWatch logging; documented stop/start for GPU cost; cost note vs SOW.
    *Test:* fresh-deploy smoke test passes end-to-end; logs flow to CloudWatch.
    *Demo:* clean deployment + runbook.
    **(End of Arc B: full NIVA.)**
  
  ## Open Options for Review
  - Insert an explicit multi-view capture task before Task 8 (GaussianAvatars-style refinement
    benefits from multi-view; the paper does monocular).
  - Evaluate `UniLS` (newer ARTalk successor) alongside ARTalk in Task 12.