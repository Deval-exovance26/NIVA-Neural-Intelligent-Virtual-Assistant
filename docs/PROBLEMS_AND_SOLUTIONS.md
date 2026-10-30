# NIVA — Problems & Solutions Log

_A running log of concrete problems hit and their (known or candidate) solutions, so we don't
re-discover them. Newest first within each section._

---

## SOLVED

### P1. GAGAvatar plain-splat render looks blurry/washed-out in the browser
- **Cause:** GAGAvatar Gaussians carry a **32-channel neural feature**, not RGB. A `StyleUNet(in_dim=32,out_dim=3)` decodes it to the final image. The browser shows only the first 3 of 32 channels → washed out. (Confirmed in `gagavatar/models/GAGAvatar/models.py`.)
- **Solution:** Don't ship GAGAvatar's splat as-is. Build a **standalone SH splat** (GaussianAvatars-style) refined to look good through a plain rasterizer. Optionally **distil** the UNet's detail into the standalone splat offline (teacher→student). See plan_quality_fix.md.

### P2. Refinement v2/v3 got HIGHER PSNR but looked WORSE (blurry + hairline speckle)
- **Cause:** optimised raw per-Gaussian RGB toward the UNet's *output* (which is itself soft) and PSNR rewards average colour match, not sharpness.
- **Solution:** refine against the **real photos**, judge by **blind visual + metric** (never PSNR alone). Keep base as primary if refined isn't visibly better.

### P3. Standalone splat stayed BLURRY no matter how many Gaussians / how it was tuned (3 retunes, 10K–80K gaussians, all ~23.6 dB)
- **Cause (confirmed):** the frontal driver video `NIVA_PSR.mp4` has **zero head-pose variation** (global-rot std = 0.0 across all 3135 frames). No multi-view → 3DGS has no constraint to resolve sharp 3D detail; any soft frontal arrangement fits equally.
- **Solution:** add real multi-view. Tracked the **360 turntable** `Nidhi_Avatar.mp4` (358.8° yaw span) → S3 jumped to 28.5 dB and sharpened. Multi-view is MANDATORY for 3DGS sharpness.

### P4. GAGAvatar_track failed to read Nidhi_Avatar.mp4 ("No frames in the video")
- **Cause:** the mp4 container/codec (H.264 + AAC) tripped the tracker's reader.
- **Solution:** re-encode video-only, clean H.264: `ffmpeg -i in.mp4 -an -vcodec libx264 -pix_fmt yuv420p -r 30 out.mp4`. Then track `out.mp4`.

### P5. Standalone avatar rendered UPSIDE-DOWN in the browser
- **Cause:** exporter wrote GAGAvatar `+Y-up` positions; browser camera assumes `-Y-up` (hardcoded `UP=[0,-1,0]`).
- **Solution:** per-asset `flip_y` flag in the json meta; `loader.js` reads it, `app.js` sets camera up accordingly. Exporter has `--flip-y`.

### P6. Standalone avatar rendered SPIKY/SPECKLED in the browser (while fine on the box)
- **Cause:** `renderer_webgl2.js` did order-independent additive blending with NO depth sorting; our splat has HIGH opacity (~0.75) and a few oversized splats → wrong compositing + giant quads.
- **Solution:** added back-to-front **depth sorting** (throttled on camera move) + per-splat **screen-size clamp** in the shader; exporter `--scale-clip 0.04` caps world splat scale. (Deployed; frontal should now match the on-box gsplat render.)

### P7. Per-frame camera ignored (turntable rotation not honored)
- **Cause:** refinement used `rt.build_forward_batch(...)['t_transform']` = canonical frontal camera, ignoring each frame's `transform_matrix`.
- **Solution:** `refine_standalone.batch_for` now uses per-frame `e['transform_matrix']` when present (turntable/portraits). t_transform only as fallback.

---

## OPEN (candidate solutions noted)

### O1. Off-axis (±20–40°) views are streaky/speckled; frontal only is good
- **Cause (likely):** training was frontally biased (frontal holdout, portraits frontal-ish, verify rotated a frontal splat instead of rendering from true side cameras). Side Gaussians under-constrained. SH deg-3 overfits frontal → fringing at angle.
- **Candidate solution (plan_quality_fix.md):** full-360 fit from each frame's TRUE camera, uniform angle sampling, holdout across all yaws; SH **deg-1**; UNet distillation across angles. Confidence: good frontal ~80%, moderate yaw ~55%, full 360 ~30%.

### O2. Frontal is a bit soft (not crisp like Spatius)
- **Candidate:** UNet **distillation** (teacher = full GAGAvatar incl. UNet; student = plain SH splat) to bake neural detail into standalone Gaussians. Let scale shrink. Possibly more Gaussians via proper densification (gradient-based clone/split with binding inheritance).

### O3. Open-mouth shows messy mouth-interior (no teeth) — S4
- **Candidate:** rigid teeth triangles bound to the jaw. Borrow AGORA `flame_with_mouth` mesh or FLAME `patch_teeth` (5143 verts/10144 faces).

### O4. Identity source is a Veo AI-generated (synthetic, watermarked) turntable
- **Not a bug, a data caveat.** Consistent 360 coverage is ideal for training, but the avatar's identity = the synthetic video, not the real subject. Flag to stakeholders. If real identity is required, need a real multi-angle capture of the subject.

### O5. Package size vs rival (~5–10 MB)
- **Candidate (S5):** `.csplat` container: fp16 positions/scales, SH **deg-1**, int8 SH residuals, zlib. Prune low-opacity gaussians. ~160K gaussians → est. 6–12 MB; prune to <10 MB.

---

## INFRA GOTCHAS (environment)
- gsplat: install in GAGAvatar env; first `rasterization()` JIT-compiles CUDA (~107s). Needs
  `CUDA_HOME=/usr/local/cuda-12.8` and the env `bin` on PATH (for `ninja`).
- Standard INRIA `diff_gaussian_rasterization` is NOT installed; the repo on box is the 32-ch
  GAGAvatar variant. **gsplat is our SH backend.**
- Box has NO AWS creds (SSO IAM blocker) → do S3 from the workstation; scp between box↔workstation.
- stop/start WIPES NVMe (`/opt/dlami/nvme`); EBS root (`~`) persists. Keep artifacts in `~/niva-artifacts`.
- `ffmpeg` not on box PATH; use the bundled `imageio_ffmpeg` binary in the niva-offline env.
