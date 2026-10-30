# NIVA — Avatar Quality Fix Plan (standalone splat → Spatius-like)

_Date: 2026-10-12 (end of day) · Status: PLAN for next session · Target: app.spatius.ai quality bar_

> **Goal.** A beautiful, lightweight, **standalone** (no server neural net), FLAME-driven
> **interactive talking** avatar of the subject — rendered client-side — matching the
> Spatius bar (on-device 3DGS, ~5–10 MB, SH colour, teeth). NOT a 360 turntable; the
> turntable video is only multi-view *training* data.

---

## 0. Current state (what works / what doesn't)

- **Works:** S1 FLAME triangle binding (+13 tests); S2 standalone SH splat renders via gsplat
  (no neural net); S3 multi-view refinement vs the 360 turntable → **frontal view is good**
  (sharp, identity-preserved); animation proven (jaw/mouth deform via FLAME params).
- **Doesn't work yet:** off-axis (±20–40°) views are **streaky / speckled / colour-fringed**;
  frontal is good but slightly soft; browser needed a renderer fix (done: depth-sort + scale-clip).
- **Verified ground truth:** `~/_verify_avatar.png` on the box — centre (frontal) good, sides bad.

## 1. Root-cause diagnosis (why it's not Spatius-good yet)

1. **GAGAvatar splat is low-quality by design.** It's a one-shot feed-forward splat whose detail
   is produced by a **neural UNet** at render time. Its raw Gaussians were never meant to look
   good alone — so our standalone splat is inherently soft. (Confirmed in code: Gaussians carry a
   32-ch feature, `StyleUNet(in_dim=32,out_dim=3)` decodes to RGB.)
2. **Our multi-view fit was frontally biased + under-trained.** Frontal holdout, portraits at
   frontal-ish angles, and the verify spun a frontal splat instead of using true side cameras →
   side Gaussians never properly constrained → streaks at angle.
3. **SH degree-3 overfits frontal views** → rainbow fringing at off-axis angles.
4. **(fixed)** Browser WebGL2 did no depth sorting → high-opacity splats composited wrong → spiky.

## 2. The fix (ordered; de-risked)

### STEP 0 — 1-hour DISTILLATION SPIKE  ✅ DONE 2026-10-13 — RESULT: **GO**
Prove the quality jump is real before committing 4–7 hrs.
- Render the **full GAGAvatar (splat + UNet)** offline on ~20 frontal turntable frames →
  sharp "teacher" images of her.
- Fine-tune the existing S3 splat (frontal only) to match those teacher images (L1 + a little
  perceptual) for ~1–2k iters.
- **Gate:** if the frontal splat visibly sharpens toward the UNet quality → commit to full plan.

**OUTCOME (verified):** Ran on the box (L40S). Teacher = `rt.render_rgb_batch(...)['sr_gen_image']`
(splat+StyleUNet) via a tracked-avatar dict built from `images/optim.pkl` `front.png`
(no `tracked.pt` needed). Student = `s3_turntable` ckpt (159,616 gaussians, SH deg-3) rendered
through gsplat. Distilled student SH+opacity+scale toward teacher, L1, 1500 iters (img L1
0.133→~0.01–0.03), binding/count frozen, anti-spike reg. **Before|Teacher|After triptychs
(f00–f03, f05) show a clear, reproducible frontal sharpening** toward UNet quality — the hazy,
speckled, washed-out "before" becomes noticeably crisper and better-defined ("after"), not
teacher-perfect but a real jump. **Gate PASSED → commit to the full plan.**
Artifacts: box `~/niva-artifacts/standalone/s3_distill_spike/` and
S3 `artifacts/v2_standalone/distill_spike/` (ckpt_distilled.pt + triptychs).

NOTE for STEP 1: the spike's "lowest-yaw" frame picker used `posecode[1]`, but the GAGAvatar
runtime zeroes global rotation and the real viewpoint lives in `transform_matrix` — so yaw must
be derived from `transform_matrix`, not posecode (a couple of selected frames, e.g. f10, were not
actually frontal). Fix the frontal/yaw selection in STEP 1's true-camera multi-view fit.

### STEP 1 — Proper full-360 multi-view fit
- Render from **every turntable frame's own tracked camera** (not synthetic rotation of a frontal
  splat). Uniform angle sampling across the 358° span. Holdout spread across ALL yaws.
- Fixes the "good centre, bad sides" bias.

### STEP 2 — SH degree 1 (kill fringing) + let scale shrink (sharpness)
- Train SH deg-1 (or deg-0 while geometry settles, then raise to 1). Degree-3 overfits.
- Lower scale LR floor so splats can become small/crisp; keep anti-spike reg.

### STEP 3 — UNet distillation across angles (bake neural detail into the standalone splat)
- Teacher = GAGAvatar UNet renders at the turntable angles. Student = our plain SH splat.
- The UNet's sharpness gets baked into the Gaussians; final splat ships WITHOUT the UNet.
- **This is the key lever to approach Spatius quality on a standalone splat.**

### STEP 4 — Teeth (S4) + container/browser (S5)
- Add rigid teeth triangles bound to the jaw (borrow AGORA/`flame_with_mouth` or FLAME patch_teeth).
- Export `.csplat` (NVSP magic, zlib, fp16 pos, SH-deg1, int8 residuals) → **target 5–10 MB** like
  the rival. Browser SH shaders (deg-1) + teeth composite.

## 3. Estimates & probabilities (honest)
- **Time:** ~4–7 hrs to a genuinely-good avatar (frontal→moderate yaw); +3–5 hrs for S4+S5 ship.
  Spans >1 session.
- **Probability good FRONTAL (talking-head, ±15°):** ~80%.
- **Probability clean MODERATE yaw (±30–40°):** ~55%.
- **Probability crisp full 360/profile:** ~30% (monocular+synthetic turntable is hard; don't oversell).
- **Lightweight:** HIGH confidence — final is a standalone SH splat, UNet used only offline as a
  teacher and discarded. `.csplat` compresses ~160K gaussians to ~6–12 MB (SH-deg1 + quantise) =
  Spatius ballpark. Can prune low-opacity gaussians to go under 10 MB.

## 4. Key caveat to remember
- Identity source is the **Veo AI-generated** turntable (synthetic, watermarked). Consistent but
  not the "real" subject. Fine for a demo; flag to stakeholders.

## 5. Why NOT other paths
- **Ship GAGAvatar UNet to browser:** it's a per-frame server GPU net → violates client-side /
  lightweight requirement. Only usable offline as a teacher.
- **Adopt AGORA:** no test-time inversion in release → can't make it HER; CUDA 11.8 + 6 compiled
  forks install risk. Mine it only for the `flame_with_mouth` teeth mesh (S4) and as a reference.
- **Keep tuning the frontal-only single-view fit:** proven dead end (zero head rotation in NIVA_PSR).

## 6. First action next session
Run STEP 0 (distillation spike). Show a before/after frontal verify image. Decide go/no-go on the
full plan from that evidence.
