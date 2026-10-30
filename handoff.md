# NIVA — Session Handoff

_Last updated: 2026-10-15 (end of day-6). Focus this session: finished the FRONTAL-ONLY
quality pass. DONE: Task 3 (Web-Worker jitter fix, pixel-verified), Task 4 (eye/blink/glitter
fix -> checkpoint v4c), Task 5 (face sharpen via display-time unsharp mask). IN PROGRESS: Task 6
(waist-up static body-image composite) — functional but NOT visually finished (dark halo around
head, alignment not perfect). Task 7 (final verify) pending. `.csplat`/WASM (Arc A) still DEFERRED._

---

## 0tri. ⚠️ DAY-6 END STATE (read first — supersedes day-5/day-4 below)

**Instance:** `i-07f8d2751a8f98d44` (g6e.4xlarge L40S, eu-central-1). Day-6 IP was `63.179.112.179`
(changes on restart). **Instance STOPPED at end of day-6** (EBS root `vol-0a5a4e53210cf2f92`,
DeleteOnTermination=TRUE -> survives STOP, lost on TERMINATE). Port 8000 ingress rule was
`sgr-0c16b63006413a882` (added day-5) — **REVOKED at end of day-6** (see below; re-add on restart to
demo). On restart: new IP, re-serve both http servers:
`cd ~/fweb && setsid python3 -m http.server 8000 </dev/null >/tmp/fweb.log 2>&1 &` and
`cd ~/fweb_ref && setsid python3 -m http.server 8001 </dev/null >/tmp/fweb_ref.log 2>&1 &`.
:8000 = live worker+sharpen+v4c+Task6-WIP build; :8001 = v3 reference (for pixel-diff).

**TASK 3 — Web-Worker jitter fix — DONE & PIXEL-VERIFIED.** `_verify_pixdiff.py` on box (niva-offline
python): NEW:8000 vs REF:8001 meanAbsDiff neu 0.0002/255, talk/blink/jawfull 0.0000, NEW
__mainThreadSorts=0 => PIXEL-EQUIVALENT. Morph+sort fully off main thread, bit-identical to v3.
HARNESS GOTCHA: run under `timeout`, write stdout to /tmp file, read from a SEPARATE ssh (chromium
hangs on EXIT after results are written — the 124 is a false alarm). Use wait_until=domcontentloaded
(load hangs with the worker). Do NOT pkill the shared chromium mid-run (one browser does NEW then REF).

**TASK 4 — eye/blink/eyelid-glitter fix — DONE -> checkpoint `s3_frontal_v4c` (SHIPPED, live).**
v3 had rainbow SPECKLE on the closed eyelid (saturated iris/lash SH gaussians through the lid);
eye-crop L1 (diag_eye2.py, rows225:290 cols150:362 teacher-vs-student) grew with blink: v3
0.0056/0.0119/0.0138 @ blink 0/0.5/1.0. Fix via re-distill warm-started from v3 with a per-pixel
SATURATION-EXCESS penalty on closed-eye frames in the eyelid band (NO total-variation — a TV attempt
= distill_v4 SMEARED the lid, abandoned). Progression: v4b (sat penalty, dead-zone 0.02) good ->
v4c (warm from v4b, no dead-zone, weight 1.5, 1500 it) BEST. v4c eval: neu 0.006 / jaw35 0.0085 /
eye60 0.0065; blink eye-crop 0.006/0.0101/0.0074 (closed-eye ~HALVED vs v3). Neutral+jaw UNCHANGED,
no smearing. Baked (bake_v4c.py = bake_v3 -> v4c): baked-vs-fullSH ~1e-8, browser-blend-vs-exact
<=0.0013. Deployed to ~/fweb/assets_frontal. Residual FAINT sparkle remains — an OPUS-4.5 sub-agent
proved it is NOT removable per-gaussian (v4c lid p99 saturation 0.364 is ALREADY BELOW teacher 0.366;
the specks match the teacher's own appearance; suppressing RAISES L1). v5 NOT produced. Logged as a
known teacher-level minor artifact. Box dirs: s3_frontal_v4/v4b/v4c (+ s3_frontal_v5 EMPTY/no-go).

**TASK 5 — frontal face sharpen — DONE (no re-distill; display-time only).** diag_sharp.py: v4c
vs UNet teacher (sr_gen_image = the sharpest signal available) full-face L1 0.0057, gradmag ratio
0.763 — but FACE is already ~90-94% of teacher (eyes 94%, mouth 88%); the gap is mostly HAIR high-freq.
A full re-distill was JUDGED NOT WORTH the risk (would regress the good face for marginal hair gain —
the PSNR-rewards-blur lesson). FIX = display-time UNSHARP MASK in the renderer composite pass
(renderer_webgl2.js CFS: 4-neighbour laplacian, uSharp uniform, gated if(uSharp>0) so uSharp=0 is
identity, alpha-weighted so no bg ringing). API setSharpen(a) clamp 0..1.5, default sharpAmount 0.35,
?sharp= query param (app_frontal.js). A/B confirmed: 0.35 visibly crisper hair/lashes/skin, NO
ringing, colour/identity unchanged, gradmag +1.30x. CONFINED to composite: setMorph + morph_worker.js
BYTE-UNCHANGED, so Task-3 pixel-equivalence still holds for the morph path.

**TASK 6 — waist-up STATIC body composite — IN PROGRESS, NOT VISUALLY DONE (first thing next session).**
Source `NIVA/images/front.png` (1176x1337 waist-up portrait) copied to `~/fweb/assets_frontal/body.png`.
Implemented (sub-agent + main-agent edits, all in ~/fweb and local web-renderer/public):
- renderer_webgl2.js composite shader now outputs TRANSPARENT alpha (o = vec4(rgb, c.a)); context
  already alpha:true,premultipliedAlpha:true. The float-accum FBO clears to (0,0,0,0); final canvas
  clear still uses opaque BG only in the no-float fallback path.
- index_frontal.html (= index.html): fixed-aspect #avatar-box (588:668), <img id=body> behind a
  transparent #gfx canvas (object-fit:cover, object-position 51% 0%), page bg set to #cfc7bd (photo
  tone). Canvas CSS mask = radial ellipse (46% 44% at 50% 42%) INTERSECT bottom linear fade (black
  70% -> transparent 84%) to show only the live face+front-hair and fade into the photo.
- VERIFIED there is NO head motion in app_frontal.js (fixed camera; only blink+jaw morph) -> no
  clamp needed.
KNOWN PROBLEM (why it's not done): captured full-page frames (/home/deval/niva-frontal-avatar/
bc_* and body_* / the earlier body_neu/body_blink) show a DARK HALO/void around the head — the
avatar's own semi-transparent dark HAIR tails veil the photo, and the live-head vs photo-head
ALIGNMENT is not exact (photo head sits lower/smaller than the live head). The radial-mask attempt
(latest edit) was deployed but NOT yet visually confirmed (capture was interrupted at EOD). NEXT
SESSION: re-capture bc_neu/bc_blink at 588x668, VIEW them (main agent — sub-agents can't see images),
and iterate the object-position/scale + mask radius until the live head lands exactly on the photo
body with no halo and a clean neck seam. Backup of pre-Task6 web files on box at ~/fweb/_bak_task6/.

**ENHANCEMENTS BACKLOG (deferred, in docs/plan_csplat_quality.md §9):** E1 smoother lip-sync (DONE
as Task 2), E2 off-main-thread (DONE as Task 3). Future: .csplat container + WASM compute core.

---

## 0ter. BACKUPS (day-6)

- **S3** `s3://niva-avatar-assets-508137168365-euc1/artifacts/v2_standalone/`:
  - `frontal_v4c/` = v4c ckpt (101MB) + web/ assets + scripts + triptych + web_js/ (sharpen build)
    + **web_build_day6/** (the FULL current web build incl. body.png + Task-6 WIP html) — uploaded OK.
  - `frontal_v3/` = v3 ckpt + assets + scripts (from day-5).
  - ⚠️ `frontal_v4b/ckpt_distilled.pt` upload FAILED (AWS token expired at EOD). v4b is a FALLBACK
    only (v4c is shipped). It IS safe on box EBS + local mirror. Re-upload next session when creds
    refresh: `aws s3 cp niva-artifacts/s3_frontal_v4b/ckpt_distilled.pt s3://.../frontal_v4b/`.
- **Local mirror** (`niva-artifacts/`): s3_frontal_v3, s3_frontal_v4b (ckpt), s3_frontal_v4c
  (ckpt + web + web_js + triptych + scripts). web-renderer/public = current Task-6 WIP build.
- **Box EBS** (survives STOP): all checkpoints s3_frontal_{fixed,anim,v3,v4,v4b,v4c,v5-empty}, live
  ~/fweb (worker+sharpen+v4c+Task6-WIP), ~/fweb_ref (v3), ~/fweb/_bak_task6, /tmp/rollback_day6
  (pre-v4c live assets), all distill_v4*/bake_v4c/diag_* scripts in ~.
- **AWS creds note:** the BOX has no AWS creds; all S3 ops run from the workstation. The workstation
  token expired mid-wrap-up (only the redundant v4b upload was lost).

---

## 0bis. ⚠️ DAY-5 END STATE (historical — superseded by DAY-6 above)

**Instance:** `i-07f8d2751a8f98d44` restarted today; **IP was `18.184.61.55`** (changes on
restart). **Instance STOPPED at end of day-5** (EBS root preserved — safe). Port 8000 ingress
rule `sgr-0c16b63006413a882` (TCP 8000 from 0.0.0.0/0) was ADDED today and LEFT IN PLACE; SSH(22)
also open. On restart: get new IP, confirm `~/fweb` http.server (was PID 1735) and re-serve if
needed: `cd ~/fweb && setsid python3 -m http.server 8000 </dev/null >/tmp/fweb.log 2>&1 &`.
A backup reference build is served from `~/fweb_ref` on :8001 (copy of the day-4 working v3) for
pixel-diff comparisons; `cd ~/fweb_ref && setsid python3 -m http.server 8001 ... &` to restore it.

**User's locked task list (this + next sessions), in order:**
1. ✅ Backups (done).
2. ✅ Smoother lip-sync (Task 2, DONE + verified).
3. ⏳ Jitter fix via Web Worker (Task 3, CODE-COMPLETE, NOT yet pixel-verified — see below).
4. ⛔ Eye + blink + glitter fix (GPU box re-bake/re-distill of frontal eye region + blink). NOT STARTED.
5. ⛔ Frontal UNet-distillation upscale (frontal only, no 360) to sharpen face. NOT STARTED.
6. ⛔ Composite waist-up BODY IMAGE below the neck + feather neck seam + clamp head motion
   (use `NIVA/images/front.png` — it is a waist-up portrait: head+hair, neck, shoulders, chest,
   torso to abdomen. NO 3D body splat; just a static image plate behind the transparent canvas,
   aligned to `meta.camera`). NOT STARTED.
7. ⛔ Final end-to-end verification.

**TASK 2 — smoother lip-sync — DONE & VERIFIED.** Rewrote `makeLipsync()` in `app_frontal.js`:
replaced the symmetric low-pass (`SMOOTH 0.35`, which snapped the jaw shut at every word gap) with
an asymmetric **attack/release envelope + voiced jaw-floor**. Constants: `JAW_CAP=0.35
ATTACK_MS=45 RELEASE_MS=160 VOICE_TH=0.06 JAW_FLOOR=0.10 FLOOR_HOLD=180 NOISE_FLOOR=0.03`.
Deterministic logic port confirms: attack ~96 ms to 90% open, release ~144 ms to 50% / ~496 ms to
closed, jaw holds ~0.20 during word gaps, 0 snap-to-zero. Headless load: 0 errors, 159,616
gaussians, audio plays & jaw moves. Added opt-in `window.__jawTrace` debug hook (off by default).

**TASK 3 — jitter fix (Web Worker) — CODE-COMPLETE, PIXEL-VERIFICATION UNFINISHED.**
Root cause (measured): `setMorph()` ran every frame and forced a FULL depth re-sort; `render()`
then did computeDepths + comparator-sort of 159,616 items + reorder (159k×14 copies) ON THE MAIN
THREAD. Morph blend alone measured ~23.6 ms/call → frame work exceeds the 16.6 ms/60fps budget →
the "whole avatar refreshes/flickers under load" the user reported.
Implementation (frontal-only, fixed camera):
- NEW `web-renderer/public/morph_worker.js` — a classic Web Worker that does the morph blend +
  depth compute + sort + reorder off-thread and RE-SORTS EVERY FRAME (NOT throttled — an earlier
  sub-agent version throttled the sort to every 10th frame and that caused a visible eye-region
  diff on blink: pixel meanAbsDiff 4.79/255. The point is to move the work off-thread, not skip it).
- `renderer_webgl2.js` — worker integration rewritten to a clean LENT-BUFFER protocol: the main
  thread owns a pool of 3 ArrayBuffers and transfers one to the worker with each frame request;
  the worker fills it and transfers it back (no detached-buffer corruption, no per-frame alloc).
  `setMorph` lends a free buffer + posts {wJaw,wBlink,viewMatrix}; `render` uploads the latest
  returned buffer or re-draws the last one (never blank, never blocks). Sync fallback kept if
  Worker unavailable. Added `window.__mainThreadSorts` counter (must stay 0 in worker mode).
- `app_frontal.js` logic UNCHANGED by Task 3 (still calls `r.setMorph`/`r.render`); it only carries
  the Task-2 lip-sync + the `__jawTrace`/`__forceMorph` hooks.
Verified so far: NEW build loads via `wait_until="domcontentloaded"` with `__r` present, N=159,616,
0 page errors, worker logs "morph_worker ready". **NOT yet verified: pixel-equivalence vs the
backup across forced morphs [0,0]/[0.35,0]/[0,1]/[1,0], and __mainThreadSorts==0.** The headless
harness kept hanging and the session ended before the diff completed — THIS IS THE FIRST THING TO
FINISH NEXT SESSION before trusting Task 3.

**⚠️ HEADLESS GOTCHA (important):** with the Web Worker build, Playwright `goto(wait_until="load")`
HANGS (the load event stays pending under the python http.server). USE
`wait_until="domcontentloaded"` instead — both builds then load fine. Verification scripts live
locally at `web-renderer/_verify_pixdiff.py` and `_diag_load.py` (scp to box `/tmp`, run with
`~/miniconda3/envs/niva-offline/bin/python`). Compare NEW (:8000) vs REF backup (:8001). swiftshader
throttles rAF to ~5-6 frames/7s, so DO NOT measure fps headless — prove Task 3 via the pixel diff
(must be ~0 mean abs diff) + `__mainThreadSorts==0`.

**Tooling installed on box this session (niva-offline conda env):** `playwright` pip pkg +
chromium rev1248 (`~/miniconda3/envs/niva-offline/bin/python -m playwright install chromium`), plus
`pillow`+`numpy` for pixel diffs. Conda envs: GAGAvatar, GAGAvatar_track, niva-offline. Box has NO
node (can't `node --check`; syntax-validate via brace balance or the browser).

**Backups made day-5 (both verified identical to day-4 working v3 before edits):**
- Local: `web-renderer/public_backup_v3_working_20261014-143617` (36M, full assets).
- Box: `~/fweb_backup_v3_working_20261014-090642` and the live-serving `~/fweb_ref` (= same v3).
- JS sha256 of the pristine v3: app_frontal.js=`cccce...`, loader.js=`f1a34...`,
  renderer_webgl2.js=`569e1...`. (Task-2 app_frontal.js = `7dde2b...`.)

**Files changed day-5 (local `web-renderer/public/` AND deployed to box `~/fweb/`):**
`app_frontal.js` (Task 2 lip-sync + debug hooks), `renderer_webgl2.js` (Task 3 worker wiring),
`morph_worker.js` (NEW, Task 3). NOTHING on the GPU/model side changed yet (Tasks 4–6 are the
box re-bake/distill + image composite, still to do).

**Design decisions locked with the user (do not relitigate):**
- Frontal-only. NO 360 / no multi-view fit. "Quality upscale" = sharper FRONTAL face + fix the
  eye glitter + fix the bad blink, NOT more viewing angles.
- Jitter is BROWSER-SIDE (main-thread stall), fixed client-side (Task 3), no retrain needed.
- Body shown to the torso via a STATIC IMAGE below the neck (not a Gaussian body, not animated).
  Head animates with small motion; clamp head motion so the neck/collar seam stays hidden.
- Use an Opus-5.5 sub-agent for the GPU/model-quality work (Tasks 4–5) "where necessary" — NOTE:
  the `subagent` tool rejected model id `opus-5.5` this session ("model not available"); either
  omit the model override (uses default) or get the correct model id from `/model`.
- ALL build/GPU work happens ON THE BOX, not the workstation. (User reminder day-5: run on the box.)

---

## -1. ⚠️ DAY-4 END STATE (read first)

**What is live and good:** http://63.180.18.195:8000/ renders the frontal Nidhi avatar with
clean neutral colour/identity, a real blink (eyelids close, no glitter), subtle lip-sync, and a
clean open mouth (teeth/lips, no two-layer ghost). Verified by headless browser captures on the
box (neutral/blink/talk/jaw-full). NOTE: the public IP and the live link are only valid while the
instance runs — **the instance is STOPPED at end of day-4**, so the link is DOWN until restart
(and the IP will change on restart).

**The v3 fix (root causes + what changed):**
- The dim/altered face + dark eyelid ring + glitter were NOT an SH bake/export bug (baked vs full
  SH diff ≈ 1e-8). Cause: the earlier `s3_frontal_anim` re-distill made the SAME splats serve both
  open-eye and closed-eye frames, so neutral drifted to a muddy average (eye-region L1 0.028 vs the
  clean model's 0.007).
- The two-layer open mouth was the browser morphing ONLY positions; stale closed-mouth
  rotation/scale left the old lip surface showing through.
- Fix = `distill_v3.py` (warm-start from the clean `s3_frontal_fixed`): neutral SH/opacity anchored
  to the clean model; jaw and blink get their OWN per-gaussian SH + opacity deltas scaled by morph
  weight. Teacher L1: neu 0.0062, jaw 0.0085, blink 0.0081. Checkpoint `s3_frontal_v3`.
- Export/browser now morph pos+quat+scale+rgb+opacity (not just pos). New binary anim format
  (29 f32/gaussian, see below). Browser blend vs exact gsplat ≤ 0.0013 at every weight.

**Files that changed (local web-renderer/public, deployed to box ~/fweb):**
- `loader.js` — loads binary `niva_frontal_anim.bin` (meta.anim), keeps `basePacked`.
- `renderer_webgl2.js` — `setMorph` does the full pos/quat(nlerp)/scale/rgb/opacity blend.
- `app_frontal.js` — unchanged logic (capped+smoothed lip-sync, blink timer, fixed camera).
- `index.html` is now a copy of `index_frontal.html` (orbit `app.js`/`index.html` quarantined).

**New asset format (`niva_frontal_anim.bin`, stride 29 f32/gaussian):**
`[base_opacity_logit(1)] [jaw: dpos3 dquat4(wxyz) dscale3 drgb3 dlogit1] [blink: same 11]`.
Static `niva_frontal.bin` is still 14 f32 (pos3 quat4 scale3 opac1 rgb3). `niva_frontal.json`
carries `meta.camera` and `meta.anim`.

**Backups (verified):**
- S3: `s3://niva-avatar-assets-508137168365-euc1/artifacts/v2_standalone/frontal_v3/` (ckpt 101 MB,
  web bin/json, scripts, triptychs) — 15 objects, listed OK.
- Local: `niva-artifacts/s3_frontal_v3/` (same contents).
- Box EBS (root `vol-0a5a4e53210cf2f92`, DeleteOnTermination=TRUE → survives STOP, lost on
  TERMINATE): `~/niva-artifacts/standalone/{s3_frontal_v3,s3_frontal_fixed,s3_frontal_anim}`, live
  `~/fweb`, scripts `~/{distill_v3,bake_v3,bake_v2,diag_exp,diag_eye,fcap4}.py`.

**Task 6 cleanup (frontal-only):** 360/orbit/experiment files moved to `_quarantine_360/` (NOT
deleted) locally and on the box (~309 MB on box). Includes old turntable/dense outputs, ~16 scripts,
`orbit.html`, `probe*.html`, orbit `app.js`/`am_main.js`, `renderer_sh.js`, `loader_sh.js`,
`niva_frontal.splat`, old `niva_frontal_anim.json`. Docs `handoff.md`, `web-renderer/RUN.md`
updated to frontal-only. KEPT by design: `renderer.js` (WebGPU), `avatar_driver.js`, all tracking
`optim.pkl`, the three frontal checkpoints.

**Enhancements backlog (requested day-4, see `docs/plan_csplat_quality.md` §9):**
- E1: smoother lip-sync — no mid-speech pauses (attack/release envelope or viseme timing).
- E2: move morph+sort off the main thread (Web Worker, pairs with the WASM core).

**IN PROGRESS / NEXT (user approved direction):** build the `.csplat` container + a C++→WASM
compute core (decode + FLAME/morph blend + sort), then bind it into the WebGL2 renderer. ALL BUILD
WORK ON THE BOX, not the workstation. Box has g++, cmake, git, ~15 GB free but NO emcc/node yet —
install emsdk (`git clone emscripten-core/emsdk; ./emsdk install 3.1.61; ./emsdk activate 3.1.61`)
and node first. Spec is in `research/AVATARKIT_CSPLAT_REFERENCE.md` and `docs/plan_csplat_quality.md`
(P4/P5).

---

## 0. ⚠️ START HERE NEXT SESSION

**Goal (reaffirmed by user):** a beautiful, LIGHTWEIGHT, **interactive talking** avatar
(lip/jaw/blink/expression), rendered **client-side, no server neural net** — matching the
Spatius bar (app.spatius.ai). **Product is frontal-only** (frontal view + blink + lip-sync + small head motion).
Multi-view 360 turntable data was used ONLY for training the base model; the deployed avatar is frontal.

**First action:** run the **distillation spike** (STEP 0 in `docs/plan_quality_fix.md`) — prove
GAGAvatar's UNet can teacher-distill sharpness into our standalone splat on the frontal view, in
~1 hr, before committing the full 4–7 hr plan. Show the user a before/after verify image.

**⚠️ ALSO CHECK THE PROJECT S3 BUCKET at session start:** `s3://niva-avatar-assets-508137168365-euc1`
(region eu-central-1). It is the durable source of truth for all artifacts/weights/tracking. List it
first to see what's already saved before re-doing work:
`aws s3 ls s3://niva-avatar-assets-508137168365-euc1/artifacts/ --recursive --region eu-central-1`.
This session's new data is under `artifacts/v2_standalone/` (standalone ckpt + multi-view trackings);
prior Arc-A data under `artifacts/v1/`.

**Read these (new this session):**
- `docs/plan_quality_fix.md` — the quality-fix plan (distillation + full-360 + SH-deg1), estimates,
  probabilities (good frontal ~80%, moderate yaw ~55%, full 360 ~30%; stays lightweight = HIGH).
- `docs/PROBLEMS_AND_SOLUTIONS.md` — every problem hit + solution (READ to avoid re-discovery).
- `docs/research_standalone_splat.md` — the evidence/research behind the standalone-splat approach.
- `research/AVATARKIT_CSPLAT_REFERENCE.md` — competitor (Spatius) analysis = the target.

---

## 1. WHAT WAS BUILT THIS SESSION (standalone splat, staged S1–S5)

New code in `offline-pipeline/src/niva_offline/binding/`:
- `triangle_binding.py` — GaussianAvatars Eq.1 FLAME-triangle rig (compute_triangle_frames,
  bind_gaussians, quat helpers). **13 unit tests pass** (`tests/test_triangle_binding.py`). CPU-only.
- `standalone_splat.py` — StandaloneSplat (SH color, FLAME-bound); `init_splat_from_faces`
  (K splats/face via `--splats-per-face`); `world_gaussians` (per-frame binding); SH<->RGB helpers.
- `refine_standalone.py` — S3 per-subject refinement vs REAL frames through gsplat SH renderer;
  anti-spike reg (position/scale/aspect); `--portrait-driver`+`--portrait-frac` (multi-view);
  per-frame tracked camera (`transform_matrix`); checkpoint/resume; held-out PSNR.
- `export_standalone.py` — bake SH-DC→RGB to browser `.bin`/`.json`; `--flip-y`, `--scale-clip`,
  `--splats-per-face`.
- `s2_probe.py`, `anim_probe.py`, `verify_avatar.py`, `_asset_check.py` — probes/verification.

Also: `offline-pipeline/src/niva_offline/compare/` (yesterday's honest 3-way comparison, 13 tests).

## 2. STATUS OF STAGES
- **S1 binding:** DONE (+tests).
- **S2 standalone SH render (gsplat, no UNet):** DONE (upright FLAME head, flat init).
- **S3 multi-view refinement:** DONE-ish. Frontal good (28.5 dB), **sides streaky/fringed**.
  Animation proven (jaw/mouth deform). Checkpoint: `~/niva-artifacts/standalone/s3_turntable/ckpt.pt`
  (159,616 gaussians, 16 splats/face, SH deg3). ALSO in S3: s3_v1 (frontal-only, blurry),
  s3_dense.
- **S4 teeth:** NOT started (open-mouth shows messy interior — teeth needed).
- **S5 .csplat + browser SH shaders:** NOT started.

## 3. THE QUALITY PROBLEM (why it's not Spatius-good) — see PROBLEMS_AND_SOLUTIONS.md
1. GAGAvatar splat is low-quality by design (detail lives in a neural UNet we can't ship).
2. Our multi-view fit was frontally biased + under-trained → good centre, bad sides.
3. SH deg-3 overfits frontal → rainbow fringing off-axis.
4. (FIXED) browser WebGL2 had no depth sorting → spiky; added sort + screen clamp + export scale-clip.
**Fix plan:** full-360 fit from true cameras + SH deg-1 + **UNet distillation** (teacher→student) to
bake neural detail into the standalone splat. STEP 0 spike first.

## 4. KEY DATA / MULTI-VIEW (the breakthrough)
- Frontal video `NIVA_PSR.mp4` has ZERO head rotation (global-rot std 0.0) → NO multi-view → the
  original blur. Tracked the **360 turntable** `NIVA/images/Nidhi_Avatar.mp4` (re-encode to
  `Nidhi_clean.mp4` first: `ffmpeg -i in -an -vcodec libx264 -pix_fmt yuv420p -r 30 out`).
- Trackings on box: `GAGAvatar_track/outputs/Nidhi_clean/` (173 frames, 358.8° yaw) and
  `.../outputs/images/` (5 portraits, ±45° yaw). Also backed up to S3 (see §6).
- CAVEAT: Nidhi_Avatar.mp4 is **Veo AI-generated** (synthetic identity, watermarked). Fine for
  demo; the avatar's identity = that synthetic video, not the real subject.

## 5. INFRA / ENV NOTES (box ubuntu@63.177.114.188 — but IP CHANGES on stop/start)
- `gsplat==1.5.3` installed in GAGAvatar env (our SH rasterizer). First `rasterization()`
  JIT-compiles CUDA (~107s). Needs `CUDA_HOME=/usr/local/cuda-12.8` and the env `bin` on PATH (ninja).
- Standard INRIA diff_gaussian_rasterization NOT installed; box repo is the 32-ch GAGAvatar variant.
- CAMERA: gsplat viewmat = GAGAvatar cam [R|t] in 4x4 with `V[:2,:]*=-1` (flipxy) → upright.
  K: fx=focal_x*W/2, fy=focal_y*H/2. Use per-frame `transform_matrix` for multi-view angles.
- Browser orientation fix shipped: per-asset `flip_y` meta in loader.js/app.js; WebGL2 now
  depth-sorts + screen-clamps (renderer_webgl2.js).
- Box has NO AWS creds → S3 from workstation; scp box↔workstation.

## 6. BACKUPS (done this session)
- **EBS** (persists stop/start): `~/niva-artifacts/standalone/{s3_turntable,s3_dense,s3_v1}`;
  trackings under `GAGAvatar_track/outputs/{Nidhi_clean,images}`.
- **S3** (durable): `s3://niva-avatar-assets-508137168365-euc1/artifacts/v2_standalone/`
  = `standalone/s3_turntable/ckpt.pt`+preview, and `tracking_mv/{Nidhi_clean,images,Nidhi_clean.mp4}`.
- **Local staging:** `artifacts_s3_stage/{standalone,tracking_mv}`.
- Yesterday's Arc-A comparison video: local `outputs/compare/niva_3way_turntable.mp4`.

## 7. GPU INSTANCE START/STOP
- **Start script saved at `/home/deval/Calza/start_avatar_g6e.sh`** (retry-loop for
  InsufficientInstanceCapacity in eu-central-1c; run from workstation). Then get the NEW public IP
  (changes each start). ASG `avatar-g6e-asg` processes stay SUSPENDED (so it won't replace the
  instance); start the instance DIRECTLY to keep its EBS root.
- Instance `i-07f8d2751a8f98d44` (g6e.4xlarge L40S, eu-central-1c). SG `sg-02488d67a1a1f0a9f`.
  Root EBS `vol-0a5a4e53210cf2f92` (DeleteOnTermination=TRUE — safe to STOP, do NOT terminate).
- **End of day-4: port 8000 ingress (sgr-0fd9742d41cb87737) REVOKED, instance STOPPED (EBS preserved).**
- On restart: re-add the port-8000 rule to demo, re-fetch the new IP, confirm `~/fweb` http.server
  (nohup) is still running or restart it: `cd ~/fweb && nohup python3 -m http.server 8000 &`.

## 8. OWED / NEXT
1. Distillation spike (STEP 0) → go/no-go on quality plan.
2. If go: full-360 fit + SH-deg1 + distillation → good talking-head avatar.
3. S4 teeth (borrow AGORA `flame_with_mouth` or FLAME patch_teeth).
4. S5 `.csplat` (NVSP, zlib, fp16, SH-deg1, int8) target 5–10 MB + browser SH shaders.
5. Then Arc B: Nova Sonic (us-east-1) → ARTalk audio→FLAME → browser.
- AGORA repo (github.com/ramazan793/AGORA): useful reference + `flame_with_mouth` teeth mesh; NOT a
  drop-in (no test-time inversion → can't make it HER; heavy CUDA-11.8 install). Don't adopt wholesale.
