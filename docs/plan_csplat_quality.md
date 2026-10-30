# NIVA Arc A — "Beautiful Standalone Splat" Plan (CSPLAT-reference-aligned)

_Date: 2026-10-12 · Owner: NIVA team · Status: PROPOSED (awaiting review before code)_

> **Goal.** Make the NIVA avatar genuinely beautiful **as a standalone client-side
> 3D Gaussian splat** — the quality level of the competitor analysed in
> `research/AVATARKIT_CSPLAT_REFERENCE.md` — with **no server-side neural
> renderer**. The browser loads one compact splat file and renders it directly.

---

## 1. Why the current avatar looks bad (root cause, verified 2026-10-12)

GAGAvatar is a **two-part** system:

1. ~58K / 180K FLAME-bound Gaussians (geometry + **flat RGB** color), **plus**
2. a **neural UNet super-resolution renderer** that converts the rasterised
   **32-channel feature** maps into the final photorealistic RGB image.

The good-looking GAGAvatar demo render uses **both** parts. Our Arc A ships the
splat **as plain RGB to the browser WITHOUT the UNet** (the UNet can't run
real-time client-side). Consequences we confirmed with measurements:

- Plain-splat render is **blurry/soft** — the detail lived in the discarded UNet.
- "Refinement" v2/v3 made it **worse** — it optimised raw per-Gaussian RGB to
  match the UNet's *output*, which averaged/blurred colours and added hairline
  speckle. (base PSNR 19.5 looked best; v2 23.4 / v3 24.1 looked worse — PSNR
  rewarded the blur.)
- Only **2.8%** of Gaussians move per frame — expression/mouth detail is painted
  by the UNet, not carried by the splat.

**Conclusion:** the model isn't "bad" — we're running **half of it**. To render
client-side with no neural net, the **splat itself must carry the quality.**
That is exactly what the CSPLAT reference competitor does.

## 2. What the competitor does (from AVATARKIT_CSPLAT_REFERENCE.md)

Their on-device 3DGS avatar looks good with **no server neural net** because:

1. **Spherical-harmonics (SH) view-dependent colour** (`f_dc` + `f_rest`), not flat RGB.
2. **Separate subdivided teeth mesh** (fixes the open-mouth interior).
3. A **genuinely self-sufficient, per-subject-optimised splat** — Gaussians carry
   all appearance.
4. Delivery via a **compressed, versioned container** (CSPLAT: zlib + CRC + version).

## 3. Feasibility on our stack (verified on the box, 2026-10-12)

- Rasterizer = `diff_gaussian_rasterization_32d` (standard INRIA 3DGS, 32-ch).
  It **natively supports `shs` + `sh_degree`**; GAGAvatar currently calls it with
  `shs=None, colors_precomp=flat_rgb`. **→ SH is a supported path, not a rewrite:
  supply SH coeffs + set sh_degree.**
- FLAME 2020 + teeth patch (`patch_teeth`, 5143 verts/10144 faces) is referenced
  in our plan and the masks are available locally → teeth mesh is attainable.
- Reference frames (real video) live in the tracker `img_lmdb` on the box → we can
  refine against the **true photos**, not the UNet output.

## 4. The fix — promote CSPLAT levers to the CORE of Arc A

Ordered by **visual impact per unit effort**:

### P1 — Refine against the REAL reference video (not the UNet output)  ★ correctness
The single biggest conceptual fix. Re-run per-subject refinement where the target
is the **actual reference frame** from `img_lmdb`, rendered through the **plain
32-ch rasterizer** (the same thing the browser will do) — never the UNet.
- Optimise: SH colour coeffs, opacity, anisotropic scale, (optionally) small
  position deltas on the non-head static points.
- Keep: FLAME binding + Gaussian count frozen (identity guarantee).
- Regularise (anti-spike, from plan Task 8): scale-freeze warmup, log-scale clamp
  (max aspect ~8), screen-space anti-anisotropy, dual-side random-bg compositing.
- *Done when:* held-out PSNR/SSIM **and a blind visual** both beat base, identity
  preserved. (We will NOT ship a higher-PSNR-but-blurrier result — lesson learned.)

### P2 — Spherical-harmonics colour  ★ biggest look upgrade
Replace flat RGB with SH (DC + degree-1 = 9 coeffs/channel; start degree-1).
- Export format: add `sh_dc` (3) + `sh_rest` (per-degree) fields, versioned.
- Offline: fit SH during P1 refinement; render via `shs=` + `sh_degree=1`.
- Browser: evaluate SH with view direction in the WebGPU/WebGL2 shaders.
- *Done when:* view-dependent shading visible on turntable; face reads as 3D, not flat.

### P3 — Separate teeth mesh  ★ fixes the worst artifact (mouth)
Attach a subdivided teeth mesh bound to the jaw, composited with the head splat.
- *Done when:* open-mouth frames show plausible teeth, no black hole.

### P4 — Compressed, versioned `.csplat` container  ★ the deliverable file
Our own CSPLAT-equivalent — **this is the "dot file" we create at the end**:
```
magic   "NVSP" (4 bytes)
uint16  version
uint32  vertex_count
uint8   compression  (1 = zlib)
uint8   sh_degree
...     field layout descriptor (positions fp16, SH int8/fp16, opacity, scale, rot)
uint32  header_crc
payload zlib-compressed interleaved per-Gaussian attributes
```
- Target **5–10 MB** (competitor parity). Keep raw `.bin`+`.json` as debug form.
- Browser loads it with `DecompressionStream('gzip')` (no new deps).
- *Done when:* round-trip loader matches offline render; file in S3; size in budget.

### P5 — Browser renderer upgrades
Update `web-renderer` WebGPU + WebGL2 shaders to: parse `.csplat`, evaluate SH,
composite teeth, keep the FPS tiers (High ~60 / Med ~30 / Low 20–30).

## 5. Honesty guardrails (carry forward)
- Never present a worse result as an improvement (PSNR ≠ beauty — require a blind
  visual check).
- Never fabricate expressions / imply speech when not driven by the source.
- Turntable limited to a yaw range the geometry actually supports (~±15°) until the
  splat has real side/back coverage; don't hide weak angles.
- Preserve all existing checkpoints (base, v2, v3) for reproducibility.

## 6. What we will NOT do (scope discipline)
- No server-side video rendering (violates locked architecture; doesn't scale).
- No swapping away from FLAME/GAGAvatar lineage (license hedge already planned).
- SH degree stays ≤ 1 for web cost/quality balance unless profiling says otherwise.

## 7. Suggested order of execution
1. P1 (refine vs real video) — proves the splat can look good standalone.
2. P2 (SH colour) — the beauty jump.
3. P3 (teeth) — removes the ugliest artifact.
4. P4 (`.csplat`) + P5 (browser) — ship the real deliverable.
Each step ends with a turntable + held-out comparison (same renderer/camera) and a
blind visual sign-off before moving on.

## 8. The "dot file" / final artifact
The end-of-pipeline artifact the browser consumes is **`niva_avatar.csplat`** (the
P4 container above): one compact (~5–10 MB), versioned, zlib-compressed,
SH-coloured, FLAME-bound Gaussian splat + binding tables + FLAME metadata,
published to S3. _(If instead you meant a Graphviz `.dot` system diagram, say so and
I'll generate `docs/architecture.dot` of the full tracking→refine→export→browser→
Nova-Sonic pipeline.)_


---

## 9. Enhancements backlog (deferred, requested 2026-10-13)

These are known improvements captured for later; not blocking the current ship.

### E1 — Smoother lip-sync (no mid-speech pauses)  ★ UX
Observed: during speech the mouth does not read as closing between words — there
are visible pauses / the jaw sticks. Cause is the amplitude-envelope driver in
`app_frontal.js`: a single low-pass (`SMOOTH 0.35`) on a coarse FFT band gives a
laggy, gappy signal, and silence gaps snap the jaw shut.
- Fix directions: use a short attack / longer release envelope (fast open, slow
  close) instead of a symmetric low-pass; add a small jaw floor during voiced
  frames; optionally drive from phoneme/viseme timing (forced alignment of the
  audio) rather than raw amplitude. Viseme track aligns with the eventual
  `animation.proto`-style motion data (see CSPLAT reference §2.4).
- *Done when:* mouth tracks speech cadence with no visible stalls at word gaps,
  verified on `common/test.wav` in-browser.

### E2 — Move rendering off the main thread (multi-thread)  ★ perf
Observed: the whole avatar (morph blend + depth sort + draw) runs on the main JS
thread, so sort/morph of 159k Gaussians can jank the UI.
- Fix directions: run the per-frame morph blend + depth sort in a Web Worker
  (transfer the packed buffer via `postMessage` with transferables, or share via
  `SharedArrayBuffer`). This pairs naturally with the planned C++→WASM compute
  core (CSPLAT decode + FLAME LBS + sort) — build the WASM core to run inside the
  worker, leaving only the WebGL2 draw on the main thread.
- *Done when:* sort/morph no longer block the main thread; frame pacing stays
  smooth while the mouth/eyes animate.
