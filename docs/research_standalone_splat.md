# NIVA — Research Findings: Standalone Beautiful Splat (verified)

_Date: 2026-10-12 · Status: RESEARCH COMPLETE · Precedes any code for the quality rebuild._

This documents the **evidence** behind the quality plan (`plan_csplat_quality.md`),
so we don't repeat the v2/v3 mistake of coding on assumptions. Every claim here is
backed by either (A) the GAGAvatar/GaussianAvatars papers, or (B) direct inspection
of the code on our box.

---

## 1. ROOT CAUSE — confirmed by paper AND our code

**Claim:** our splat looks bad because GAGAvatar's quality lives in a neural
renderer we're not shipping, and the Gaussians carry a **32-channel neural
feature**, not RGB.

**Evidence (B) — GAGAvatar code on the box** (`gagavatar/models/GAGAvatar/models.py`):
- Gaussian param head: `out_dim = 32 + 1 + 3 + 4 + 1` → **color = 32-channel
  feature**, + opacity + scale(3) + rotation(4) + position(1 lifting).
- `self.upsampler = StyleUNet(in_size=512, in_dim=32, out_dim=3, out_size=512)`.
- Render path is **two stages**: `gen_images = render_gaussian(...)` (32-ch splat)
  → `sr_gen_images = self.upsampler(gen_images)` (StyleUNet → **RGB**).
- Only `colors[..., :3] = sigmoid(colors[..., :3])` — i.e. just the first 3 of 32
  channels are a crude RGB; the other 29 are features the StyleUNet decodes.

**Evidence (A) — GAGAvatar paper (arXiv:2410.07971):** *"Finally, we use a **neural
renderer to refine the splatting-rendered results**, producing the final reenacted
image."* The NeurIPS slide likewise lists a *"rendering module to refine the
Gaussian Splatting result."*

**Therefore:** our browser plain-splat shows a **3-channel slice of a 32-channel
neural feature**. It can never look good this way. Our v2/v3 "refinement"
optimised that 3-ch slice toward the StyleUNet's *output* → blur + speckle, higher
PSNR but uglier. **This is the whole problem, now proven.**

**Implication:** GAGAvatar is the wrong *final* representation for a no-neural-net
browser. It's still fine as the **tracker + initialization** (dual-lifting gives a
good initial point cloud), but the appearance must be rebuilt as a true,
self-sufficient 3DGS splat.

## 2. THE RIGHT REPRESENTATION — GaussianAvatars (CVPR 2024, arXiv:2312.02069)

A per-subject-optimised, FLAME-rigged 3DGS head that renders with the **standard
3DGS rasterizer (SH color), no neural net** — exactly our target.

### 2.1 Triangle binding (their Eq.1)
Each Gaussian is rigged to a **parent FLAME triangle**. From the triangle per
frame we derive:
- **b** = triangle center (world position origin for the splat),
- **R** = triangle orientation (rotation matrix from its local frame: edge dir,
  normal, their cross),
- **k** = a scalar scale = the triangle's mean edge length (so splats scale with
  the mesh).
World transform of a splat with local offset Δμ, local rotation r, local scale s:
```
μ_world = k · R · Δμ + b          (position)
R_world = R · r                   (rotation)
s_world = k · s                   (scale)
```
Local params (Δμ, r, s, opacity, SH color) are what we optimise; the triangle
(b, R, k) comes from FLAME each frame. **This is standard FLAME math → license-safe
to reimplement** (our plan's binding hedge).

### 2.2 Binding inheritance (densification without losing control)
During adaptive density control, when a Gaussian is **cloned/split**, the children
**inherit the parent's triangle index**, so every splat stays bound to a triangle
throughout. Prune removes splats but never orphans bindings. (GaussianAvatars §3.3.)

### 2.3 Spherical-harmonics color
- Vanilla 3DGS and GaussianAvatars use **SH degree 3 → (3+1)² = 16 coeffs/channel ×
  3 channels = 48 floats/Gaussian**.
- The INRIA rasterizer evaluates SH with the **camera position** (view-dependent).
- **For a browser**, degree 1 (4 coeffs/ch = 12 floats) is a strong cost/quality
  balance; degree 0 = flat (what we have). The reference competitor uses
  `f_dc` + `f_rest` (SH). **Decision: train degree 3 offline for quality; export
  degree 1 (or baked DC + 1 band) for the web to keep the file small.** (degree is
  a tunable; validate visually.)

### 2.4 Teeth
FLAME's mouth interior is weak. GaussianAvatars/`patch_teeth` add **rigid teeth
triangles**: upper teeth bound to the head, lower teeth bound to the **jaw**, so
splats can represent the mouth interior that opens/closes with jaw pose. Needs a
teeth mesh patch (FLAME teeth patch → 5143 verts / 10144 faces per our plan).

### 2.5 Optimization & anti-spike regularization (concrete defaults)
From the official GaussianAvatars config / 3DGS lineage (use as starting points,
then tune):
- **~30K iterations**, Adam; densify every ~100 iters within ~[500, 15000].
- **Position reg** `L_pos = ‖max(Δμ, ε)‖²` — keeps each splat near its triangle
  (prevents splats flying off → floaters/speckle).
- **Scaling reg** `L_scale = ‖max(s, s_max)‖²` and/or a **max aspect-ratio clamp
  (~8:1)** — prevents spiky, needle Gaussians (the artifact source).
- Opacity reset + prune low-opacity/oversized splats periodically.
These three (position reg, scale reg, aspect clamp) are the **anti-spike** triad in
our plan Task 8 — the thing our broken v2/v3 lacked.

### 2.6 Monocular caveat (IMPORTANT — manage expectations)
GaussianAvatars is designed for **multi-view** capture. We have a **single
monocular video** (`NIVA_PSR.mp4`) + 5 portraits. Monocular is harder: side/back of
head is unobserved → will always be weaker at large yaw (matches our ±15° turntable
finding). Monocular-friendly relatives to borrow tricks from:
- **FlashAvatar** (permissive license; UV-embedded splat; monocular, fast).
- **MonoGaussianAvatar / SplattingAvatar / Gaussian Déjà-vu** (monocular, per-subject).
**Realistic target:** beautiful **frontal + moderate yaw** avatar. Full 360° is not
achievable from this capture — don't promise it.

## 3. OUR STACK — what's available (verified on box)

- Repos present: `GAGAvatar`, `GAGAvatar_track`, `diff-gaussian-rasterization`
  (the 32-ch INRIA rasterizer, **supports `shs` + `sh_degree`**).
- **GaussianAvatars repo is NOT cloned** — we'd reimplement the binding +
  densification + reg ourselves (license-safe) OR fetch FlashAvatar.
- Assets: `FLAME_with_eye.pt`, `GAGAvatar.pt`, `canonical.obj`. **No teeth asset**
  present → need FLAME teeth patch for §2.4.
- Driver data (real video frames) in tracker `img_lmdb` → correct refinement target.
- Rasterizer SH path: pass `shs` of shape **(N, (deg+1)², 3)** with
  `colors_precomp=None`, and provide **campos** (already in `build_camera_matrices`).

## 4. DECISION TREE (what to actually build)

**Chosen path: rebuild appearance as a standalone GaussianAvatars-style SH splat,
initialized from GAGAvatar/FLAME geometry, refined against the REAL monocular video,
with teeth + anti-spike reg, exported as SH `.csplat` for the browser.**

Why not alternatives:
- *Keep GAGAvatar + stream video from server:* violates locked architecture
  (client renders), doesn't scale. Rejected (stopgap only).
- *Port the StyleUNet to WebGPU:* a 32→3 StyleUNet per frame in-browser is heavy and
  fragile; defeats the "lightweight on-device" goal. Rejected for v1.
- *Keep tweaking flat RGB refinement:* proven dead end (v2/v3). Rejected.

## 5. STAGED, TEST-DRIVEN BUILD (each stage ends with a blind visual + metric)

- **S1 — Binding core + unit tests.** Reimplement Eq.1 local→world transform;
  unit-test against known triangle transforms (identity, translation, rotation,
  uniform scale). No rendering yet. *Gate:* math tests pass.
- **S2 — Standalone SH splat init.** Initialize splats from FLAME triangles (+ seed
  from GAGAvatar dual-lift point cloud for density where helpful); SH degree 3;
  render through plain rasterizer (`shs`, no StyleUNet). *Gate:* neutral frontal
  render is recognisable, upright, no neural net.
- **S3 — Per-subject refinement vs REAL video** with position+scale reg + aspect
  clamp + densification/binding-inheritance; checkpointed/resumable. *Gate:*
  held-out PSNR/SSIM up **and** blind visual beats current base; no speckle; aspect
  bounded.
- **S4 — Teeth mesh** bound to jaw. *Gate:* open-mouth frames show teeth.
- **S5 — SH `.csplat` export + browser SH shaders.** *Gate:* browser render matches
  offline within tolerance at frontal/moderate yaw; file 5–10 MB.

Each gate requires a side-by-side with the same camera/renderer and a human look —
never PSNR alone.

## 6. SOURCES
- GaussianAvatars, Qian et al., CVPR 2024 — arXiv:2312.02069 ·
  https://shenhanqian.github.io/gaussian-avatars (binding Eq.1, inheritance, reg).
- GAGAvatar, Chu & Harada, NeurIPS 2024 — arXiv:2410.07971 ·
  https://github.com/xg-chu/GAGAvatar (neural renderer dependence; dual-lifting).
- 3D Gaussian Splatting, Kerbl et al., SIGGRAPH 2023 (SH color, densification).
- Monocular relatives: FlashAvatar, MonoGaussianAvatar, SplattingAvatar,
  Gaussian Déjà-vu (arXiv:2409.16147).
- Our box: `gagavatar/models/GAGAvatar/models.py`, `libs/utils_renderer.py`,
  `modules/style_unet.py`; `diff-gaussian-rasterization` repo.
