# AvatarKit / CSPLAT — Reverse-Engineering Reference & NIVA Implementation Guide

_Author of analysis: NIVA team · Date: 2026-10-09_
_Source analysed: `avatar_core_wasm-36bf5449.wasm` (958 KB, WebAssembly MVP) — the
on-device client renderer behind a Spatius-style avatar product
(`app.spatius.ai`). Build path embedded in the binary:
`/Users/Zhuanz/Project/avatarkit-sdks/avatar-core`._

> **Purpose.** Document how their client-side avatar engine works (as far as can
> be recovered from the shipped WASM + public product docs), and translate it
> into concrete, legally-clean recommendations for NIVA. We treat this strictly
> as a **reference for interoperability and architecture** — NIVA ships its own
> independent implementation, not their proprietary decoder.

---

## 1. How the analysis was done

- `file` / section parse → valid WASM v1 (MVP), C++-compiled (Emscripten-style).
- Exports are **minified to single letters** (`q,r,s,…`) in the release build, so
  function **signatures are not recoverable**. The intelligence is in the
  **data section strings** (error messages, protobuf descriptors, PLY property
  names), which reveal the formats and pipeline unambiguously.
- No custom name/debug sections present.

Everything below is derived from recovered strings; where we infer, it is marked
**(inferred)**.

---

## 2. Their architecture (recovered)

A single **C++ core compiled to WASM** runs client-side (browser/mobile), paired
with a GPU backend (WebGL/WebGPU) for rasterization. This matches the public
Spatius description: *on-device 3DGS, 5–10 MB model from a single photo,
lightweight motion data instead of video, client-side rendering via their SDK.*

Pipeline the WASM implements:

```
  CSPLAT / PLY / NPY  ──►  FlameModel (+ LBS weights, shapedirs, template)
        (geometry)              │
                                ├─ TeethMesh (separate, subdivided)
  animation.proto frames ──►  per-frame FLAME params (expression, pose, eye_lid,
        (motion data)            body_anchor_translation)
                                │
                                ▼
                        deform Gaussians (FLAME-bound) ──► GPU splat raster
```

### 2.1 Geometry formats

Three loaders are present:

| Format | Evidence | Role |
|---|---|---|
| **CSPLAT** (custom) | `Invalid CSPLAT magic`, `CSPLAT data too small for header`, `Unsupported CSPLAT version`, `[CSPLAT] Detected compressed point cloud`, `header crc mismatch` | Primary compact avatar container |
| **PLY** | `Failed to parse PLY header`, `element vertex %u`, `property float f_dc_`, `property float opacity`, `property float rot_2/rot_3` | Standard 3DGS interchange (import/debug) |
| **NPY** | `Invalid NPY magic number`, `Unsupported NPY version`, `Cannot find 'descr'/'shape' in NPY header` | FLAME arrays (LBS weights, shapedirs) |

### 2.2 The CSPLAT container (reconstructed)

From the error strings the layout is:

```
┌───────────────────────────────────────────────┐
│ magic            (checked: "Invalid CSPLAT magic")        │
│ version          ("Unsupported CSPLAT version: <n>")       │
│ header           ("too small for header"); includes:       │
│    - vertex_count ("Invalid metadata: vertex_count=")      │
│    - compression method ("Unsupported compression method") │
│    - header CRC   ("header crc mismatch")                  │
│ payload = gzip/zlib-compressed vertex block                │
│    ("Detected compressed point cloud", "Gzip decompression │
│     failed", "incorrect header check", "unknown header     │
│     flags set" → zlib)                                     │
└───────────────────────────────────────────────┘
```
- Compression is **zlib/gzip** (zlib error strings: `incorrect header check`,
  `unknown compression method`, `unknown header flags set`).
- Integrity via **CRC** on the header.
- Decoded payload is validated: `CSPLAT data truncated: expected <N>`,
  `Vertex count mismatch`.

### 2.3 Per-Gaussian attributes (from PLY property names)

The standard **3DGS INRIA layout** — critically, with **spherical-harmonics
color**, not flat RGB:

| Property | Meaning |
|---|---|
| `x,y,z` (position) | `PLY file missing position properties` guards it |
| `f_dc_*` | **SH DC term** = base (view-independent) color |
| `f_rest_*` (inferred) | higher-order SH = **view-dependent** color |
| `opacity` | per-Gaussian alpha (pre-activation) |
| `scale_*` (inferred) | anisotropic scale |
| `rot_0..rot_3` (`rot_2`,`rot_3` seen) | rotation **quaternion** |

> **Key quality insight.** `f_dc_` + (inferred) `f_rest_` means they use
> **spherical harmonics for color**, giving **view-dependent shading** that stays
> cheap enough for the client. This is almost certainly why their splat looks
> better than NIVA's current **flat-RGB** export.

### 2.4 FLAME + animation (protobuf)

- `flame_model.pb`, `flame_template.pb`, `flame.proto`, `flame_template.proto`,
  `animation.proto`.
- Messages recovered: `FlameTemplateResources` (+ `.version`), `TemplateMesh`,
  `TeethMesh` (+ `.version`), `EyelidDeformation`, `AnimationFrame`, `ShapeData`.
- FLAME math: `shapedirs_flat` (blendshape basis), `blendShapesPartial`, LBS
  weights (`Warning: LBS weights vertex count mismatch`) → **linear blend
  skinning** FLAME, standard.
- Per-frame animation fields: `expression`, `eye_lid`, pose, and
  `body_anchor_translation` (3 floats). This is the **lightweight motion data**
  streamed instead of video (Spatius' stated approach; NIVA Arc B parallel).

### 2.5 Teeth are a separate mesh

- `TeethMesh`, `loadSubdividedTeeth()`, `Subdivided teeth data`,
  `Error: Failed to extract teeth mesh from protobuf`.
- They model **teeth as a dedicated (subdividable) mesh**, not relying on FLAME's
  weak teeth region. **This directly addresses the teeth gap we saw in NIVA's
  GAGAvatar render.**

---

## 3. What this confirms vs. NIVA's current build

| Dimension | Them (avatarkit) | NIVA now | Gap / action |
|---|---|---|---|
| Representation | 3DGS, FLAME-bound | 3DGS, FLAME-bound (GAGAvatar) | aligned |
| Color | **SH (view-dependent)** | **flat RGB** | **adopt SH color** (biggest visual win) |
| Container | CSPLAT (zlib + CRC + version) | raw `.bin` (14 f32/gaussian) + json | add compressed, versioned container |
| Teeth | **separate TeethMesh** | FLAME-only (weak teeth) | add separate teeth mesh |
| Motion | protobuf `AnimationFrame` | (Arc B, not built) | mirror: expr+pose+eye_lid+anchor |
| Render | C++→WASM + WebGL/WebGPU | JS + WebGPU/WebGL2 | ours is fine for MVP; WASM later for perf |
| Server neural upsampler | **none** (pure on-device) | none in browser path | aligned (refinement is the quality lever) |

---

## 4. Recommended NIVA implementation plan (prioritised)

1. **Switch export color to spherical harmonics (SH).**
   - Store `f_dc` (DC, 3) + optional `f_rest` (degree-1 → 9 extra per channel).
   - Update the WebGPU/WebGL2 shaders to evaluate SH with the view direction.
   - Expected: the single biggest fidelity jump; closes much of the gap to
     their look. Degree 1 is a good cost/quality balance for the web.

2. **Add a separate teeth mesh.**
   - Load/attach a subdivided teeth mesh bound to the jaw region, composited with
     the Gaussian head. Fixes the open-mouth/teeth artifact we observed.

3. **Define a NIVA compressed container (our own CSPLAT-equivalent).**
   - Header: `magic` (`"NVSP"`), `uint16 version`, `uint32 vertex_count`,
     `uint8 compression`, field layout descriptor, `uint32 header_crc`.
   - Payload: zlib-compressed interleaved attributes (quantise where safe:
     fp16 positions/scales, int8 SH residuals). Target 5–10 MB like theirs.
   - Keep our current `.bin`+`.json` as the uncompressed/debug form; add a
     loader for the compressed form in the browser (DecompressionStream('gzip')
     is available in-browser, no extra deps).

4. **Protobuf animation frames (Arc B).**
   - `AnimationFrame { repeated float expression; float[6] pose; float[?] eye_lid;
     float[3] body_anchor_translation; }` streamed at ~25 fps over the data
     channel — exactly the Nova Sonic → ARTalk-FLAME → renderer path already in
     docs/plan.md.

5. **(Later) C++→WASM renderer** only if JS/WebGPU perf is insufficient on target
   devices. Not needed for the MVP.

---

## 5. Legal / licensing note

- The WASM is a third-party proprietary binary; we did **not** copy its code. All
  recovered items are **formats and architecture** (facts), re-implemented
  independently in NIVA.
- SH-color 3DGS, FLAME LBS, zlib containers, and protobuf motion are all
  **standard, publicly-documented techniques** — safe to implement from scratch.
- FLAME itself remains license-gated (see `docs/LICENSES.md`); GaussianAvatars'
  binding is NC — NIVA already plans an independent binding (`niva_offline/binding/`).

---

## 6. One-line takeaways

- Their quality edge over NIVA's current demo ≈ **SH view-dependent color** +
  **separate teeth mesh** + **per-subject refinement** — all reproducible.
- Their delivery model = **compressed, versioned splat container + protobuf
  motion, rendered 100% on-device** — no server neural net. NIVA's architecture
  already matches; we just need the SH-color + container + teeth upgrades.
