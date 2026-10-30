# NIVA WebGPU Avatar Demo — Stage 1 (run guide)

Renders the **real female NIVA avatar** (GAGAvatar base, 180,255 Gaussians) in the
browser via WebGPU. No build step and no Node required — it's plain ES modules +
WGSL, served statically.

## Files (`web-renderer/public/`)
- `index_frontal.html` — **Frontal-only avatar** (primary product): blink + lip-sync, fixed camera
- `renderer_webgl2.js` — WebGL2 3D Gaussian splat pipeline (optimized for all devices)
- `loader.js` — parses the shared `.bin`/`.json` export format (supports animation deltas)
- `math.js` — mat4 / quaternion helpers
- `assets_frontal/niva_frontal.bin` + `niva_frontal.json` — the frontal avatar export
- `assets_frontal/niva_frontal_anim.bin` — per-gaussian jaw + blink morphs

## Run

```bash
cd web-renderer/public
python3 -m http.server 8000
```

Then open **http://localhost:8000/index_frontal.html** in any modern browser (Chrome, Safari, Edge, Firefox).
Serve over `http://localhost`, not `file://` (module + fetch CORS).

## What you should see
NIVA's head (female), front-facing with realistic blink and audio-driven lip-sync. Jaw and eye motion are controlled by independent drivers, not orbit camera input.

## Known limitations (Frontal-only product)
- **Fixed camera:** the camera is fixed to the training view (frontal). This ensures baked-in colors render as trained. No orbit/head-tracking.
- **Small head motion supported:** blink + jaw/lip-sync morphs via audio. No full-body or large yaw tracking.
- **Color baked for frontal view:** per-Gaussian RGB is optimized for the frontal camera angle (the training view). Colors are highest quality when viewed straight-on.

## Validation without a browser
`python -m niva_offline.export.validate_render --export-dir <dir> --name niva_base
--out front.png` renders the same export with a numpy splatter using the SAME
camera conventions — used to confirm identity + orientation before the browser.
