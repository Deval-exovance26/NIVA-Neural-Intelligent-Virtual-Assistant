# web-renderer/

The **browser-side** of NIVA (Arc A, Tasks 10–11; consumed by Arc B). All avatar
rendering happens here, **client-side on the user's GPU** — the backend never renders
avatar video.

## Role (per docs/architecture.md)

- Load the exported avatar asset (packed Gaussians + binding tables + FLAME metadata) from S3.
- Render the FLAME-mesh-bound 3D Gaussian head via **WebGPU** (Three.js as host/scene layer).
- Apply the FLAME binding transform (Eq. 1) **per-frame on the GPU** to deform Gaussians from
  incoming FLAME param frames (expr/jaw/pose/eye at 25fps).
- Maintain an A/V sync buffer so lips align to the audio stream.
- Target: desktop **Chrome** with WebGPU. **WebGL2 fallback** for the Low tier.

## Live-runtime data path (Arc B)

```
Nova Sonic audio ─► Speaker (client)
FLAME param frames ─(WebRTC data channel)─► web-renderer
        │
        ▼ WebGPU splat renderer: apply FLAME binding transform per-frame on GPU
        ▼ A/V sync buffer aligns lips to audio
     NIVA avatar on screen
```

## Performance tiers (SOW targets)

| Tier   | Target FPS | Levers |
|--------|-----------|--------|
| High   | ~60       | full Gaussian count, full res |
| Medium | ~30       | Gaussian LOD, reduced res |
| Low    | 20–30     | WebGL2 fallback, aggressive LOD / frame pacing |

Stable 30 is preferred over unstable higher. Time-to-first-audio target < 1.5s (runtime-server owns the audio clock).

## Status

Scaffold only. Implementation is Arc A Task 10 (static neutral render) and Task 11
(browser-side FLAME animation). Tooling choice (Vite + TypeScript + Three.js r1xx with
WebGPURenderer) is deferred to those tasks; nothing is installed yet.
