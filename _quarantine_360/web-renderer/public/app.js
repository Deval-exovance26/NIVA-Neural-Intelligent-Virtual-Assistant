// app.js — cross-device NIVA avatar viewer.
//
// Backend selection (production, "runs on any device"):
//   1. WebGPU  (Chrome/Edge 113+, high-end) — renderer.js
//   2. WebGL2  (nearly all modern browsers incl. mobile Safari/Android) — renderer_webgl2.js
//   3. neither -> clear message.
//
// Both backends load the same exported Gaussian asset and use the same camera
// math, so the refined avatar (Task 8) drops in with no renderer changes.

import { mat4 } from "./math.js";

const canvas = document.getElementById("gfx");
const statEl = document.getElementById("stat");
const badgeEl = document.getElementById("badge");
const setStat = (s) => (statEl.textContent = s);

function fail(msg) {
  const e = document.getElementById("err");
  e.style.display = "grid";
  document.getElementById("errmsg").innerHTML = msg;
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
}

async function pickRenderer() {
  const params = new URLSearchParams(location.search);
  const forced = params.get("backend"); // ?backend=webgl2 to force
  const asset = params.get("asset") || "niva_base"; // ?asset=niva_refined
  const base = params.get("dir") || "./assets";
  if (forced !== "webgl2" && navigator.gpu) {
    try {
      const { createRenderer } = await import("./renderer.js");
      const r = await createRenderer(canvas, base, asset, setStat);
      r.backend = "webgpu";
      return r;
    } catch (e) {
      console.warn("WebGPU init failed, falling back to WebGL2:", e);
    }
  }
  const { createWebGL2Renderer } = await import("./renderer_webgl2.js");
  return await createWebGL2Renderer(canvas, base, asset, setStat);
}

(async function main() {
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  try {
    const r = await pickRenderer();
    badgeEl.textContent = r.backend.toUpperCase();
    const { center, radius, N, meta } = r.avatar;
    setStat(`${N.toLocaleString()} gaussians · ${r.backend}`);

    let yaw = 0, pitch = 0, dist = radius * 2.6;
    let dragging = false, lx = 0, ly = 0;

    const start = (x, y) => { dragging = true; lx = x; ly = y; };
    const move = (x, y) => {
      if (!dragging) return;
      yaw -= (x - lx) * 0.01;
      pitch -= (y - ly) * 0.01;
      pitch = Math.max(-1.3, Math.min(1.3, pitch));
      lx = x; ly = y;
    };
    const end = () => (dragging = false);

    canvas.addEventListener("pointerdown", (e) => { start(e.clientX, e.clientY); canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointermove", (e) => move(e.clientX, e.clientY));
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      dist *= 1 + Math.sign(e.deltaY) * 0.08;
      dist = Math.max(radius * 1.1, Math.min(radius * 8, dist));
    }, { passive: false });
    // touch pinch zoom
    let pinch0 = null;
    canvas.addEventListener("touchmove", (e) => {
      if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const d = Math.hypot(dx, dy);
        if (pinch0) { dist *= pinch0 / d; dist = Math.max(radius * 1.1, Math.min(radius * 8, dist)); }
        pinch0 = d;
        e.preventDefault();
      }
    }, { passive: false });
    canvas.addEventListener("touchend", () => (pinch0 = null));

    // Camera UP vector depends on the asset's Y convention:
    // - needsYFlip = true (flip_y = false/unset): Asset uses GAGAvatar native Y-up,
    //   so camera UP = (0, -1, 0) to flip Y and render upright.
    // - needsYFlip = false (flip_y = true): Asset was exported with Y already negated
    //   (Y-down convention), so camera UP = (0, 1, 0) to render upright.
    const UP = r.avatar.needsYFlip ? [0, -1, 0] : [0, 1, 0];
    let frames = 0, t0 = performance.now();

    function frame() {
      const aspect = canvas.width / canvas.height;
      const fovy = 0.6;
      const eye = [
        center[0] + dist * Math.cos(pitch) * Math.sin(yaw),
        center[1] + dist * Math.sin(pitch),
        center[2] + dist * Math.cos(pitch) * Math.cos(yaw),
      ];
      const view = mat4.lookAt(eye, center, UP);
      const proj = mat4.perspective(fovy, aspect, 0.01, 100);
      const viewProj = mat4.mul(proj, view);
      const focalY = canvas.height / (2 * Math.tan(fovy / 2));
      r.render({
        view, viewProj, camPos: eye,
        viewport: [canvas.width, canvas.height],
        focal: [focalY, focalY],
      });
      frames++;
      const now = performance.now();
      if (now - t0 > 500) {
        const fps = (frames * 1000) / (now - t0);
        badgeEl.textContent = `${r.backend.toUpperCase()} · ${fps.toFixed(0)} fps`;
        frames = 0; t0 = now;
      }
      requestAnimationFrame(frame);
    }
    frame();
  } catch (e) {
    console.error(e);
    fail(`<b>Could not start a renderer.</b><br/><br/>${e.message}<br/><br/>
      This device needs <code>WebGPU</code> (Chrome/Edge 113+) or <code>WebGL2</code>
      (most modern browsers). Try updating your browser.`);
  }
})();
