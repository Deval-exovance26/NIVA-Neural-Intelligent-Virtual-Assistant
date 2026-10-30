// app_frontal.js — FRONTAL-ONLY NIVA avatar: fixed camera, independent blink,
// audio-driven lip-sync. No orbit/360. Loads the distilled frontal splat with
// per-gaussian jaw/blink deltas and morphs them each frame.

import { mat4 } from "./math.js";

const canvas = document.getElementById("gfx");
const statEl = document.getElementById("stat");
const badgeEl = document.getElementById("badge");
const setStat = (s) => (statEl && (statEl.textContent = s));
function fail(msg){ const e=document.getElementById("err"); if(e){e.style.display="grid"; document.getElementById("errmsg").innerHTML=msg;} }

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
}

// ---- independent blink driver ---------------------------------------------
// Returns blink weight 0..1. Eyes open most of the time; a quick blink every
// 2.5-6s, closing (~80ms) then opening (~120ms).
function makeBlink() {
  let next = performance.now() + 1500 + Math.random() * 2500;
  let phase = "idle", t0 = 0;
  const CLOSE = 80, OPEN = 130;
  return (now) => {
    if (phase === "idle" && now >= next) { phase = "closing"; t0 = now; }
    if (phase === "closing") {
      const u = (now - t0) / CLOSE;
      if (u >= 1) { phase = "opening"; t0 = now; return 1; }
      return u;
    }
    if (phase === "opening") {
      const u = (now - t0) / OPEN;
      if (u >= 1) { phase = "idle"; next = now + 2500 + Math.random() * 3500; return 0; }
      return 1 - u;
    }
    return 0;
  };
}

// ---- lip-sync driver: Web Audio amplitude envelope -> jaw weight -----------
async function makeLipsync(audioUrl) {
  let jaw = 0, analyser = null, data = null, playing = false;
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const btn = document.getElementById("playbtn");
  async function load() {
    const buf = await fetch(audioUrl).then(r => r.arrayBuffer()).then(b => ctx.decodeAudioData(b));
    return buf;
  }
  let audioBuf = null;
  try { audioBuf = await load(); } catch (e) { console.warn("lipsync audio load failed", e); }
  async function play() {
    if (!audioBuf || playing) return;
    if (ctx.state === "suspended") await ctx.resume();
    const src = ctx.createBufferSource(); src.buffer = audioBuf;
    analyser = ctx.createAnalyser(); analyser.fftSize = 1024;
    data = new Uint8Array(analyser.frequencyBinCount);
    src.connect(analyser); analyser.connect(ctx.destination);
    src.start(); playing = true;
    src.onended = () => { playing = false; analyser = null; jaw = 0; };
  }
  if (btn) btn.addEventListener("click", play);
  // auto-start attempt (may be blocked until user gesture)
  play().catch(()=>{});

  // ---- Lip-sync envelope (E1): fast attack, slow release, voiced jaw floor ----
  // Problem with the old driver: a single symmetric low-pass closed the jaw as
  // fast as it opened, so every short silence between words snapped the mouth
  // shut -> visible stalls/pauses. Fix: an asymmetric envelope that opens quickly
  // on voiced onset but closes slowly, plus a small jaw floor while speech is
  // active so the mouth never fully slams shut between syllables. Time-based
  // (per-ms) coefficients keep it frame-rate independent.
  const JAW_CAP   = 0.35;    // max fraction of full jaw-open morph (unchanged look)
  const ATTACK_MS = 45;      // time constant to OPEN  (fast / snappy)
  const RELEASE_MS= 160;     // time constant to CLOSE (slow / no snap)
  const VOICE_TH  = 0.06;    // amplitude above this = "voiced"
  const JAW_FLOOR = 0.10;    // min jaw (fraction of cap) while voiced -> no full stall
  const FLOOR_HOLD= 180;     // keep the floor this long (ms) after last voiced frame
  const NOISE_FLOOR = 0.03;  // ignore amplitude below this (room noise)

  let jawSm = 0;
  let lastMs = performance.now();
  let lastVoicedMs = -1e9;

  return () => {
    const now = performance.now();
    let dt = now - lastMs; lastMs = now;
    if (dt < 0) dt = 0; else if (dt > 100) dt = 100; // clamp after tab-switch stalls

    // --- raw amplitude from a low-mid speech band ---
    let amp = 0;
    if (analyser) {
      analyser.getByteFrequencyData(data);
      let s = 0, n = 24; for (let i = 2; i < 2 + n; i++) s += data[i];
      amp = s / (n * 255);                                  // 0..1
      if (amp < NOISE_FLOOR) amp = 0;                        // gate room noise
    }

    const voiced = amp > VOICE_TH;
    if (voiced) lastVoicedMs = now;
    const recentlyVoiced = (now - lastVoicedMs) < FLOOR_HOLD;

    // target jaw from amplitude, capped; apply a floor only while (recently) voiced
    let target = Math.max(0, Math.min(1, amp * 2.0)) * JAW_CAP;
    if (recentlyVoiced) target = Math.max(target, JAW_FLOOR * JAW_CAP);

    // asymmetric exponential smoothing: fast toward a higher target, slow toward lower
    const tc = (target > jawSm) ? ATTACK_MS : RELEASE_MS;
    const k = 1 - Math.exp(-dt / tc);                        // 0..1 per-frame coeff
    jawSm += (target - jawSm) * k;
    return jawSm;
  };
}

(async function main() {
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  try {
    const params = new URLSearchParams(location.search);
    const asset = params.get("asset") || "niva_frontal";
    const base = params.get("dir") || "./assets_frontal";
    const { createWebGL2Renderer } = await import("./renderer_webgl2.js");
    const r = await createWebGL2Renderer(canvas, base, asset, setStat);
    // Optional display-time sharpen (unsharp mask). Default 0.35; ?sharp=0 disables.
    const sharpParam = params.get("sharp");
    if (sharpParam !== null && r.setSharpen) r.setSharpen(parseFloat(sharpParam));
    badgeEl && (badgeEl.textContent = "WEBGL2");
    const { center, radius, N } = r.avatar;
    setStat(`${N.toLocaleString()} gaussians · frontal`);

    // FIXED FRONTAL camera = the distillation's training camera (colors were baked
    // for exactly this view). Falls back to a generic frontal camera if absent.
    const camMeta = r.avatar.meta && r.avatar.meta.camera;
    const UP = [0, 1, 0];
    const eye = camMeta ? camMeta.eye : [center[0], center[1], center[2] + radius * 2.6];
    const fwd = camMeta ? camMeta.forward : [0, 0, -1];
    const target = [eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2]];
    const fovy = camMeta ? (camMeta.fovy_deg * Math.PI) / 180 : 0.6;

    const blink = makeBlink();
    const lipsync = await makeLipsync((params.get("audio")) || "./common/test.wav");

    // Verification/debug hook: window.__forceMorph=[wJaw,wBlink] overrides drivers.
    window.__r = r;
    window.__forceMorph = null;
    // Opt-in lip-sync trace: set window.__jawTrace=[] to start recording {t,jaw}
    // samples (used to verify the attack/release envelope). Off by default.
    window.__jawTrace = null;

    function frame() {
      const now = performance.now();
      let wBlink = blink(now);
      let wJaw = lipsync();
      if (window.__jawTrace) { window.__jawTrace.push({ t: now, jaw: wJaw }); }
      if (window.__forceMorph) { wJaw = window.__forceMorph[0]; wBlink = window.__forceMorph[1]; }
      r.setMorph(wJaw, wBlink);

      const aspect = canvas.width / canvas.height;
      const view = mat4.lookAt(eye, target, UP);
      const proj = mat4.perspective(fovy, aspect, 0.01, 100);
      const viewProj = mat4.mul(proj, view);
      const focalY = canvas.height / (2 * Math.tan(fovy / 2));
      r.render({ view, viewProj, position: eye, camPos: eye,
                 viewport: [canvas.width, canvas.height], focal: [focalY, focalY] });
      requestAnimationFrame(frame);
    }
    frame();
  } catch (e) {
    console.error(e);
    fail(`<b>Could not start the frontal avatar.</b><br/>${e.message}`);
  }
})();
