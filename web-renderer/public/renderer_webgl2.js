// renderer_webgl2.js — universal WebGL2 3D Gaussian Splatting renderer.
//
// This is the cross-device path (phones, tablets, older desktops): WebGL2 is
// far more widely supported than WebGPU. Same splatting math as the WebGPU
// renderer (EWA projection of the 3D covariance), implemented with instanced
// quads. Produces the same image so the two paths are interchangeable.
//
// DEPTH SORTING:
// For correct alpha compositing with high-opacity splats, we must render
// back-to-front (farthest first). This implementation maintains a sorted
// index array that's reordered by view-space depth whenever the camera moves.
//
// WORKER OFFLOAD:
// For frontal-only avatars with fixed camera, the heavy morph-blend + sort
// work is offloaded to a Web Worker. The main thread only uploads and draws.
//
// Public API mirrors renderer_webgpu.js:
//   createWebGL2Renderer(canvas, assetBase, assetName, statusCb)
//     -> { backend:'webgl2', avatar, render(camState), resize() }

import { loadGaussianAvatar } from "./loader.js";

const VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aCorner;      // unit quad corner [-1..1]
layout(location=1) in vec3 aPos;          // gaussian center
layout(location=2) in vec3 aScale;        // linear scale
layout(location=3) in vec3 aColor;        // rgb
layout(location=4) in float aOpacity;
layout(location=5) in vec4 aQuat;         // wxyz

uniform mat4 uView;
uniform mat4 uViewProj;
uniform vec2 uViewport;
uniform vec2 uFocal;

out vec3 vColor;
out float vOpacity;
out vec2 vConicXY;    // conic a,b (c via symmetry stored in vConicZ)
out float vConicZ;
out vec2 vDelta;      // offset from center in pixels

mat3 quatToMat3(vec4 q){
  float w=q.x,x=q.y,y=q.z,z=q.w;
  float n=inversesqrt(max(w*w+x*x+y*y+z*z,1e-12)); w*=n;x*=n;y*=n;z*=n;
  float xx=x*x,yy=y*y,zz=z*z,xy=x*y,xz=x*z,yz=y*z,wx=w*x,wy=w*y,wz=w*z;
  return mat3(
    1.0-2.0*(yy+zz), 2.0*(xy+wz),     2.0*(xz-wy),
    2.0*(xy-wz),     1.0-2.0*(xx+zz), 2.0*(yz+wx),
    2.0*(xz+wy),     2.0*(yz-wx),     1.0-2.0*(xx+yy)
  );
}

void main(){
  // 3D covariance Sigma = R S S^T R^T
  mat3 R = quatToMat3(aQuat);
  mat3 S = mat3(aScale.x,0.0,0.0, 0.0,aScale.y,0.0, 0.0,0.0,aScale.z);
  mat3 M = R*S;
  mat3 Sigma = M*transpose(M);

  // camera-space center
  vec4 viewC = uView * vec4(aPos,1.0);
  if (viewC.z > -0.01) { gl_Position = vec4(2.0,2.0,2.0,1.0); return; } // behind cam -> cull
  float tz = -viewC.z;
  float fx = uFocal.x, fy = uFocal.y;

  // Jacobian of perspective projection (EWA), gsplat/antimatter15 convention
  // Jacobian of u = fx*x/(-z), v = fy*y/(-z) (OpenGL camera looks down -Z).
  // GLSL mat3() takes COLUMNS: rows are (fx/tz, 0, fx*x/tz^2) and (0, fy/tz, fy*y/tz^2).
  mat3 J = mat3(
    fx/tz, 0.0, 0.0,
    0.0,   fy/tz, 0.0,
    (fx*viewC.x)/(tz*tz), (fy*viewC.y)/(tz*tz), 0.0
  );
  mat3 W = mat3(uView);
  mat3 T = J * W;
  mat3 cov = T * Sigma * transpose(T);

  // 2D covariance (add tiny antialias term)
  float c00 = cov[0][0] + 0.3;
  float c01 = cov[0][1];
  float c11 = cov[1][1] + 0.3;

  // conic = inverse of 2D covariance
  float det = c00*c11 - c01*c01;
  if (det <= 0.0) { gl_Position = vec4(2.0,2.0,2.0,1.0); return; }
  float invDet = 1.0/det;
  vec3 conic = vec3(c11*invDet, -c01*invDet, c00*invDet);

  // quad half-extent in pixels = 3 sigma of the 2D covariance eigenvalues
  float mid = 0.5*(c00+c11);
  float disc = sqrt(max(mid*mid-det, 0.0));
  float l1 = mid+disc, l2 = max(mid-disc, 0.0);
  float rad = ceil(3.0*sqrt(max(l1,l2)));

  vec4 clip = uViewProj * vec4(aPos,1.0);
  vec2 deltaPix = aCorner * rad;                       // pixel offset for this corner
  vec2 ndc = vec2(2.0*deltaPix.x/uViewport.x, 2.0*deltaPix.y/uViewport.y) * clip.w;

  gl_Position = vec4(clip.xy + ndc, clip.z, clip.w);
  vColor = aColor;
  vOpacity = aOpacity;
  vConicXY = conic.xy;
  vConicZ = conic.z;
  vDelta = deltaPix;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vColor;
in float vOpacity;
in vec2 vConicXY;
in float vConicZ;
in vec2 vDelta;
out vec4 fragColor;
void main(){
  // power = -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2) using the conic
  float dx = vDelta.x, dy = vDelta.y;
  float power = -0.5*(vConicXY.x*dx*dx + vConicZ*dy*dy) - vConicXY.y*dx*dy;
  if (power > 0.0) discard;
  float alpha = min(0.99, vOpacity * exp(power));
  if (alpha < 0.004) discard;
  fragColor = vec4(vColor*alpha, alpha); // premultiplied
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error("shader compile: " + gl.getShaderInfoLog(s));
  }
  return s;
}

// ============================================================================
// DEPTH SORTING UTILITIES
// ============================================================================

/**
 * Compute view-space depths for all gaussians.
 * @param {Float32Array} packed - Original packed data (N*14 floats)
 * @param {Float32Array} viewMatrix - 4x4 view matrix (column-major)
 * @param {Float32Array} out - Output depth array (N floats)
 */
function computeDepths(packed, viewMatrix, out) {
  const N = packed.length / 14;
  // Extract view matrix elements for manual transform (faster than per-iteration mat4 multiply)
  const v00 = viewMatrix[0], v01 = viewMatrix[1], v02 = viewMatrix[2], v03 = viewMatrix[3];
  const v10 = viewMatrix[4], v11 = viewMatrix[5], v12 = viewMatrix[6], v13 = viewMatrix[7];
  const v20 = viewMatrix[8], v21 = viewMatrix[9], v22 = viewMatrix[10], v23 = viewMatrix[11];
  const v30 = viewMatrix[12], v31 = viewMatrix[13], v32 = viewMatrix[14], v33 = viewMatrix[15];
  
  for (let i = 0; i < N; i++) {
    const base = i * 14;
    const px = packed[base + 0];
    const py = packed[base + 1];
    const pz = packed[base + 2];
    // View-space Z = row 2 of the COLUMN-MAJOR view matrix: m[2], m[6], m[10], m[14]
    const vz = v02*px + v12*py + v22*pz + v32;
    out[i] = -vz; // Negate because we want positive depth in front of camera
  }
}

/**
 * Sort indices by depth (back-to-front: farthest first).
 * Uses Array.sort with a depth array for comparison.
 * @param {Uint32Array} indices - Index array to sort in-place
 * @param {Float32Array} depths - Depth values for each gaussian
 */
function sortIndicesByDepth(indices, depths) {
  // Convert to regular arrays for sort (faster than typed array sort in most JS engines)
  const arr = Array.from(indices);
  arr.sort((a, b) => depths[b] - depths[a]); // Farthest first (high depth first)
  for (let i = 0; i < arr.length; i++) {
    indices[i] = arr[i];
  }
}

/**
 * Reorder packed data according to sorted indices.
 * @param {Float32Array} src - Original packed data
 * @param {Uint32Array} indices - Sorted indices
 * @param {Float32Array} dst - Destination buffer for reordered data
 */
function reorderPackedData(src, indices, dst) {
  const N = indices.length;
  const stride = 14;
  for (let i = 0; i < N; i++) {
    const srcIdx = indices[i];
    const dstBase = i * stride;
    const srcBase = srcIdx * stride;
    for (let j = 0; j < stride; j++) {
      dst[dstBase + j] = src[srcBase + j];
    }
  }
}

/**
 * Check if camera has moved significantly since last sort.
 * Uses position and forward vector dot product for threshold.
 */
function cameraMoved(cam, lastCamPos, lastCamFwd, threshold) {
  if (!lastCamPos) return true;
  
  const pos = cam.position;
  const dx = pos[0] - lastCamPos[0];
  const dy = pos[1] - lastCamPos[1];
  const dz = pos[2] - lastCamPos[2];
  const posDelta = Math.sqrt(dx*dx + dy*dy + dz*dz);
  
  // Check forward direction change via view matrix
  // Forward vector is -Z column of view matrix: (view[2], view[6], view[10])
  const fwd0 = -cam.view[2];
  const fwd1 = -cam.view[6];
  const fwd2 = -cam.view[10];
  const fDot = fwd0*lastCamFwd[0] + fwd1*lastCamFwd[1] + fwd2*lastCamFwd[2];
  
  return posDelta > threshold || fDot < 0.99; // 0.99 ≈ 8 degree threshold
}

// ============================================================================

export async function createWebGL2Renderer(canvas, assetBase, assetName, statusCb) {
  const gl = canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: true, antialias: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error("WebGL2 not available");

  statusCb && statusCb("loading avatar…");
  const avatar = await loadGaussianAvatar(assetBase, assetName);
  statusCb && statusCb(`uploading ${avatar.N.toLocaleString()} gaussians (WebGL2)…`);

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("program link: " + gl.getProgramInfoLog(prog));
  }

  // unit quad
  const quad = new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]);
  const quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

  // ==========================================================================
  // DEPTH SORTING SETUP
  // ==========================================================================
  // Maintain sorted order for back-to-front rendering.
  // - indices: permutation array mapping sorted-order -> original-order
  // - sortedPacked: reordered copy of instance data for GPU upload
  // - depths: scratch buffer for view-space depth values
  const N = avatar.N;
  const stride = 14;
  
  // Initialize indices as identity (0, 1, 2, ..., N-1)
  const indices = new Uint32Array(N);
  for (let i = 0; i < N; i++) indices[i] = i;
  
  // Sorted copy of packed data (will be uploaded to GPU)
  const sortedPacked = new Float32Array(avatar.packed.length);
  
  // Scratch buffers for sorting
  const depths = new Float32Array(N);
  let lastCamPos = null;
  let lastCamFwd = [0, 0, -1];
  
  // Sort threshold: re-sort when camera moves more than this distance
  // or rotates more than ~8 degrees. Tuned for ~160k splats.
  const SORT_THRESHOLD = 0.01;
  
  // Initial sort (will be done on first render with valid camera)
  let needsSort = true;
  
  // ==========================================================================

  // per-instance interleaved buffer (stride 14 floats: pos3 scale3 color3 op1 quat4)
  // NOTE: Using DYNAMIC_DRAW because we'll update this buffer when sort order changes
  const instBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
  gl.bufferData(gl.ARRAY_BUFFER, avatar.packed.length * 4, gl.DYNAMIC_DRAW);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  // corner
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  // instance attrs
  const FB = 4, strideBytes = stride * FB;
  gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
  const setup = (loc, size, offsetFloats) => {
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, strideBytes, offsetFloats * FB);
    gl.vertexAttribDivisor(loc, 1);
  };
  setup(1, 3, 0);   // pos
  setup(2, 3, 3);   // scale
  setup(3, 3, 6);   // color
  setup(4, 1, 9);   // opacity
  setup(5, 4, 10);  // quat
  gl.bindVertexArray(null);

  const uView = gl.getUniformLocation(prog, "uView");
  const uViewProj = gl.getUniformLocation(prog, "uViewProj");
  const uViewport = gl.getUniformLocation(prog, "uViewport");
  const uFocal = gl.getUniformLocation(prog, "uFocal");

  // Premultiplied over-compositing: with back-to-front order,
  // blendFunc(ONE, ONE_MINUS_SRC_ALPHA) gives correct alpha blending.
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.disable(gl.DEPTH_TEST);

  // ==========================================================================
  // FLOAT ACCUMULATION TARGET
  // The distilled splat uses overbright colors (>1) that later Gaussians cancel.
  // An RGBA8 canvas clamps to [0,1] on every blend and breaks that cancellation
  // (gray patches). gsplat accumulates in float, so we do too, then composite.
  // ==========================================================================
  const hasFloatRT = !!(gl.getExtension("EXT_color_buffer_float") || gl.getExtension("EXT_color_buffer_half_float"));
  const BG = [0.06, 0.07, 0.09];
  let accTex = null, accFbo = null, accW = 0, accH = 0, compProg = null, uAccTex = null, uBg = null, uSharp = null;
  let sharpAmount = 0.35; // unsharp-mask strength (0 = off). Tunable via setSharpen / ?sharp=
  if (hasFloatRT) {
    const CVS = `#version 300 es
    const vec2 P[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    out vec2 vUv;
    void main(){ vec2 p = P[gl_VertexID]; vUv = p*0.5+0.5; gl_Position = vec4(p,0.0,1.0); }`;
    // TRANSPARENT CANVAS: Output alpha = c.a (accumulated gaussian alpha) so that
    // outside the avatar (c.a==0) the canvas is transparent, letting the body
    // photo layer behind show through. The RGB is kept premultiplied.
    const CFS = `#version 300 es
    precision highp float;
    uniform sampler2D uAcc; uniform vec3 uBg; uniform float uSharp; in vec2 vUv; out vec4 o;
    void main(){
      vec4 c = texture(uAcc, vUv);
      vec3 rgb = c.rgb;
      // Optional unsharp mask: boost high-frequency detail (hair/skin edges) at
      // display time. uSharp=0 => identity (no change, cannot regress the model).
      // Applied on the premultiplied accumulation so it does not ring on the
      // transparent background (weighted by neighbour alpha).
      if (uSharp > 0.0) {
        vec2 px = 1.0 / vec2(textureSize(uAcc, 0));
        vec4 l = texture(uAcc, vUv + vec2(-px.x, 0.0));
        vec4 r = texture(uAcc, vUv + vec2( px.x, 0.0));
        vec4 u = texture(uAcc, vUv + vec2(0.0, -px.y));
        vec4 d = texture(uAcc, vUv + vec2(0.0,  px.y));
        vec3 lap = 4.0*rgb - l.rgb - r.rgb - u.rgb - d.rgb;
        float wa = min(min(l.a, r.a), min(u.a, d.a)); // only sharpen inside the avatar
        rgb = rgb + uSharp * lap * wa;
      }
      // TRANSPARENT OUTPUT: premultiplied alpha blending with the body photo behind.
      // Where gaussians exist (c.a > 0), output the avatar color; where c.a == 0,
      // output fully transparent so the CSS-layered body image shows through.
      // We keep the color premultiplied and output alpha = c.a.
      o = vec4(clamp(rgb, 0.0, 1.0), c.a);
    }`;
    compProg = gl.createProgram();
    gl.attachShader(compProg, compile(gl, gl.VERTEX_SHADER, CVS));
    gl.attachShader(compProg, compile(gl, gl.FRAGMENT_SHADER, CFS));
    gl.linkProgram(compProg);
    if (!gl.getProgramParameter(compProg, gl.LINK_STATUS)) throw new Error("composite link: " + gl.getProgramInfoLog(compProg));
    uAccTex = gl.getUniformLocation(compProg, "uAcc");
    uBg = gl.getUniformLocation(compProg, "uBg");
    uSharp = gl.getUniformLocation(compProg, "uSharp");
  } else {
    console.warn("No float render target; overbright colors will clamp (possible gray patches).");
  }
  function ensureAcc(w, h) {
    if (accTex && accW === w && accH === h) return;
    if (accTex) { gl.deleteTexture(accTex); gl.deleteFramebuffer(accFbo); }
    accTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, accTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    accFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, accFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, accTex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("float FBO incomplete");
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    accW = w; accH = h;
  }
  // ==========================================================================
  // WEB WORKER FOR MORPH + SORT OFFLOAD
  // ==========================================================================
  // For frontal-only avatars with fixed camera, offload the heavy morph-blend
  // + depth-sort + reorder to a Web Worker. Main thread only uploads and draws.
  let morphWorker = null;
  let workerReady = false;
  let pendingFrameId = 0;
  let currentViewMatrix = null;  // Current view matrix sent to worker

  // Main-thread-owned buffer pool. We LEND a free buffer to the worker with each
  // frame request (transferred in); the worker fills it and transfers it back.
  // This avoids detached-buffer corruption: a buffer is owned by exactly one side
  // at a time. `uploadBuf` is the most recently returned sorted buffer to draw.
  const BUF_POOL = 3;
  const freeBuffers = [];        // ArrayBuffers currently owned by the main thread, idle
  for (let i = 0; i < BUF_POOL; i++) freeBuffers.push(new Float32Array(N * 14).buffer);
  let uploadView = null;         // Float32Array view over the latest sorted buffer to upload
  let haveNewUpload = false;     // true when a fresh buffer arrived since last upload

  // Counter for main-thread sorts (must stay 0 in worker mode)
  window.__mainThreadSorts = 0;

  // Try to init worker (fallback to sync path if unavailable)
  if (avatar.animFull && typeof Worker !== 'undefined') {
    try {
      // Classic worker (not module) for compatibility with python http.server
      morphWorker = new Worker('./morph_worker.js');

      morphWorker.onmessage = function(e) {
        const msg = e.data;
        if (msg.type === 'ready') {
          workerReady = true;
          console.log('morph_worker ready');
          return;
        }
        if (msg.type === 'frame') {
          // Worker returned our lent buffer, now filled with sorted data.
          // Adopt it for upload; the previously-held upload buffer goes back to
          // the free pool so it can be lent again.
          if (uploadView) freeBuffers.push(uploadView.buffer);
          uploadView = new Float32Array(msg.sorted);
          haveNewUpload = true;
          return;
        }
      };

      morphWorker.onerror = function(e) {
        console.warn('morph_worker error, falling back to sync path:', e.message);
        try { morphWorker.terminate(); } catch (_) {}
        morphWorker = null;
        workerReady = false;
      };

      // Give the worker its OWN copies of base + anim data (transfer copies so the
      // main-thread fallback keeps avatar.basePacked/animFull intact).
      const baseCopy = avatar.basePacked.slice().buffer;
      const animCopy = avatar.animFull.slice().buffer;
      morphWorker.postMessage(
        { type: 'init', basePacked: baseCopy, animFull: animCopy, N: N },
        [baseCopy, animCopy]
      );

    } catch (e) {
      console.warn('Failed to create morph_worker, falling back to sync path:', e.message);
      morphWorker = null;
      workerReady = false;
    }
  }
  
  // ==========================================================================
  // SYNC FALLBACK: morph + sort on main thread (for when worker unavailable)
  // ==========================================================================
  function syncMorphAndSort(wJaw, wBlink, cam) {
    const af = avatar.animFull;
    if (af) {
      // Full morph (stride 29): base_logit | jaw[dp3 dq4 ds3 drgb3 dlogit1] | blink[same].
      // Blends position, rotation (nlerp), scale, color and opacity so the neutral
      // look stays clean and open-mouth / closed-eye use their own learned appearance.
      const packed = avatar.packed, b = avatar.basePacked;
      const j = wJaw, k = wBlink;
      for (let i = 0; i < N; i++) {
        const p = i * 14, a = i * 29, J = a + 1, B = a + 15;
        packed[p + 0] = b[p + 0] + j * af[J + 0] + k * af[B + 0];
        packed[p + 1] = b[p + 1] + j * af[J + 1] + k * af[B + 1];
        packed[p + 2] = b[p + 2] + j * af[J + 2] + k * af[B + 2];
        const qw = b[p + 10] + j * af[J + 3] + k * af[B + 3];
        const qx = b[p + 11] + j * af[J + 4] + k * af[B + 4];
        const qy = b[p + 12] + j * af[J + 5] + k * af[B + 5];
        const qz = b[p + 13] + j * af[J + 6] + k * af[B + 6];
        const inv = 1 / (Math.hypot(qw, qx, qy, qz) || 1);
        packed[p + 10] = qw * inv; packed[p + 11] = qx * inv;
        packed[p + 12] = qy * inv; packed[p + 13] = qz * inv;
        packed[p + 3] = b[p + 3] + j * af[J + 7] + k * af[B + 7];
        packed[p + 4] = b[p + 4] + j * af[J + 8] + k * af[B + 8];
        packed[p + 5] = b[p + 5] + j * af[J + 9] + k * af[B + 9];
        packed[p + 6] = Math.max(0, b[p + 6] + j * af[J + 10] + k * af[B + 10]);
        packed[p + 7] = Math.max(0, b[p + 7] + j * af[J + 11] + k * af[B + 11]);
        packed[p + 8] = Math.max(0, b[p + 8] + j * af[J + 12] + k * af[B + 12]);
        packed[p + 9] = 1 / (1 + Math.exp(-(af[a] + j * af[J + 13] + k * af[B + 13])));
      }
    } else {
      const jd = avatar.jawDelta, bd = avatar.blinkDelta, base = avatar.basePositions;
      if (!jd || !base) return;
      const packed = avatar.packed;
      for (let i = 0; i < N; i++) {
        const p = i * 14, t = i * 3;
        packed[p + 0] = base[t + 0] + wJaw * jd[t + 0] + wBlink * bd[t + 0];
        packed[p + 1] = base[t + 1] + wJaw * jd[t + 1] + wBlink * bd[t + 1];
        packed[p + 2] = base[t + 2] + wJaw * jd[t + 2] + wBlink * bd[t + 2];
      }
    }
    
    // Check if camera moved enough to warrant re-sorting
    if (cameraMoved(cam, lastCamPos, lastCamFwd, SORT_THRESHOLD)) {
      needsSort = true;
      lastCamPos = cam.position ? [...cam.position] : [0, 0, 0];
      lastCamFwd = [-cam.view[2], -cam.view[6], -cam.view[10]];
    }
    
    if (needsSort) {
      // Increment main-thread sort counter
      window.__mainThreadSorts++;
      
      // Compute view-space depths
      computeDepths(avatar.packed, cam.view, depths);
      
      // Sort indices back-to-front (farthest first)
      sortIndicesByDepth(indices, depths);
      
      // Reorder packed data into sortedPacked buffer
      reorderPackedData(avatar.packed, indices, sortedPacked);
      
      needsSort = false;
    }
  }
  // ==========================================================================
  
  return {
    backend: "webgl2",
    avatar,
    resize() { gl.viewport(0, 0, canvas.width, canvas.height); },
    // Display-time unsharp mask strength (0 = off). Does not alter the model;
    // boosts perceived hair/skin detail. Tunable live for A/B.
    setSharpen(a) { sharpAmount = Math.max(0, Math.min(1.5, a)); },
    get sharpen() { return sharpAmount; },
    // Apply per-gaussian morph: pos = base + wJaw*jawDelta + wBlink*blinkDelta.
    // In worker mode, sends weights to worker. In fallback, does sync morph+sort.
    setMorph(wJaw, wBlink) {
      // Store for render() / fallback
      this._pendingWJaw = wJaw;
      this._pendingWBlink = wBlink;

      // Worker path: lend a free buffer along with the frame request. If no free
      // buffer is available (worker still busy with the last few), skip posting
      // this frame — the worker will catch up; we never queue unbounded work and
      // never block the main thread.
      if (workerReady && morphWorker) {
        if (freeBuffers.length === 0) return;      // worker saturated; drop this tick
        const lent = freeBuffers.pop();            // ArrayBuffer, main thread hands off
        const vm = (currentViewMatrix ? currentViewMatrix.slice() : new Float32Array(16));
        morphWorker.postMessage(
          { type: 'frame', wJaw, wBlink, viewMatrix: vm, buffer: lent, frameId: ++pendingFrameId },
          [lent, vm.buffer]
        );
        return;
      }
      // Fallback: sync morph+sort happens in render()
    },
    render(cam) {
      // Keep the view matrix for the next worker frame request.
      currentViewMatrix = cam.view;

      if (workerReady && morphWorker) {
        // Worker path. Upload a freshly-returned sorted buffer if we have one;
        // otherwise re-draw the last uploaded buffer (no blank, no stall).
        if (haveNewUpload && uploadView) {
          gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, uploadView);
          haveNewUpload = false;
        }
        // else: GPU buffer still holds the previous frame's data — fine for a
        // fixed-camera talking head; the next worker result arrives shortly.
      } else {
        // Fallback: sync morph + sort on main thread
        const wJaw = this._pendingWJaw !== undefined ? this._pendingWJaw : 0;
        const wBlink = this._pendingWBlink !== undefined ? this._pendingWBlink : 0;
        if (wJaw === this._lj && wBlink === this._lb && !needsSort) {
          // already up to date
        } else {
          this._lj = wJaw;
          this._lb = wBlink;
          syncMorphAndSort(wJaw, wBlink, cam);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, sortedPacked);
      }
      
      const w = canvas.width, h = canvas.height;
      if (hasFloatRT) {
        ensureAcc(w, h);
        gl.bindFramebuffer(gl.FRAMEBUFFER, accFbo);
        gl.viewport(0, 0, w, h);
        gl.clearColor(0, 0, 0, 0);
      } else {
        gl.viewport(0, 0, w, h);
        gl.clearColor(BG[0], BG[1], BG[2], 1);
      }
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.useProgram(prog);
      gl.uniformMatrix4fv(uView, false, cam.view);
      gl.uniformMatrix4fv(uViewProj, false, cam.viewProj);
      gl.uniform2fv(uViewport, cam.viewport);
      gl.uniform2fv(uFocal, cam.focal);
      gl.bindVertexArray(vao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, avatar.N);
      gl.bindVertexArray(null);

      if (hasFloatRT) {
        // composite float accumulation over the background onto the canvas
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, w, h);
        gl.disable(gl.BLEND);
        gl.useProgram(compProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, accTex);
        gl.uniform1i(uAccTex, 0);
        gl.uniform3fv(uBg, BG);
        if (uSharp) gl.uniform1f(uSharp, sharpAmount);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
    },
  };
}
