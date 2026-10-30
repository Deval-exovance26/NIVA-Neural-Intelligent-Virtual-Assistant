// morph_worker.js — Web Worker that offloads the per-frame Gaussian
// morph-blend + depth-sort + reorder from the main thread. Frontal-only NIVA
// avatar. Moving this work off the main thread stops the per-frame stall that
// caused the whole avatar to "refresh"/flicker under load.
//
// CORRECTNESS: to stay PIXEL-EQUIVALENT to the single-threaded v3 renderer we
// reproduce its math exactly AND re-sort every frame (the single-threaded path
// re-sorts on every morph). We do NOT throttle the sort — throttling left a
// stale depth order that visibly changed the eye region during blink. The whole
// point is to keep the identical result but compute it in the background.
//
// Protocol:
//   {type:'init', basePacked(ArrayBuffer), animFull(ArrayBuffer), N} -> {type:'ready'}
//   {type:'frame', wJaw, wBlink, viewMatrix(Float32Array), buffer(ArrayBuffer), frameId}
//        -> {type:'frame', sorted(ArrayBuffer), frameId}   (buffer transferred back)
//
// Buffer ownership: the MAIN THREAD owns the pool of output buffers and lends one
// to the worker with each frame request (transferred in). The worker fills it and
// transfers it back. The worker never allocates per frame and never holds a buffer
// the main thread also holds — this avoids detached-buffer corruption.

let basePacked = null;   // Float32Array N*14 (neutral)
let animFull = null;     // Float32Array N*29 (jaw+blink deltas)
let N = 0;

let indices = null;      // Uint32Array N (current permutation)
let depths = null;       // Float32Array N (scratch)
let morphed = null;      // Float32Array N*14 (scratch: morphed-but-unsorted)

function computeDepths(packed, vm, out) {
  // view matrix column-major; view-space z uses row 2: m[2],m[6],m[10],m[14]
  const v02 = vm[2], v12 = vm[6], v22 = vm[10], v32 = vm[14];
  for (let i = 0; i < N; i++) {
    const b = i * 14;
    out[i] = -(v02 * packed[b] + v12 * packed[b + 1] + v22 * packed[b + 2] + v32);
  }
}

function sortIndicesByDepth(idx, d) {
  // match the single-threaded renderer exactly (Array.from + comparator, farthest first)
  const arr = Array.from(idx);
  arr.sort((a, b) => d[b] - d[a]);
  for (let i = 0; i < arr.length; i++) idx[i] = arr[i];
}

// Exact reproduction of renderer_webgl2.js setMorph() full-morph (29-float) math.
function applyMorphBlend(wJaw, wBlink, out) {
  const b = basePacked, af = animFull, j = wJaw, k = wBlink;
  for (let i = 0; i < N; i++) {
    const p = i * 14, a = i * 29, J = a + 1, B = a + 15;
    out[p + 0] = b[p + 0] + j * af[J + 0] + k * af[B + 0];
    out[p + 1] = b[p + 1] + j * af[J + 1] + k * af[B + 1];
    out[p + 2] = b[p + 2] + j * af[J + 2] + k * af[B + 2];
    const qw = b[p + 10] + j * af[J + 3] + k * af[B + 3];
    const qx = b[p + 11] + j * af[J + 4] + k * af[B + 4];
    const qy = b[p + 12] + j * af[J + 5] + k * af[B + 5];
    const qz = b[p + 13] + j * af[J + 6] + k * af[B + 6];
    const inv = 1 / (Math.hypot(qw, qx, qy, qz) || 1);
    out[p + 10] = qw * inv; out[p + 11] = qx * inv;
    out[p + 12] = qy * inv; out[p + 13] = qz * inv;
    out[p + 3] = b[p + 3] + j * af[J + 7] + k * af[B + 7];
    out[p + 4] = b[p + 4] + j * af[J + 8] + k * af[B + 8];
    out[p + 5] = b[p + 5] + j * af[J + 9] + k * af[B + 9];
    out[p + 6] = Math.max(0, b[p + 6] + j * af[J + 10] + k * af[B + 10]);
    out[p + 7] = Math.max(0, b[p + 7] + j * af[J + 11] + k * af[B + 11]);
    out[p + 8] = Math.max(0, b[p + 8] + j * af[J + 12] + k * af[B + 12]);
    out[p + 9] = 1 / (1 + Math.exp(-(af[a] + j * af[J + 13] + k * af[B + 13])));
  }
}

function reorder(src, idx, dst) {
  const stride = 14;
  for (let i = 0; i < N; i++) {
    const s = idx[i] * stride, dBase = i * stride;
    for (let f = 0; f < stride; f++) dst[dBase + f] = src[s + f];
  }
}

self.onmessage = function (e) {
  const msg = e.data;

  if (msg.type === 'init') {
    basePacked = new Float32Array(msg.basePacked);
    animFull = new Float32Array(msg.animFull);
    N = msg.N;
    indices = new Uint32Array(N);
    for (let i = 0; i < N; i++) indices[i] = i;
    depths = new Float32Array(N);
    morphed = new Float32Array(N * 14);
    self.postMessage({ type: 'ready' });
    return;
  }

  if (msg.type === 'frame') {
    const { wJaw, wBlink, viewMatrix, buffer, frameId } = msg;
    const sorted = new Float32Array(buffer); // main-thread-owned buffer, lent to us

    // 1) morph into scratch
    applyMorphBlend(wJaw, wBlink, morphed);
    // 2) depths from the morphed positions, then sort (every frame = pixel-exact)
    computeDepths(morphed, viewMatrix, depths);
    sortIndicesByDepth(indices, depths);
    // 3) reorder morphed -> the lent buffer
    reorder(morphed, indices, sorted);

    // 4) hand the buffer back (transfer so no copy)
    self.postMessage({ type: 'frame', sorted: sorted.buffer, frameId }, [sorted.buffer]);
    return;
  }
};
