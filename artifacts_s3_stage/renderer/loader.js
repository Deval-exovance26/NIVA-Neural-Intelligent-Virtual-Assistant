// loader.js — parse the NIVA shared Gaussian export format in the browser.
//
// Format (see niva_offline/export/gaussian_format.py):
//   <name>.json : metadata (num_gaussians, stride_floats=14, field_order, conventions)
//   <name>.bin  : (N, 14) float32, little-endian, row-major
//                 columns = positions(3) rotations(4,wxyz) scales(3) opacities(1) colors(3)
//
// Returns typed sub-arrays (copies are avoided; we create views into one buffer
// then de-interleave into packed per-attribute arrays suitable for GPU upload).

export async function loadGaussianAvatar(baseUrl, name = "avatar") {
  const meta = await fetch(`${baseUrl}/${name}.json`).then((r) => {
    if (!r.ok) throw new Error(`failed to fetch ${name}.json: ${r.status}`);
    return r.json();
  });

  if (meta.format !== "niva-gaussian-avatar") {
    throw new Error(`unexpected format: ${meta.format}`);
  }
  const N = meta.num_gaussians;
  const stride = meta.stride_floats; // 14
  if (stride !== 14) throw new Error(`unexpected stride ${stride}`);

  const buf = await fetch(`${baseUrl}/${name}.bin`).then((r) => {
    if (!r.ok) throw new Error(`failed to fetch ${name}.bin: ${r.status}`);
    return r.arrayBuffer();
  });

  const flat = new Float32Array(buf);
  const expected = N * stride;
  if (flat.length !== expected) {
    throw new Error(
      `bin size mismatch: got ${flat.length} floats, expected ${expected} (N=${N} x ${stride})`
    );
  }

  // De-interleave into packed attribute arrays. The GPU buffer layout we use
  // packs each Gaussian as: pos.xyz, scale.xyz, color.rgb, opacity, quat.wxyz
  // => 14 floats/gaussian (reordered for shader convenience).
  const packed = new Float32Array(N * 14);
  for (let i = 0; i < N; i++) {
    const s = i * stride;
    const d = i * 14;
    // positions (0..2)
    packed[d + 0] = flat[s + 0];
    packed[d + 1] = flat[s + 1];
    packed[d + 2] = flat[s + 2];
    // scales (src 7..9) -> dst 3..5
    packed[d + 3] = flat[s + 7];
    packed[d + 4] = flat[s + 8];
    packed[d + 5] = flat[s + 9];
    // colors (src 11..13) -> dst 6..8
    packed[d + 6] = flat[s + 11];
    packed[d + 7] = flat[s + 12];
    packed[d + 8] = flat[s + 13];
    // opacity (src 10) -> dst 9
    packed[d + 9] = flat[s + 10];
    // quat wxyz (src 3..6) -> dst 10..13
    packed[d + 10] = flat[s + 3];
    packed[d + 11] = flat[s + 4];
    packed[d + 12] = flat[s + 5];
    packed[d + 13] = flat[s + 6];
  }

  // Compute a bounding sphere for camera framing.
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < N; i++) {
    const s = i * stride;
    cx += flat[s]; cy += flat[s + 1]; cz += flat[s + 2];
  }
  cx /= N; cy /= N; cz /= N;
  let radius = 0;
  for (let i = 0; i < N; i++) {
    const s = i * stride;
    const dx = flat[s] - cx, dy = flat[s + 1] - cy, dz = flat[s + 2] - cz;
    radius = Math.max(radius, Math.hypot(dx, dy, dz));
  }

  return { meta, N, packed, center: [cx, cy, cz], radius };
}
