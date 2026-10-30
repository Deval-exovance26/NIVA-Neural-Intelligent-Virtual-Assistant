// loader.js — parse the NIVA shared Gaussian export format in the browser.
//
// Format (see niva_offline/export/gaussian_format.py):
//   <name>.json : metadata (num_gaussians, stride_floats=14, field_order, conventions)
//   <name>.bin  : (N, 14) float32, little-endian, row-major
//                 columns = positions(3) rotations(4,wxyz) scales(3) opacities(1) colors(3)
//
// EXTENDED FORMAT for animation:
//   <name>_flame.json : FLAME mesh topology { vertices: [V,3], triangles: [T,3] }
//   <name>_binding.json : Per-gaussian binding data { triangle_index: [N], barycentric: [N,3] }
//
// Returns typed sub-arrays (copies are avoided; we create views into one buffer
// then de-interleave into packed per-attribute arrays suitable for GPU upload).
//
// ORIENTATION HANDLING:
// The renderer uses a canonical coordinate system where the camera's UP vector
// determines the Y orientation. Assets may be exported with different Y conventions:
//   - flip_y = false (or unset): Y+ is UP in the asset (GAGAvatar native) → camera UP = (0, -1, 0)
//   - flip_y = true: Y- is UP in the asset (already flipped during export) → camera UP = (0, 1, 0)
// The `needsYFlip` flag in the returned object indicates whether the camera needs to flip Y.

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

  // Determine Y orientation from metadata.
  // flip_y = true means the asset's Y coordinates are already negated (Y-down convention),
  // so the camera should NOT apply an additional flip (use UP = (0, 1, 0)).
  // flip_y = false or unset means the asset uses GAGAvatar native Y-up convention,
  // so the camera needs to flip Y (use UP = (0, -1, 0)).
  const needsYFlip = !meta.flip_y;

  // Try to load FLAME binding data (optional, for animation)
  let flameMesh = null;
  let bindings = null;
  
  try {
    const flameMeta = await fetch(`${baseUrl}/${name}_flame.json`).then((r) => {
      if (!r.ok) return null;
      return r.json();
    });
    
    if (flameMeta) {
      flameMesh = {
        vertices: new Float32Array(flameMeta.vertices),
        triangles: new Uint32Array(flameMeta.triangles),
        numVertices: flameMeta.vertices.length / 3,
        numTriangles: flameMeta.triangles.length / 3
      };
      console.log(`Loaded FLAME mesh: ${flameMesh.numVertices} vertices, ${flameMesh.numTriangles} triangles`);
    }
  } catch (e) {
    console.log("No FLAME mesh data available (optional for animation)");
  }
  
  try {
    const bindingMeta = await fetch(`${baseUrl}/${name}_binding.json`).then((r) => {
      if (!r.ok) return null;
      return r.json();
    });
    
    if (bindingMeta) {
      bindings = {
        triangle_index_per_gaussian: new Uint32Array(bindingMeta.triangle_index),
        barycentric_coords: bindingMeta.barycentric // Keep as array of arrays for now
      };
      console.log(`Loaded gaussian bindings: ${bindings.triangle_index_per_gaussian.length} gaussians`);
    }
  } catch (e) {
    console.log("No gaussian binding data available (optional for animation)");
  }

  // Try to load per-gaussian animation deltas (jaw-open + blink) for frontal avatar.
  // Format: niva_frontal_anim.json { num_gaussians, jaw_delta:[N*3], blink_delta:[N*3] }
  // These are WORLD-space position deltas (Y already flipped to match .bin).
  let jawDelta = null, blinkDelta = null, animFull = null;
  // Preferred: binary pos+quat+scale deltas (meta.anim). Position-only morphs leave
  // splats with stale orientation/scale when the jaw opens -> "double layer" + glitter.
  if (meta.anim && meta.anim.file) {
    const ab = await fetch(`${baseUrl}/${meta.anim.file}`).then((r) => (r.ok ? r.arrayBuffer() : null));
    if (ab && ab.byteLength === N * meta.anim.stride_floats * 4) {
      animFull = new Float32Array(ab); // per gaussian: jaw[dp3 dq4 ds3], blink[dp3 dq4 ds3]
      console.log(`Loaded full anim deltas (pos+quat+scale) for ${N} gaussians`);
    }
  }
  if (!animFull) try {
    const anim = await fetch(`${baseUrl}/${name}_anim.json`).then((r) => (r.ok ? r.json() : null));
    if (anim && anim.jaw_delta && anim.blink_delta) {
      jawDelta = new Float32Array(anim.jaw_delta);
      blinkDelta = new Float32Array(anim.blink_delta);
      console.log(`Loaded anim deltas for ${jawDelta.length / 3} gaussians (jaw + blink)`);
    }
  } catch (e) {
    console.log("No anim deltas (static avatar)");
  }

  // Keep a copy of the base (neutral) positions so morphs are always applied
  // relative to neutral, not accumulated.
  const basePositions = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    basePositions[i * 3 + 0] = packed[i * 14 + 0];
    basePositions[i * 3 + 1] = packed[i * 14 + 1];
    basePositions[i * 3 + 2] = packed[i * 14 + 2];
  }

  return {
    meta,
    N,
    packed,
    center: [cx, cy, cz],
    radius,
    needsYFlip,
    // Animation data (null if not available)
    flameMesh,
    bindings,
    jawDelta,
    blinkDelta,
    animFull,
    basePositions,
    basePacked: packed.slice(), // neutral pos/scale/quat for full morphs
  };
}
