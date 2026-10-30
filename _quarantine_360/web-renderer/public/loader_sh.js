// loader_sh.js — load the SH-color NIVA frontal asset.
// niva_frontal.json : { num_gaussians, stride_floats:11, has_sh, sh_degree, sh_coeffs, flip_y }
// niva_frontal.bin  : (N,11) f32  pos3 quat4(wxyz) scale3 opac1
// niva_frontal_sh.bin: (N,48) f32  16 SH coeffs x3 channels
// niva_frontal_anim.json: jaw_delta[N*3], blink_delta[N*3]
export async function loadSHAvatar(baseUrl, name = "niva_frontal") {
  const meta = await fetch(`${baseUrl}/${name}.json`).then(r => r.json());
  const N = meta.num_gaussians, stride = meta.stride_floats;
  const geo = new Float32Array(await fetch(`${baseUrl}/${name}.bin`).then(r => r.arrayBuffer()));
  const sh  = new Float32Array(await fetch(`${baseUrl}/${name}_sh.bin`).then(r => r.arrayBuffer()));
  if (geo.length !== N * stride) throw new Error(`geo size ${geo.length} != ${N*stride}`);
  const shCoeffs = meta.sh_coeffs || 16;
  if (sh.length !== N * shCoeffs * 3) throw new Error(`sh size ${sh.length} != ${N*shCoeffs*3}`);

  // base positions (for morph), bounding sphere
  const basePositions = new Float32Array(N * 3);
  let cx=0, cy=0, cz=0;
  for (let i=0;i<N;i++){ const s=i*stride;
    basePositions[i*3]=geo[s]; basePositions[i*3+1]=geo[s+1]; basePositions[i*3+2]=geo[s+2];
    cx+=geo[s]; cy+=geo[s+1]; cz+=geo[s+2]; }
  cx/=N; cy/=N; cz/=N;
  let radius=0; for (let i=0;i<N;i++){ const s=i*stride;
    const dx=geo[s]-cx,dy=geo[s+1]-cy,dz=geo[s+2]-cz; radius=Math.max(radius,Math.hypot(dx,dy,dz)); }

  // anim deltas
  let jawDelta=null, blinkDelta=null;
  try { const a=await fetch(`${baseUrl}/${name}_anim.json`).then(r=>r.ok?r.json():null);
    if(a){ jawDelta=new Float32Array(a.jaw_delta); blinkDelta=new Float32Array(a.blink_delta); } } catch(e){}

  return { meta, N, stride, geo, sh, shDegree: meta.sh_degree||3, shCoeffs,
           center:[cx,cy,cz], radius, needsYFlip: !meta.flip_y,
           jawDelta, blinkDelta, basePositions };
}
