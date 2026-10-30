// renderer.js — minimal WebGPU 3D Gaussian Splatting renderer for NIVA.
//
// Pipeline: one instanced quad per Gaussian. The vertex shader computes the
// screen-space 2D covariance from the 3D covariance (built from scale+rotation),
// sizes the quad accordingly, and the fragment shader applies a Gaussian falloff
// weighted by opacity. Back-to-front alpha blending.
//
// Coordinate conventions match the Python export (see gaussian_format.py):
//   positions world-space (GAGAvatar/FLAME), quaternion wxyz, linear scales.
//
// This is the Stage-1 MVP: static render of the real female avatar with
// mouse-orbit. Known limitation: per-pixel depth sorting is approximate
// (we sort instances CPU-side by view depth each frame).

import { loadGaussianAvatar } from "./loader.js";
import { mat4, quatToMat3 } from "./math.js";

const WGSL = /* wgsl */ `
struct Camera {
  viewProj : mat4x4<f32>,
  view     : mat4x4<f32>,
  camPos   : vec4<f32>,
  viewport : vec2<f32>,
  focal    : vec2<f32>,
};
@group(0) @binding(0) var<uniform> cam : Camera;

// Per-gaussian data, 14 floats: pos(3) scale(3) color(3) opacity(1) quat(4,wxyz)
struct Gaussian {
  px:f32, py:f32, pz:f32,
  sx:f32, sy:f32, sz:f32,
  r:f32, g:f32, b:f32,
  opacity:f32,
  qw:f32, qx:f32, qy:f32, qz:f32,
};
@group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) color : vec3<f32>,
  @location(2) opacity : f32,
};

// unit quad corners
var<private> QUAD : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0),
  vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,1.0), vec2<f32>(-1.0,1.0)
);

fn quatToMat3(w:f32,x:f32,y:f32,z:f32) -> mat3x3<f32> {
  let xx=x*x; let yy=y*y; let zz=z*z;
  let xy=x*y; let xz=x*z; let yz=y*z;
  let wx=w*x; let wy=w*y; let wz=w*z;
  return mat3x3<f32>(
    vec3<f32>(1.0-2.0*(yy+zz), 2.0*(xy+wz),     2.0*(xz-wy)),
    vec3<f32>(2.0*(xy-wz),     1.0-2.0*(xx+zz), 2.0*(yz+wx)),
    vec3<f32>(2.0*(xz+wy),     2.0*(yz-wx),     1.0-2.0*(xx+yy))
  );
}

@vertex
fn vs(@builtin(vertex_index) vi:u32, @builtin(instance_index) ii:u32) -> VSOut {
  let gs = gaussians[ii];
  let center = vec3<f32>(gs.px, gs.py, gs.pz);

  // 3D covariance Sigma = R S S^T R^T
  let R = quatToMat3(gs.qw, gs.qx, gs.qy, gs.qz);
  let S = mat3x3<f32>(
    vec3<f32>(gs.sx,0.0,0.0), vec3<f32>(0.0,gs.sy,0.0), vec3<f32>(0.0,0.0,gs.sz)
  );
  let M = R * S;
  let Sigma = M * transpose(M);

  // view-space center
  let viewC = cam.view * vec4<f32>(center, 1.0);
  let t = viewC.xyz;
  let tz = max(-t.z, 1e-4);

  // Jacobian of perspective projection (EWA splatting)
  let fx = cam.focal.x; let fy = cam.focal.y;
  let J = mat3x3<f32>(
    vec3<f32>(fx/tz, 0.0, -(fx*t.x)/(tz*tz)),
    vec3<f32>(0.0, fy/tz, -(fy*t.y)/(tz*tz)),
    vec3<f32>(0.0, 0.0, 0.0)
  );
  let W = mat3x3<f32>(
    cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz
  );
  let T = J * W;
  let cov2 = T * Sigma * transpose(T);

  // 2x2 screen covariance (+ small blur for AA)
  var a = cov2[0][0] + 0.3;
  var b = cov2[0][1];
  var c = cov2[1][1] + 0.3;

  // eigen-decomposition for quad extents (in pixels)
  let mid = 0.5*(a+c);
  let disc = sqrt(max(mid*mid - (a*c - b*b), 0.0));
  let l1 = mid + disc;
  let l2 = max(mid - disc, 0.0);
  let majorAxis = 3.0 * sqrt(l1);
  let minorAxis = 3.0 * sqrt(l2);
  // eigenvector of larger eigenvalue
  var ev = normalize(vec2<f32>(b, l1 - a));
  if (abs(b) < 1e-6) { ev = vec2<f32>(1.0, 0.0); }
  let evp = vec2<f32>(-ev.y, ev.x);

  let clip = cam.viewProj * vec4<f32>(center, 1.0);
  let corner = QUAD[vi];
  // offset in pixels -> NDC
  let offsetPix = ev * (corner.x * majorAxis) + evp * (corner.y * minorAxis);
  let ndcOffset = vec2<f32>(
    2.0 * offsetPix.x / cam.viewport.x,
    2.0 * offsetPix.y / cam.viewport.y
  ) * clip.w;

  var out : VSOut;
  out.pos = vec4<f32>(clip.xy + ndcOffset, clip.z, clip.w);
  out.uv = corner * 3.0; // falloff radius ~3 sigma
  out.color = vec3<f32>(gs.r, gs.g, gs.b);
  out.opacity = gs.opacity;
  return out;
}

@fragment
fn fs(in:VSOut) -> @location(0) vec4<f32> {
  let d2 = dot(in.uv, in.uv);
  let alpha = in.opacity * exp(-0.5 * d2);
  if (alpha < 0.004) { discard; }
  return vec4<f32>(in.color * alpha, alpha);
}
`;

export async function createRenderer(canvas, assetBase, assetName, statusCb) {
  if (!navigator.gpu) throw new Error("WebGPU not available in this browser.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter.");
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "premultiplied" });

  statusCb && statusCb("loading avatar…");
  const avatar = await loadGaussianAvatar(assetBase, assetName);
  statusCb && statusCb(`uploading ${avatar.N.toLocaleString()} gaussians…`);

  // storage buffer with packed gaussians
  const gBuf = device.createBuffer({
    size: avatar.packed.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gBuf, 0, avatar.packed);

  // camera uniform
  const camBufSize = (16 + 16 + 4 + 2 + 2) * 4;
  const camBuf = device.createBuffer({
    size: Math.ceil(camBufSize / 16) * 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const module = device.createShaderModule({ code: WGSL });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: {
      module, entryPoint: "fs",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camBuf } },
      { binding: 1, resource: { buffer: gBuf } },
    ],
  });

  return {
    device, ctx, format, pipeline, bind, camBuf, avatar,
    render(camState) {
      const { viewProj, view, camPos, viewport, focal } = camState;
      const data = new Float32Array(Math.ceil(camBufSize / 16) * 4);
      data.set(viewProj, 0);
      data.set(view, 16);
      data.set([camPos[0], camPos[1], camPos[2], 1], 32);
      data.set(viewport, 36);
      data.set(focal, 38);
      device.queue.writeBuffer(camBuf, 0, data);

      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: ctx.getCurrentTexture().createView(),
          clearValue: { r: 0.06, g: 0.07, b: 0.09, a: 1 },
          loadOp: "clear", storeOp: "store",
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.draw(6, avatar.N, 0, 0);
      pass.end();
      device.queue.submit([enc.finish()]);
    },
  };
}
