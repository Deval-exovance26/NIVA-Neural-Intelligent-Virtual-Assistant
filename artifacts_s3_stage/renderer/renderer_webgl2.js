// renderer_webgl2.js — universal WebGL2 3D Gaussian Splatting renderer.
//
// This is the cross-device path (phones, tablets, older desktops): WebGL2 is
// far more widely supported than WebGPU. Same splatting math as the WebGPU
// renderer (EWA projection of the 3D covariance), implemented with instanced
// quads. Produces the same image so the two paths are interchangeable.
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

out vec2 vUV;
out vec3 vColor;
out float vOpacity;

mat3 quatToMat3(vec4 q){
  float w=q.x,x=q.y,y=q.z,z=q.w;
  float xx=x*x,yy=y*y,zz=z*z,xy=x*y,xz=x*z,yz=y*z,wx=w*x,wy=w*y,wz=w*z;
  return mat3(
    1.0-2.0*(yy+zz), 2.0*(xy+wz),     2.0*(xz-wy),
    2.0*(xy-wz),     1.0-2.0*(xx+zz), 2.0*(yz+wx),
    2.0*(xz+wy),     2.0*(yz-wx),     1.0-2.0*(xx+yy)
  );
}

void main(){
  mat3 R = quatToMat3(aQuat);
  mat3 S = mat3(aScale.x,0.0,0.0, 0.0,aScale.y,0.0, 0.0,0.0,aScale.z);
  mat3 M = R*S;
  mat3 Sigma = M*transpose(M);

  vec4 viewC = uView * vec4(aPos,1.0);
  float tz = max(-viewC.z, 1e-4);
  float fx=uFocal.x, fy=uFocal.y;
  mat3 J = mat3(
    fx/tz, 0.0, 0.0,
    0.0, fy/tz, 0.0,
    -(fx*viewC.x)/(tz*tz), -(fy*viewC.y)/(tz*tz), 0.0
  );
  mat3 W = mat3(uView);
  mat3 T = J*W;
  mat3 cov = T*Sigma*transpose(T);

  float a=cov[0][0]+0.3, b=cov[0][1], c=cov[1][1]+0.3;
  float mid=0.5*(a+c);
  float disc=sqrt(max(mid*mid-(a*c-b*b),0.0));
  float l1=mid+disc, l2=max(mid-disc,0.0);
  float major=3.0*sqrt(l1), minor=3.0*sqrt(l2);
  vec2 ev = abs(b)<1e-6 ? vec2(1.0,0.0) : normalize(vec2(b,l1-a));
  vec2 evp = vec2(-ev.y, ev.x);

  vec4 clip = uViewProj * vec4(aPos,1.0);
  vec2 offPix = ev*(aCorner.x*major) + evp*(aCorner.y*minor);
  vec2 ndcOff = vec2(2.0*offPix.x/uViewport.x, 2.0*offPix.y/uViewport.y)*clip.w;

  gl_Position = vec4(clip.xy + ndcOff, clip.z, clip.w);
  vUV = aCorner*3.0;
  vColor = aColor;
  vOpacity = aOpacity;
}`;

const FS = `#version 300 es
precision highp float;
in vec2 vUV;
in vec3 vColor;
in float vOpacity;
out vec4 fragColor;
void main(){
  float d2 = dot(vUV,vUV);
  float alpha = vOpacity*exp(-0.5*d2);
  if(alpha<0.004) discard;
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

export async function createWebGL2Renderer(canvas, assetBase, assetName, statusCb) {
  const gl = canvas.getContext("webgl2", { premultipliedAlpha: true, antialias: false });
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

  // per-instance interleaved buffer (stride 14 floats: pos3 scale3 color3 op1 quat4)
  const instBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
  gl.bufferData(gl.ARRAY_BUFFER, avatar.packed, gl.STATIC_DRAW);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  // corner
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  // instance attrs
  const FB = 4, stride = 14 * FB;
  gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
  const setup = (loc, size, offsetFloats) => {
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offsetFloats * FB);
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

  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.disable(gl.DEPTH_TEST);

  return {
    backend: "webgl2",
    avatar,
    resize() { gl.viewport(0, 0, canvas.width, canvas.height); },
    render(cam) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.06, 0.07, 0.09, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(prog);
      gl.uniformMatrix4fv(uView, false, cam.view);
      gl.uniformMatrix4fv(uViewProj, false, cam.viewProj);
      gl.uniform2fv(uViewport, cam.viewport);
      gl.uniform2fv(uFocal, cam.focal);
      gl.bindVertexArray(vao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, avatar.N);
      gl.bindVertexArray(null);
    },
  };
}
