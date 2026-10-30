// renderer_sh.js — WebGL2 3DGS renderer with VIEW-DEPENDENT SH color (deg-3),
// matching gsplat. Geometry per-instance; SH (16x3) sampled from a float texture.
// Proven conic projection + premultiplied back-to-front + depth sort + setMorph.
import { loadSHAvatar } from "./loader_sh.js";

const VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aCorner;
layout(location=1) in vec3 aPos;
layout(location=2) in vec3 aScale;
layout(location=3) in float aOpacity;
layout(location=4) in vec4 aQuat;      // wxyz
layout(location=5) in float aIndex;    // gaussian index -> SH texture row

uniform mat4 uView, uViewProj;
uniform vec2 uViewport, uFocal;
uniform vec3 uCamPos;
uniform sampler2D uSH;    // RGB32F, packed: texel index = gaussianIndex*16 + coeff
uniform int uSHtexW;      // texture width in texels

out vec3 vColor; out float vOpacity; out vec2 vConicXY; out float vConicZ; out vec2 vDelta;

mat3 q2m(vec4 q){ float w=q.x,x=q.y,y=q.z,z=q.w; float n=inversesqrt(max(w*w+x*x+y*y+z*z,1e-12));
  w*=n;x*=n;y*=n;z*=n; return mat3(1.-2.*(y*y+z*z),2.*(x*y+w*z),2.*(x*z-w*y),
  2.*(x*y-w*z),1.-2.*(x*x+z*z),2.*(y*z+w*x), 2.*(x*z+w*y),2.*(y*z-w*x),1.-2.*(x*x+y*y)); }

vec3 shC(int row, int c){ int lin=row*16+c; int tx=lin%uSHtexW; int ty=lin/uSHtexW; return texelFetch(uSH, ivec2(tx,ty), 0).rgb; }

// SH deg-3 (gsplat/INRIA basis constants)
vec3 evalSH(int row, vec3 dir){
  const float C0=0.28209479177387814;
  const float C1=0.4886025119029199;
  const float C2a=1.0925484305920792, C2b=0.31539156525252005, C2c=1.2615662610100802;
  const float C3a=0.5900435899266435, C3b=2.890611442640554, C3c=0.4570457994644658,
              C3d=0.3731763325901154, C3e=1.445305721320277, C3f=0.5900435899266435;
  float x=dir.x,y=dir.y,z=dir.z;
  vec3 r = C0*shC(0,row);
  r += -C1*y*shC(1,row) + C1*z*shC(2,row) - C1*x*shC(3,row);
  float xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z;
  r += C2a*xy*shC(4,row) + (-C2a*yz)*shC(5,row) + C2b*(2.0*zz-xx-yy)*shC(6,row)
     + (-C2a*xz)*shC(7,row) + C2c*0.5*(xx-yy)*shC(8,row);
  r += C3a*y*(3.0*xx-yy)*shC(9,row) + C3b*xy*z*shC(10,row) + C3c*y*(4.0*zz-xx-yy)*shC(11,row)
     + C3d*z*(2.0*zz-3.0*xx-3.0*yy)*shC(12,row) + C3c*x*(4.0*zz-xx-yy)*shC(13,row)
     + C3e*z*(xx-yy)*shC(14,row) + C3f*x*(xx-3.0*yy)*shC(15,row);
  return r + 0.5;
}

void main(){
  mat3 R=q2m(aQuat); mat3 S=mat3(aScale.x,0.,0., 0.,aScale.y,0., 0.,0.,aScale.z);
  mat3 M=R*S; mat3 Sigma=M*transpose(M);
  vec4 vc=uView*vec4(aPos,1.0);
  if(vc.z>-0.01){ gl_Position=vec4(2.,2.,2.,1.); return; }
  float tz=-vc.z, fx=uFocal.x, fy=uFocal.y;
  mat3 J=mat3(fx/tz,0.,-(fx*vc.x)/(tz*tz), 0.,fy/tz,-(fy*vc.y)/(tz*tz), 0.,0.,0.);
  mat3 T=J*mat3(uView); mat3 cov=T*Sigma*transpose(T);
  float c00=cov[0][0]+0.3, c01=cov[0][1], c11=cov[1][1]+0.3;
  float det=c00*c11-c01*c01; if(det<=0.0){ gl_Position=vec4(2.,2.,2.,1.); return; }
  float id=1.0/det; vConicXY=vec2(c11*id,-c01*id); vConicZ=c00*id;
  float mid=0.5*(c00+c11), disc=sqrt(max(mid*mid-det,0.0));
  float rad=ceil(3.0*sqrt(max(mid+disc, max(mid-disc,0.0))));
  vec4 clip=uViewProj*vec4(aPos,1.0);
  vec2 dpx=aCorner*rad; vDelta=dpx;
  vec2 ndc=vec2(2.0*dpx.x/uViewport.x,2.0*dpx.y/uViewport.y)*clip.w;
  gl_Position=vec4(clip.xy+ndc, clip.z, clip.w);
  vec3 dir=normalize(aPos-uCamPos);
  vColor=clamp(evalSH(int(aIndex+0.5), dir),0.0,1.0);
  vOpacity=aOpacity;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vColor; in float vOpacity; in vec2 vConicXY; in float vConicZ; in vec2 vDelta;
out vec4 fragColor;
void main(){
  float dx=vDelta.x, dy=vDelta.y;
  float power=-0.5*(vConicXY.x*dx*dx+vConicZ*dy*dy)-vConicXY.y*dx*dy;
  if(power>0.0) discard;
  float alpha=min(0.99, vOpacity*exp(power));
  if(alpha<0.004) discard;
  fragColor=vec4(vColor*alpha, alpha);
}`;

function sh(gl,t,src){const s=gl.createShader(t);gl.shaderSource(s,src);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error("shader: "+gl.getShaderInfoLog(s));return s;}

export async function createSHRenderer(canvas, base, name, statusCb){
  const gl=canvas.getContext("webgl2",{premultipliedAlpha:true,antialias:false,preserveDrawingBuffer:true});
  if(!gl) throw new Error("WebGL2 unavailable");
  if(!gl.getExtension("EXT_color_buffer_float")) console.warn("no float color buffer (SH tex still ok)");
  statusCb&&statusCb("loading SH avatar…");
  const A=await loadSHAvatar(base,name);
  const N=A.N, stride=A.stride;

  const prog=gl.createProgram();
  gl.attachShader(prog,sh(gl,gl.VERTEX_SHADER,VS)); gl.attachShader(prog,sh(gl,gl.FRAGMENT_SHADER,FS));
  gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog,gl.LINK_STATUS)) throw new Error("link: "+gl.getProgramInfoLog(prog));

  // SH texture: width=16 coeffs, height=N, RGB32F
  const texW=A.shCoeffs, texH=N;
  const shTex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,shTex);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB32F,texW,texH,0,gl.RGB,gl.FLOAT,A.sh);

  // instance geometry buffer: pos3 scale3 opac1 quat4 index1 = 12 floats (repacked)
  const INST=12;
  const inst=new Float32Array(N*INST);
  const rebuild=(geo)=>{ for(let i=0;i<N;i++){ const s=i*stride,d=i*INST;
    inst[d]=geo[s];inst[d+1]=geo[s+1];inst[d+2]=geo[s+2];        // pos
    inst[d+3]=geo[s+7];inst[d+4]=geo[s+8];inst[d+5]=geo[s+9];    // scale
    inst[d+6]=geo[s+10];                                         // opac
    inst[d+7]=geo[s+3];inst[d+8]=geo[s+4];inst[d+9]=geo[s+5];inst[d+10]=geo[s+6]; // quat wxyz
    inst[d+11]=i; } };
  rebuild(A.geo);

  const quad=new Float32Array([-1,-1,1,-1,1,1,-1,-1,1,1,-1,1]);
  const quadBuf=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,quadBuf); gl.bufferData(gl.ARRAY_BUFFER,quad,gl.STATIC_DRAW);
  const instBuf=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,instBuf); gl.bufferData(gl.ARRAY_BUFFER,inst,gl.DYNAMIC_DRAW);
  const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER,quadBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  const FB=4,sb=INST*FB; gl.bindBuffer(gl.ARRAY_BUFFER,instBuf);
  const set=(l,sz,off)=>{gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,sz,gl.FLOAT,false,sb,off*FB);gl.vertexAttribDivisor(l,1);};
  set(1,3,0); set(2,3,3); set(3,1,6); set(4,4,7); set(5,1,11);
  gl.bindVertexArray(null);

  const U=(n)=>gl.getUniformLocation(prog,n);
  const uView=U("uView"),uVP=U("uViewProj"),uVp=U("uViewport"),uF=U("uFocal"),uCp=U("uCamPos"),uSH=U("uSH"),uW=U("uSHtexW");
  gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.ONE,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA); gl.disable(gl.DEPTH_TEST);

  // depth sort
  const idx=new Uint32Array(N); for(let i=0;i<N;i++) idx[i]=i;
  const depth=new Float32Array(N); const sorted=new Float32Array(N*INST); let needsSort=true;
  const curPos=new Float32Array(A.basePositions); // live positions after morph

  function doSort(view){
    const v8=view[8],v9=view[9],v10=view[10],v11=view[11];
    for(let i=0;i<N;i++){ const t=i*3; depth[i]=-(v8*curPos[t]+v9*curPos[t+1]+v10*curPos[t+2]+v11); }
    const a=Array.from(idx); a.sort((p,q)=>depth[q]-depth[p]);
    for(let i=0;i<N;i++){ const si=a[i], d=i*INST, s=si*INST; for(let j=0;j<INST;j++) sorted[d+j]=inst[s+j]; }
    gl.bindBuffer(gl.ARRAY_BUFFER,instBuf); gl.bufferSubData(gl.ARRAY_BUFFER,0,sorted); needsSort=false;
  }

  return {
    backend:"webgl2-sh", avatar:A,
    resize(){ gl.viewport(0,0,canvas.width,canvas.height); },
    setMorph(wJaw,wBlink){
      const jd=A.jawDelta, bd=A.blinkDelta, base=A.basePositions; if(!jd) return;
      for(let i=0;i<N;i++){ const t=i*3, d=i*INST;
        const x=base[t]+wJaw*jd[t]+wBlink*bd[t], y=base[t+1]+wJaw*jd[t+1]+wBlink*bd[t+1], z=base[t+2]+wJaw*jd[t+2]+wBlink*bd[t+2];
        inst[d]=x;inst[d+1]=y;inst[d+2]=z; curPos[t]=x;curPos[t+1]=y;curPos[t+2]=z; }
      needsSort=true;
    },
    render(cam){
      if(needsSort) doSort(cam.view);
      gl.viewport(0,0,canvas.width,canvas.height);
      gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(prog);
      gl.uniformMatrix4fv(uView,false,cam.view); gl.uniformMatrix4fv(uVP,false,cam.viewProj);
      gl.uniform2fv(uVp,cam.viewport); gl.uniform2fv(uF,cam.focal); gl.uniform3fv(uCp,cam.camPos||[0,0,0]);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,shTex); gl.uniform1i(uSH,0); gl.uniform1i(uW,texW);
      gl.bindVertexArray(vao); gl.drawArraysInstanced(gl.TRIANGLES,0,6,N); gl.bindVertexArray(null);
    },
  };
}
