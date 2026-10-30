"""Bake v2: pose at a TRAINED expression (turntable frontal frame used in distillation),
bake SH3 at the training camera, and export jaw/blink morphs as pos+quat+scale deltas (binary)."""
import sys, json, pickle, math
from pathlib import Path
import numpy as np, torch, imageio.v2 as imageio
GAGA=Path('/home/ubuntu/GAGAvatar'); NIVA_SRC=Path('/home/ubuntu/NIVA-Neural-Intelligent-Virtual-Assistant/offline-pipeline/src')
sys.path.insert(0,str(GAGA)); sys.path.insert(0,str(NIVA_SRC))
from gagavatar.runtime import GAGAvatarRuntime,GAGAvatarRuntimeConfig
from gagavatar.assets import GAGAvatarAssets
from gsplat import rasterization
try: from gsplat import spherical_harmonics
except ImportError: from gsplat.cuda._wrapper import spherical_harmonics
from niva_offline.binding.triangle_binding import compute_triangle_frames,bind_gaussians
dev='cuda'; root=GAGA/'assets'; DST=Path(sys.argv[1] if len(sys.argv)>1 else '/home/ubuntu/fweb/assets_frontal')
JAW_REF=0.35; EYE_REF=0.6
rt=GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=GAGAvatarAssets(root=root,model_path=root/'GAGAvatar.pt',tracked_path=None,flame_model_path=root/'FLAME_with_eye.pt'),device=dev))
av=pickle.load(open('/home/ubuntu/GAGAvatar_track/outputs/images/optim.pkl','rb'))['front.png']
rt.set_tracked_avatar({k:(torch.tensor(v).float() if isinstance(v,np.ndarray) else v) for k,v in av.items()},'front.png')
cp=rt.model.cam_params; H,W=int(cp['size'][0]),int(cp['size'][1]); f=float(cp['focal_x'])*W/2
K=torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1.0]]],device=dev)
faces=torch.load(root/'FLAME_with_eye.pt',map_location='cpu',weights_only=False)['flame_model']['f'].long().to(dev)
tt=pickle.load(open('/home/ubuntu/GAGAvatar_track/outputs/Nidhi_clean/optim.pkl','rb'))
def yaw(e):
    R=np.asarray(e['transform_matrix']).reshape(3,4)[:3,:3]; return np.degrees(np.arctan2(R[0,2],R[2,2]))
ks=sorted(tt.keys(),key=lambda k:int(k.split('_')[-1])); c=np.median([yaw(tt[k]) for k in ks])
k0=sorted(ks,key=lambda k:abs(yaw(tt[k])-c))[0]
exp=torch.tensor(np.asarray(tt[k0]['expcode']).reshape(-1)[:100]).float().to(dev)
print('[v2] base expression from trained frame',k0)
ck=torch.load('/home/ubuntu/niva-artifacts/standalone/s3_frontal_anim/ckpt_distilled.pt',map_location=dev,weights_only=False)
lo,lq,ls,pi=ck['local_offset'],ck['local_quat'],torch.exp(ck['log_scale']),ck['parent_idx']
op=torch.sigmoid(ck['opacity_logit']); sh=ck['sh']
def batch(jaw):
    m=torch.zeros(106,device=dev); m[:100]=exp; m[103]=jaw; return rt.build_forward_batch(m[None])
def eyemesh(e):
    ep=torch.zeros(1,6,device=dev); ep[0,0]=e; ep[0,3]=e
    return rt.flame_model(shape_params=rt._shape_code,pose_params=torch.zeros(1,6,device=dev),expression_params=exp[None],eye_pose_params=ep).float()[0]
def bind(v):
    b,R,k=compute_triangle_frames(v,faces); return bind_gaussians(lo,lq,ls,pi,b,R,k)
def align(q,q0):
    s=torch.sign((q*q0).sum(1,keepdim=True)); s[s==0]=1; return q*s
with torch.no_grad():
    b0=batch(0.0); cam=b0['t_transform'][0]; vN=b0['t_points'][0]
    pN,qN,sN=bind(vN); pJ,qJ,sJ=bind(batch(JAW_REF)['t_points'][0])
    pE0,qE0,sE0=bind(eyemesh(0.0)); pB,qB,sB=bind(eyemesh(EYE_REF))
    qJ=align(qJ,qN); qB=align(qB,qE0)
    print('[v2] |eyemesh0 - vN| max',(eyemesh(0.0)-vN).norm(dim=1).max().item())
    V=torch.eye(4,device=dev); V[:3,:3]=cam[:3,:3]; V[:3,3]=cam[:3,3]; V[:2,:]*=-1
    ctr=-V[:3,:3].T@V[:3,3]; fwd=V[:3,:3].T@torch.tensor([0.,0.,1.],device=dev)
    d=pN-ctr[None]; d=d/d.norm(dim=1,keepdim=True)
    rgb=(spherical_harmonics(3,d,sh)+0.5).clamp_min(0)
    full,_,_=rasterization(pN,qN,sN,op,sh,V[None],K,width=W,height=H,sh_degree=3)
    bk,_,_=rasterization(pN,qN,sN,op,rgb,V[None],K,width=W,height=H,sh_degree=None)
    print('[v2] mean|full-baked|',(full[0].clamp(0,1)-bk[0].clamp(0,1)).abs().mean().item(),'img mean',bk[0].clamp(0,1).mean().item())
    # validate linear blend of pos+quat+scale vs exact at half jaw
    w=0.5; pH,qH,sH=bind(batch(JAW_REF*w)['t_points'][0])
    ql=qN+w*(qJ-qN); ql=ql/ql.norm(dim=1,keepdim=True)
    ex,_,_=rasterization(pH,qH,sH,op,rgb,V[None],K,width=W,height=H,sh_degree=None)
    li,_,_=rasterization(pN+w*(pJ-pN),ql,sN+w*(sJ-sN),op,rgb,V[None],K,width=W,height=H,sh_degree=None)
    po,_,_=rasterization(pN+w*(pJ-pN),qN,sN,op,rgb,V[None],K,width=W,height=H,sh_degree=None)
    print('[v2] half-jaw |lin_full-exact|',(li-ex).abs().mean().item(),' |pos_only-exact|',(po-ex).abs().mean().item())
    imageio.imwrite('/tmp/v2_neutral.png',(bk[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8))
    imageio.imwrite('/tmp/v2_jaw.png',(torch.cat([ex[0],li[0],po[0]],1).clamp(0,1).cpu().numpy()*255).astype(np.uint8))
N=pN.shape[0]; n=lambda t:t.detach().cpu().numpy().astype(np.float32)
out=np.zeros((N,14),np.float32)
out[:,0:3]=n(pN); out[:,3:7]=n(qN); out[:,7:10]=n(sN); out[:,10]=n(op); out[:,11:14]=n(rgb)
out.tofile(DST/'niva_frontal.bin')
# anim bin: [jaw: dpos3 dquat4 dscale3][blink: dpos3 dquat4 dscale3] per gaussian = 20 f32
an=np.concatenate([n(pJ-pN),n(qJ-qN),n(sJ-sN),n(pB-pE0),n(qB-qE0),n(sB-sE0)],1).astype(np.float32)
an.tofile(DST/'niva_frontal_anim.bin')
json.dump({'format':'niva-gaussian-avatar','num_gaussians':int(N),'stride_floats':14,
  'field_order':['px','py','pz','qw','qx','qy','qz','sx','sy','sz','opacity','r','g','b'],
  'flip_y':False,'frontal':True,'color':'SH3 baked at training camera; posed at trained expression '+k0,
  'camera':{'eye':n(ctr).tolist(),'forward':n(fwd).tolist(),'fovy_deg':2*math.degrees(math.atan(H/2/f))},
  'anim':{'file':'niva_frontal_anim.bin','stride_floats':20,'layout':['jaw dpos3 dquat4(wxyz) dscale3','blink dpos3 dquat4 dscale3'],'jaw_ref':JAW_REF,'eye_ref':EYE_REF}},
  open(DST/'niva_frontal.json','w'))
print('[v2] wrote',DST,'anim bytes',an.nbytes)
