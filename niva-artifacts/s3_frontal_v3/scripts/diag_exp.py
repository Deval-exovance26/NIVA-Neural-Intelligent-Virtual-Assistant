import sys, pickle
from pathlib import Path
import numpy as np, torch, imageio.v2 as imageio
GAGA=Path('/home/ubuntu/GAGAvatar'); NIVA_SRC=Path('/home/ubuntu/NIVA-Neural-Intelligent-Virtual-Assistant/offline-pipeline/src')
sys.path.insert(0,str(GAGA)); sys.path.insert(0,str(NIVA_SRC))
from gagavatar.runtime import GAGAvatarRuntime,GAGAvatarRuntimeConfig
from gagavatar.assets import GAGAvatarAssets
from gsplat import rasterization
from niva_offline.binding.triangle_binding import compute_triangle_frames,bind_gaussians
dev='cuda'; root=GAGA/'assets'
rt=GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=GAGAvatarAssets(root=root,model_path=root/'GAGAvatar.pt',tracked_path=None,flame_model_path=root/'FLAME_with_eye.pt'),device=dev))
av=pickle.load(open('/home/ubuntu/GAGAvatar_track/outputs/images/optim.pkl','rb'))['front.png']
rt.set_tracked_avatar({k:(torch.tensor(v).float() if isinstance(v,np.ndarray) else v) for k,v in av.items()},'front.png')
cp=rt.model.cam_params; H,W=int(cp['size'][0]),int(cp['size'][1]); fx=float(cp['focal_x'])*W/2
K=torch.tensor([[[fx,0,W/2],[0,fx,H/2],[0,0,1.0]]],device=dev)
faces=torch.load(root/'FLAME_with_eye.pt',map_location='cpu',weights_only=False)['flame_model']['f'].long().to(dev)
tt=pickle.load(open('/home/ubuntu/GAGAvatar_track/outputs/Nidhi_clean/optim.pkl','rb'))
def yaw(e):
    R=np.asarray(e['transform_matrix']).reshape(3,4)[:3,:3]; return np.degrees(np.arctan2(R[0,2],R[2,2]))
ks=sorted(tt.keys(),key=lambda k:int(k.split('_')[-1])); c=np.median([yaw(tt[k]) for k in ks])
k0=sorted(ks,key=lambda k:abs(yaw(tt[k])-c))[0]
E=lambda e: torch.tensor(np.asarray(e['expcode']).reshape(-1)[:100]).float().to(dev)
expT=E(tt[k0]); expP=E(av)
print('k0',k0,'|expT-expP|',(expT-expP).abs().mean().item(),'max',(expT-expP).abs().max().item())
ck=torch.load('/home/ubuntu/niva-artifacts/standalone/s3_frontal_anim/ckpt_distilled.pt',map_location=dev,weights_only=False)
lo,lq,ls,pi=ck['local_offset'],ck['local_quat'],torch.exp(ck['log_scale']),ck['parent_idx']
op=torch.sigmoid(ck['opacity_logit']); sh=ck['sh']
def batch(exp,jaw=0.0):
    m=torch.zeros(106,device=dev); m[:100]=exp; m[103]=jaw; return rt.build_forward_batch(m[None])
def vm(cm):
    V=torch.eye(4,device=dev); V[:3,:3]=cm[:3,:3]; V[:3,3]=cm[:3,3]; V[:2,:]*=-1; return V[None]
def bind(v):
    b,R,k=compute_triangle_frames(v,faces); return bind_gaussians(lo,lq,ls,pi,b,R,k)
def rs(pos,quat,sc,V):
    o,_,_=rasterization(pos,quat,sc,op,sh,V,K,width=W,height=H,sh_degree=3); return o[0].clamp(0,1)
out={}
with torch.no_grad():
    bT=batch(expT); bP=batch(expP)
    print('cam diff',(bT['t_transform']-bP['t_transform']).abs().max().item())
    V=vm(bT['t_transform'][0])
    out['teacherT']=rt.render_rgb_batch(bT)[0].permute(1,2,0).clamp(0,1)
    out['teacherP']=rt.render_rgb_batch(bP)[0].permute(1,2,0).clamp(0,1)
    vT=bT['t_points'][0]; vP=bP['t_points'][0]
    print('mesh |vT-vP| mean',(vT-vP).norm(dim=1).mean().item(),'max',(vT-vP).norm(dim=1).max().item())
    out['studT']=rs(*bind(vT),V); out['studP']=rs(*bind(vP),V)
    # jaw: exact vs linear-pos-delta
    vJ=batch(expT,0.35)['t_points'][0]; vJh=batch(expT,0.175)['t_points'][0]
    pN,qN,sN=bind(vT); pJ,qJ,sJ=bind(vJ); pH,qH,sH=bind(vJh)
    out['jaw_exact']=rs(pH,qH,sH,V)
    out['jaw_lin']=rs(pN+0.5*(pJ-pN),qN,sN,V)
    out['teacher_jaw']=rt.render_rgb_batch(batch(expT,0.175))[0].permute(1,2,0).clamp(0,1)
for k,v in out.items(): print(k,'mean',round(v.mean().item(),4))
print('L1 studT-teacherT',(out['studT']-out['teacherT']).abs().mean().item())
print('L1 studP-teacherP',(out['studP']-out['teacherP']).abs().mean().item())
print('L1 jaw_exact-teacher',(out['jaw_exact']-out['teacher_jaw']).abs().mean().item(),'jaw_lin-teacher',(out['jaw_lin']-out['teacher_jaw']).abs().mean().item())
g=lambda *n: torch.cat([out[x] for x in n],1).cpu().numpy()
imageio.imwrite('/tmp/d_exp.png',(g('teacherT','studT','studP','teacherP')*255).astype(np.uint8))
imageio.imwrite('/tmp/d_jaw.png',(g('teacher_jaw','jaw_exact','jaw_lin')*255).astype(np.uint8))
