import sys, pickle
from pathlib import Path
import numpy as np, torch, imageio.v2 as imageio
import torch.nn.functional as F
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
cp=rt.model.cam_params; H,W=int(cp['size'][0]),int(cp['size'][1]); f=float(cp['focal_x'])*W/2
K=torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1.0]]],device=dev)
fl=torch.load(root/'FLAME_with_eye.pt',map_location='cpu',weights_only=False)
faces=fl['flame_model']['f'].long().to(dev)
print('faces',faces.shape,'max vid',faces.max().item())
tt=pickle.load(open('/home/ubuntu/GAGAvatar_track/outputs/Nidhi_clean/optim.pkl','rb'))
exp=torch.tensor(np.asarray(tt['Nidhi_clean_83']['expcode']).reshape(-1)[:100]).float().to(dev)
m=torch.zeros(106,device=dev); m[:100]=exp; b=rt.build_forward_batch(m[None]); v=b['t_points'][0]; cam=b['t_transform'][0]
print('verts',v.shape)
V=torch.eye(4,device=dev); V[:3,:3]=cam[:3,:3]; V[:3,3]=cam[:3,3]; V[:2,:]*=-1
with torch.no_grad(): teacher=rt.render_rgb_batch(b)[0].permute(1,2,0).clamp(0,1)
# eyeball faces: FLAME head has 5023 verts; extra verts belong to eyeballs
eyef=(faces>=5023).any(1)
print('eyeball faces',eyef.sum().item())
def rend(ck,hide=None):
    lo,lq,ls,pi=ck['local_offset'],ck['local_quat'],torch.exp(ck['log_scale']),ck['parent_idx']
    bb,R,k=compute_triangle_frames(v,faces); pos,q,s=bind_gaussians(lo,lq,ls,pi,bb,R,k)
    op=torch.sigmoid(ck['opacity_logit']).clone()
    if hide is not None: op[hide[pi]]=0
    o,_,_=rasterization(pos,q,s,op,ck['sh'],V[None],K,width=W,height=H,sh_degree=ck['sh_degree']); return o[0].clamp(0,1), pi
L=lambda p: torch.load(p,map_location=dev,weights_only=False)
A=L('/home/ubuntu/niva-artifacts/standalone/s3_frontal_anim/ckpt_distilled.pt')
Fx=L('/home/ubuntu/niva-artifacts/standalone/s3_frontal_fixed/ckpt_distilled.pt')
with torch.no_grad():
    a,pi=rend(A); fx,_=rend(Fx); ah,_=rend(A,eyef); fh,_=rend(Fx,eyef)
    print('eyeball gaussians',eyef[pi].sum().item())
# eye crop: find eye region rows ~ y 220-290, x 150-370 at 512
def crop(im): return F.interpolate(im[225:290,150:362].permute(2,0,1)[None],scale_factor=3,mode='nearest')[0].permute(1,2,0)
for n,im in [('fixed',fx),('anim',a)]:
    print(n,'eye L1 vs teacher',(crop(im)-crop(teacher)).abs().mean().item())
rows=[crop(x) for x in [teacher,fx,a,ah,fh]]
imageio.imwrite('/tmp/d_eye.png',(torch.cat(rows,0).cpu().numpy()*255).astype(np.uint8))
