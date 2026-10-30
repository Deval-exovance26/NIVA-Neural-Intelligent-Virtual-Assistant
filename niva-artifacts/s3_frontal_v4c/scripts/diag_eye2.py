import sys, pickle
from pathlib import Path
import numpy as np, torch, imageio.v2 as imageio
import torch.nn.functional as F
GAGA=Path("/home/ubuntu/GAGAvatar"); NIVA_SRC=Path("/home/ubuntu/NIVA-Neural-Intelligent-Virtual-Assistant/offline-pipeline/src")
sys.path.insert(0,str(GAGA)); sys.path.insert(0,str(NIVA_SRC))
from gagavatar.runtime import GAGAvatarRuntime,GAGAvatarRuntimeConfig
from gagavatar.assets import GAGAvatarAssets
from gsplat import rasterization
from niva_offline.binding.triangle_binding import compute_triangle_frames,bind_gaussians
dev="cuda"; root=GAGA/"assets"
rt=GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=GAGAvatarAssets(root=root,model_path=root/"GAGAvatar.pt",tracked_path=None,flame_model_path=root/"FLAME_with_eye.pt"),device=dev))
av=pickle.load(open("/home/ubuntu/GAGAvatar_track/outputs/images/optim.pkl","rb"))["front.png"]
rt.set_tracked_avatar({k:(torch.tensor(v).float() if isinstance(v,np.ndarray) else v) for k,v in av.items()},"front.png")
cp=rt.model.cam_params; H,W=int(cp["size"][0]),int(cp["size"][1]); f=float(cp["focal_x"])*W/2
K=torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1.0]]],device=dev)
faces=torch.load(root/"FLAME_with_eye.pt",map_location="cpu",weights_only=False)["flame_model"]["f"].long().to(dev)
tt=pickle.load(open("/home/ubuntu/GAGAvatar_track/outputs/Nidhi_clean/optim.pkl","rb"))
exp=torch.tensor(np.asarray(tt["Nidhi_clean_83"]["expcode"]).reshape(-1)[:100]).float().to(dev)
EREF=0.6
ck=torch.load("/home/ubuntu/niva-artifacts/standalone/s3_frontal_v3/ckpt_distilled.pt",map_location=dev,weights_only=False)
lo,lq,ls,pi=ck["local_offset"],ck["local_quat"],torch.exp(ck["log_scale"]),ck["parent_idx"]
sh=ck["sh"]; ol=ck["opacity_logit"]; dB=ck["dsh_blink"]; oB=ck["dop_blink"]
m=torch.zeros(106,device=dev); m[:100]=exp; b0=rt.build_forward_batch(m[None]); cam=b0["t_transform"][0]
V=torch.eye(4,device=dev); V[:3,:3]=cam[:3,:3]; V[:3,3]=cam[:3,3]; V[:2,:]*=-1
def eyemesh(e):
    ep=torch.zeros(1,6,device=dev); ep[0,0]=e; ep[0,3]=e
    return rt.flame_model(shape_params=rt._shape_code,pose_params=torch.zeros(1,6,device=dev),expression_params=exp[None],eye_pose_params=ep).float()[0]
def bind(v):
    bb,R,k=compute_triangle_frames(v,faces); return bind_gaussians(lo,lq,ls,pi,bb,R,k)
def student(wb):
    v=eyemesh(EREF*wb); pos,q,s=bind(v)
    o,_,_=rasterization(pos,q,s,torch.sigmoid(ol+wb*oB),sh+wb*dB,V[None],K,width=W,height=H,sh_degree=3); return o[0].clamp(0,1)
def teacher(wb):
    bb={k:(vv.clone() if torch.is_tensor(vv) else vv) for k,vv in b0.items()}
    ep=torch.zeros(1,6,device=dev); ep[0,0]=EREF*wb; ep[0,3]=EREF*wb
    bb["t_points"]=rt.flame_model(shape_params=rt._shape_code,pose_params=torch.zeros(1,6,device=dev),expression_params=exp[None],eye_pose_params=ep).float()
    return rt.render_rgb_batch(bb)[0].permute(1,2,0).clamp(0,1)
def crop(im): return F.interpolate(im[225:290,150:362].permute(2,0,1)[None],scale_factor=3,mode="nearest")[0].permute(1,2,0)
rows=[]
with torch.no_grad():
    for wb in [0.0,0.5,1.0]:
        s=student(wb); t=teacher(wb); print("blink",wb,"eye L1",round((crop(s)-crop(t)).abs().mean().item(),4),flush=True)
        rows.append(torch.cat([crop(t),crop(s)],1))
imageio.imwrite("/tmp/d_eye2.png",(torch.cat(rows,0).cpu().numpy()*255).astype(np.uint8))
print("wrote /tmp/d_eye2.png rows blink 0/0.5/1.0 left=teacher right=student",flush=True)
