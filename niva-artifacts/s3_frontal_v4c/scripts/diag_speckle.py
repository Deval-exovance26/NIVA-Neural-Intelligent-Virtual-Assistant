import sys, pickle
from pathlib import Path
import numpy as np, torch
GAGA=Path("/home/ubuntu/GAGAvatar"); NIVA_SRC=Path("/home/ubuntu/NIVA-Neural-Intelligent-Virtual-Assistant/offline-pipeline/src")
sys.path.insert(0,str(GAGA)); sys.path.insert(0,str(NIVA_SRC))
from gagavatar.runtime import GAGAvatarRuntime,GAGAvatarRuntimeConfig
from gagavatar.assets import GAGAvatarAssets
try: from gsplat import spherical_harmonics
except ImportError: from gsplat.cuda._wrapper import spherical_harmonics
from niva_offline.binding.triangle_binding import compute_triangle_frames,bind_gaussians
dev="cuda"; root=GAGA/"assets"
rt=GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=GAGAvatarAssets(root=root,model_path=root/"GAGAvatar.pt",tracked_path=None,flame_model_path=root/"FLAME_with_eye.pt"),device=dev))
av=pickle.load(open("/home/ubuntu/GAGAvatar_track/outputs/images/optim.pkl","rb"))["front.png"]
rt.set_tracked_avatar({k:(torch.tensor(v).float() if isinstance(v,np.ndarray) else v) for k,v in av.items()},"front.png")
cp=rt.model.cam_params; H,W=int(cp["size"][0]),int(cp["size"][1]); f=float(cp["focal_x"])*W/2
faces=torch.load(root/"FLAME_with_eye.pt",map_location="cpu",weights_only=False)["flame_model"]["f"].long().to(dev)
tt=pickle.load(open("/home/ubuntu/GAGAvatar_track/outputs/Nidhi_clean/optim.pkl","rb"))
exp=torch.tensor(np.asarray(tt["Nidhi_clean_83"]["expcode"]).reshape(-1)[:100]).float().to(dev)
ck=torch.load("/home/ubuntu/niva-artifacts/standalone/s3_frontal_v3/ckpt_distilled.pt",map_location=dev,weights_only=False)
lo,lq,ls,pi=ck["local_offset"],ck["local_quat"],torch.exp(ck["log_scale"]),ck["parent_idx"]
sh=ck["sh"]; ol=ck["opacity_logit"]; dB=ck["dsh_blink"]; oB=ck["dop_blink"]
m=torch.zeros(106,device=dev); m[:100]=exp; _b0=rt.build_forward_batch(m[None]); cam=_b0["t_transform"][0]; ep=torch.zeros(1,6,device=dev); ep[0,0]=0.6; ep[0,3]=0.6
v=rt.flame_model(shape_params=rt._shape_code,pose_params=torch.zeros(1,6,device=dev),expression_params=exp[None],eye_pose_params=ep).float()[0]
V=torch.eye(4,device=dev); V[:3,:3]=cam[:3,:3]; V[:3,3]=cam[:3,3]; V[:2,:]*=-1
bb,R,k=compute_triangle_frames(v,faces); pos,q,s=bind_gaussians(lo,lq,ls,pi,bb,R,k)
# project to pixels
ph=torch.cat([pos,torch.ones(pos.shape[0],1,device=dev)],1)@V.T
x=f*ph[:,0]/ph[:,2]+W/2; y=f*ph[:,1]/ph[:,2]+H/2
inbox=(x>150)&(x<362)&(y>225)&(y<290)
# blink colour + opacity
ctr=-V[:3,:3].T@V[:3,3]; d=pos-ctr[None]; d=d/d.norm(dim=1,keepdim=True)
rgb=(spherical_harmonics(3,d,sh+dB)+0.5).clamp(0,1)
opac=torch.sigmoid(ol+oB)
sat=(rgb.max(1).values-rgb.min(1).values)  # colour saturation
speckle=inbox & (sat>0.22) & (opac>0.3)
print("eyelid-box gaussians",int(inbox.sum()),"| speckle (sat>0.22,opac>0.3)",int(speckle.sum()),flush=True)
print("sat in box: mean",round(float(sat[inbox].mean()),3),"p95",round(float(sat[inbox].quantile(0.95)),3),"max",round(float(sat[inbox].max()),3),flush=True)
# which flame verts/region? report parent triangle vertex ids range
pidx=pi[speckle]
vids=faces[pidx].flatten().unique()
print("speckle count",int(speckle.sum()),"spanning",int(vids.numel()),"flame verts; vid range",int(vids.min()),int(vids.max()),flush=True)
torch.save({"speckle_mask":speckle.cpu(),"inbox_mask":inbox.cpu()},"/tmp/speckle.pt")
print("saved /tmp/speckle.pt",flush=True)
