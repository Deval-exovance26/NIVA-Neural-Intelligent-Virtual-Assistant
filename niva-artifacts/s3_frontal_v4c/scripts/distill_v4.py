"""distill_v4: warm-start from v3; kill eyelid blink GLITTER. Adds more closed-eye frames
and a saturation+TV penalty on the blink-state render in the eyelid band so the closed lid
reads as smooth skin (teacher) instead of rainbow speckle. Jaw path and neutral untouched.
Rollback guard: writes to s3_frontal_v4, never overwrites v3."""
import sys, pickle, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
GAGA=Path("/home/ubuntu/GAGAvatar"); NIVA_SRC=Path("/home/ubuntu/NIVA-Neural-Intelligent-Virtual-Assistant/offline-pipeline/src")
WARM=Path("/home/ubuntu/niva-artifacts/standalone/s3_frontal_v3/ckpt_distilled.pt")
OUT=Path("/home/ubuntu/niva-artifacts/standalone/s3_frontal_v4"); (OUT/"triptych").mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(GAGA)); sys.path.insert(0,str(NIVA_SRC))
from gagavatar.runtime import GAGAvatarRuntime,GAGAvatarRuntimeConfig
from gagavatar.assets import GAGAvatarAssets
from gsplat import rasterization
from niva_offline.binding.triangle_binding import compute_triangle_frames,bind_gaussians
dev="cuda"; ITERS=2500; NFR=10; JREF=0.35; EREF=0.6; root=GAGA/"assets"
rt=GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=GAGAvatarAssets(root=root,model_path=root/"GAGAvatar.pt",tracked_path=None,flame_model_path=root/"FLAME_with_eye.pt"),device=dev))
av=pickle.load(open("/home/ubuntu/GAGAvatar_track/outputs/images/optim.pkl","rb"))["front.png"]
rt.set_tracked_avatar({k:(torch.tensor(v).float() if isinstance(v,np.ndarray) else v) for k,v in av.items()},"front.png")
cp=rt.model.cam_params; H,W=int(cp["size"][0]),int(cp["size"][1]); f=float(cp["focal_x"])*W/2
K=torch.tensor([[[f,0,W/2],[0,f,H/2],[0,0,1.0]]],device=dev)
faces=torch.load(root/"FLAME_with_eye.pt",map_location="cpu",weights_only=False)["flame_model"]["f"].long().to(dev)
tt=pickle.load(open("/home/ubuntu/GAGAvatar_track/outputs/Nidhi_clean/optim.pkl","rb"))
def yaw(e):
    R=np.asarray(e["transform_matrix"]).reshape(3,4)[:3,:3]; return np.degrees(np.arctan2(R[0,2],R[2,2]))
ks=sorted(tt.keys(),key=lambda k:int(k.split("_")[-1])); c=np.median([yaw(tt[k]) for k in ks])
frontal=sorted(ks,key=lambda k:abs(yaw(tt[k])-c))[:NFR]
# more closed-eye coverage to constrain the lid
VAR=[("neu",0,0),("neu",0,0),("jaw10",0.10,0),("jaw20",0.20,0),("jaw35",0.35,0),
     ("eye30",0,0.3),("eye45",0,0.45),("eye60",0,0.6),("eye60",0,0.6)]
# eyelid band (rows) in the 512 render where the closed lid sits
EY0,EY1,EX0,EX1=235,285,150,362
S=[]
with torch.no_grad():
    for k in frontal:
        exp=torch.tensor(np.asarray(tt[k]["expcode"]).reshape(-1)[:100]).float().to(dev)
        for tag,jaw,eye in VAR:
            m=torch.zeros(106,device=dev); m[:100]=exp; m[103]=jaw
            b=rt.build_forward_batch(m[None]); cam=b["t_transform"][0].clone()
            if eye>0:
                ep=torch.zeros(1,6,device=dev); ep[0,0]=eye; ep[0,3]=eye
                b["t_points"]=rt.flame_model(shape_params=rt._shape_code,pose_params=torch.zeros(1,6,device=dev),expression_params=exp[None],eye_pose_params=ep).float()
            timg=rt.render_rgb_batch(b)[0].permute(1,2,0).clamp(0,1).clone()
            S.append((b["t_points"][0].clone(),timg,cam,tag,jaw/JREF,eye/EREF))
print("[v4] samples",len(S),flush=True)
ck=torch.load(WARM,map_location=dev,weights_only=False)
lq=ck["local_quat"].to(dev); pi=ck["parent_idx"].to(dev); N=pi.shape[0]
P=lambda t: t.to(dev).clone().requires_grad_(True)
sh=P(ck["sh"]); ol=P(ck["opacity_logit"]); ls=P(ck["log_scale"]); lo=P(ck["local_offset"])
dshJ=P(ck["dsh_jaw"]); dshB=P(ck["dsh_blink"]); dopJ=P(ck["dop_jaw"]); dopB=P(ck["dop_blink"])
sh0=ck["sh"].to(dev).clone(); ol0=ck["opacity_logit"].to(dev).clone()
dshB0=ck["dsh_blink"].to(dev).clone()
def vm(c):
    V=torch.eye(4,device=dev); V[:3,:3]=c[:3,:3]; V[:3,3]=c[:3,3]; V[:2,:]*=-1; return V[None]
def render(v,c,wj,wb):
    bb,R,k=compute_triangle_frames(v,faces); pos,q,s=bind_gaussians(lo,lq,torch.exp(ls),pi,bb,R,k)
    o,_,_=rasterization(pos,q,s,torch.sigmoid(ol+wj*dopJ+wb*dopB),sh+wj*dshJ+wb*dshB,vm(c),K,width=W,height=H,sh_degree=3)
    return o[0].clamp(0,1)
def sat(img):  # per-pixel colour saturation (max-min over channels)
    return img.max(2).values-img.min(2).values
def band_tv(img):  # total variation in the eyelid band
    b=img[EY0:EY1,EX0:EX1]
    return (b[:,1:]-b[:,:-1]).abs().mean()+(b[1:]-b[:-1]).abs().mean()
opt=torch.optim.Adam([{"params":[sh],"lr":0.002},{"params":[ol],"lr":0.002},{"params":[ls],"lr":0.0005},{"params":[lo],"lr":0.0001},
                      {"params":[dshJ,dshB],"lr":0.006},{"params":[dopJ,dopB],"lr":0.01}],betas=(0.9,0.99))
rng=np.random.default_rng(1); t0=time.time()
for it in range(ITERS):
    v,timg,cam,tag,wj,wb=S[int(rng.integers(len(S)))]
    im=render(v,cam,wj,wb); img=F.l1_loss(im,timg)
    anchor=(sh-sh0).abs().mean()+0.1*(ol-ol0).abs().mean()
    sparse=dshJ.abs().mean()+dshB.abs().mean()+0.1*(dopJ.abs().mean()+dopB.abs().mean())
    # GLITTER penalty only on closed-eye frames in the eyelid band: student saturation
    # should not exceed teacher saturation; plus a TV smoothness term.
    glit=torch.zeros((),device=dev)
    if wb>0.4:
        es=sat(im)[EY0:EY1,EX0:EX1]; ts=sat(timg)[EY0:EY1,EX0:EX1]
        glit=(es-ts).clamp_min(0).pow(2).mean()*wb + 0.5*band_tv(im)*wb
    loss=img+0.05*anchor+0.01*sparse+10*(lo.abs()-0.3).clamp_min(0).pow(2).mean()+2.0*glit
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        s=torch.exp(ls); ls.data=torch.log(torch.maximum(s,s.max(1,keepdim=True).values/8).clamp_min(1e-6))
    if it%500==0: print("[v4] it",it,tag,"img",round(img.item(),4),"glit",round(float(glit),5),round((it+1)/(time.time()-t0)),"it/s",flush=True)
with torch.no_grad():
    for tag in ["neu","jaw35","eye60"]:
        errs=[F.l1_loss(render(v,cam,wj,wb),timg).item() for (v,timg,cam,t,wj,wb) in S if t==tag]
        print("[v4] eval",tag,"L1",round(float(np.mean(errs)),4),flush=True)
        v,timg,cam,t,wj,wb=[x for x in S if x[3]==tag][0]
        imageio.imwrite(OUT/"triptych"/f"{tag}.png",(torch.cat([timg,render(v,cam,wj,wb)],1).cpu().numpy()*255).astype(np.uint8))
torch.save({"sh":sh.detach(),"opacity_logit":ol.detach(),"log_scale":ls.detach(),"local_offset":lo.detach(),"local_quat":lq,"parent_idx":pi,"sh_degree":3,
            "dsh_jaw":dshJ.detach(),"dsh_blink":dshB.detach(),"dop_jaw":dopJ.detach(),"dop_blink":dopB.detach(),"jaw_ref":JREF,"eye_ref":EREF},OUT/"ckpt_distilled.pt")
print("[v4] DONE",flush=True)
