"""Verify the exported browser asset renders cleanly (through the known-good
GAGAvatar frontal camera, after undoing the flip_y bake). Writes _asset_check.png."""
import sys, json, math, pickle, pathlib
import numpy as np, torch
from gsplat import rasterization

GAGA = pathlib.Path(sys.argv[1])
DRV = pathlib.Path(sys.argv[2])
ASSET = pathlib.Path(sys.argv[3])
dev = "cuda"

meta = json.load(open(ASSET / "niva_standalone.json"))
n, st = meta["num_gaussians"], meta["stride_floats"]
m = np.fromfile(ASSET / "niva_standalone.bin", dtype="<f4").reshape(n, st)

pos = m[:, 0:3].copy(); pos[:, 1] *= -1.0          # undo flip_y bake
rot = m[:, 3:7].copy(); rot[:, 2] *= -1.0
pos = torch.tensor(pos, device=dev); rot = torch.tensor(rot, device=dev)
sc = torch.tensor(m[:, 7:10], device=dev); op = torch.tensor(m[:, 10], device=dev)
col = torch.tensor(m[:, 11:14], device=dev)

optim = pickle.load(open(DRV / "optim.pkl", "rb"))
def yaw(e):
    R = np.asarray(e["transform_matrix"])[:3, :3]
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))
fk = min(optim, key=lambda k: abs(abs(yaw(optim[k])) - 180))
cam = torch.tensor(np.asarray(optim[fk]["transform_matrix"]), device=dev).float()[:3, :4]
V = torch.eye(4, device=dev); V[:3, :3] = cam[:3, :3]; V[:3, 3] = cam[:3, 3]; V[:2, :] *= -1; V = V[None]

from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig
from gagavatar.assets import GAGAvatarAssets
a = GAGAvatarAssets(root=GAGA / "assets", model_path=GAGA / "assets/GAGAvatar.pt",
                    tracked_path=GAGA / "render_results/tracked/tracked.pt",
                    flame_model_path=GAGA / "assets/FLAME_with_eye.pt")
rt = GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=a, device=dev)); rt.set_avatar_id("front.png")
cp = rt.model.cam_params
W = H = int(cp["size"][0]); fx = cp["focal_x"] * W / 2; fy = cp["focal_y"] * H / 2
K = torch.tensor([[[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]]], device=dev)
sh = ((col - 0.5) / 0.2820948)[:, None, :]
out, _, _ = rasterization(pos, rot, sc, op, sh, V, K, width=W, height=H,
                          sh_degree=0, render_mode="RGB")
import imageio.v2 as imageio
im = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
imageio.imwrite("/home/ubuntu/_asset_check.png", im)
print("wrote _asset_check.png img mean", round(float(im.mean()), 1), "scale max", round(float(sc.max()), 4))
