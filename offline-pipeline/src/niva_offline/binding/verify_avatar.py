"""Render the trained S3 standalone avatar via gsplat (the SAME renderer that
trained it) from several yaw angles, so you can verify on the instance that the
AVATAR is good and only the browser/WebGL2 path was the problem.

Writes a strip PNG: ~/_verify_avatar.png  (5 views: -40, -20, 0, +20, +40 deg)

Run on the box:
  cd ~/NIVA-Neural-Intelligent-Virtual-Assistant/offline-pipeline/checkpoints/repos/GAGAvatar
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH=~/miniconda3/envs/GAGAvatar/bin:$CUDA_HOME/bin:$PATH
  GAGA=$PWD; NIVA=~/NIVA-Neural-Intelligent-Virtual-Assistant/offline-pipeline/src
  PYTHONPATH=$GAGA:$NIVA python -m niva_offline.binding.verify_avatar \
      --gaga-root $GAGA --niva-src $NIVA \
      --driver $GAGA/../GAGAvatar_track/outputs/Nidhi_clean \
      --ckpt ~/niva-artifacts/standalone/s3_turntable/ckpt.pt \
      --out ~/_verify_avatar.png --splats-per-face 16
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gaga-root", required=True, type=Path)
    p.add_argument("--niva-src", required=True, type=Path)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--splats-per-face", type=int, default=16)
    p.add_argument("--avatar-id", default="front.png")
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.gaga_root.resolve()))
    sys.path.insert(0, str(args.niva_src.resolve()))

    import imageio.v2 as imageio
    from gsplat import rasterization
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig
    from gagavatar.assets import GAGAvatarAssets
    from niva_offline.binding.standalone_splat import init_splat_from_faces
    from niva_offline.binding.triangle_binding import compute_triangle_frames, bind_gaussians

    dev = "cuda"
    assets = GAGAvatarAssets(
        root=args.gaga_root / "assets",
        model_path=args.gaga_root / "assets/GAGAvatar.pt",
        tracked_path=args.gaga_root / "render_results/tracked/tracked.pt",
        flame_model_path=args.gaga_root / "assets/FLAME_with_eye.pt",
    )
    rt = GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=assets, device=dev))
    rt.set_avatar_id(args.avatar_id)
    cp = rt.model.cam_params
    H, W = int(cp["size"][0]), int(cp["size"][1])
    fx = float(cp["focal_x"]) * W / 2.0
    fy = float(cp["focal_y"]) * H / 2.0
    K = torch.tensor([[[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]]], device=dev)

    flame = torch.load(args.gaga_root / "assets/FLAME_with_eye.pt",
                       map_location="cpu", weights_only=False)
    faces = flame["flame_model"]["f"].long().to(dev)

    # frontal driver frame (neutral-ish) for pose + camera
    with open(args.driver / "optim.pkl", "rb") as f:
        optim = pickle.load(f)
    def yaw_of(e):
        R = np.asarray(e["transform_matrix"])[:3, :3]
        return math.degrees(math.atan2(R[0, 2], R[2, 2]))
    fk = min(optim, key=lambda k: abs(abs(yaw_of(optim[k])) - 180.0))
    exp = torch.as_tensor(optim[fk]["expcode"], device=dev).float().reshape(-1)[:100]
    po = torch.as_tensor(optim[fk]["posecode"], device=dev).float().reshape(-1)
    cam = torch.as_tensor(optim[fk]["transform_matrix"], device=dev).float()[:3, :4]
    batch = rt.build_forward_batch(torch.cat([exp, po[:3], po[3:6]])[None])
    verts = batch["t_points"][0]

    # load trained splat
    splat = init_splat_from_faces(faces, verts, sh_degree=args.sh_degree,
                                  splats_per_face=args.splats_per_face, device=dev)
    s = torch.load(args.ckpt, map_location=dev)
    sh = s["sh"]; opac = torch.sigmoid(s["opacity_logit"])
    scale = torch.exp(s["log_scale"]); loff = s["local_offset"]; lquat = splat.local_quat

    b, R, k = compute_triangle_frames(verts, faces)
    pos, quat, wscale = bind_gaussians(loff, lquat, scale, splat.parent_idx, b, R, k)
    ctr = pos.mean(0)

    def viewmat(c):
        V = torch.eye(4, device=dev); V[:3, :3] = c[:3, :3]; V[:3, 3] = c[:3, 3]
        V[:2, :] *= -1.0; return V

    def roty(deg):
        r = math.radians(deg); cc, ss = math.cos(r), math.sin(r)
        return torch.tensor([[cc, 0, ss], [0, 1, 0], [-ss, 0, cc]], device=dev, dtype=torch.float32)

    strip = []
    for ang in [-40, -20, 0, 20, 40]:
        Rm = roty(ang)
        pr = (pos - ctr) @ Rm.T + ctr
        out, _, _ = rasterization(pr, quat, wscale, opac, sh, viewmat(cam)[None], K,
                                  width=W, height=H, sh_degree=args.sh_degree,
                                  render_mode="RGB")
        im = (out[0].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        strip.append(im)
    row = np.concatenate(strip, axis=1)
    imageio.imwrite(args.out, row)
    print(f"[verify] wrote {args.out}  N={pos.shape[0]}  views=[-40,-20,0,+20,+40] deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
