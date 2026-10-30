"""Animation probe: drive the trained standalone splat with different FLAME
expression/jaw params and render frontally, to confirm the avatar ANIMATES
(talks/blinks/emotes) — the actual interactive-avatar deliverable.

Loads an S3 checkpoint, then for a set of expression codes (neutral, smile,
mouth-open/talking, blink) poses the FLAME mesh, binds the splat, and renders a
frontal view. Writes a strip so we can see the mouth/eyes change.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gaga-root", required=True, type=Path)
    p.add_argument("--niva-src", required=True, type=Path)
    p.add_argument("--driver", required=True, type=Path, help="turntable driver (for a frontal cam + base pose)")
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
    args.out.mkdir(parents=True, exist_ok=True)

    assets = GAGAvatarAssets(
        root=args.gaga_root / "assets",
        model_path=args.gaga_root / "assets/GAGAvatar.pt",
        tracked_path=args.gaga_root / "render_results/tracked/tracked.pt",
        flame_model_path=args.gaga_root / "assets/FLAME_with_eye.pt",
    )
    rt = GAGAvatarRuntime(GAGAvatarRuntimeConfig(assets=assets, device=dev))
    rt.set_avatar_id(args.avatar_id)
    cam_params = rt.model.cam_params
    H, W = int(cam_params["size"][0]), int(cam_params["size"][1])
    fx = float(cam_params["focal_x"]) * W / 2.0
    fy = float(cam_params["focal_y"]) * H / 2.0
    K = torch.tensor([[[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]]], device=dev)

    flame = torch.load(args.gaga_root / "assets/FLAME_with_eye.pt",
                       map_location="cpu", weights_only=False)
    faces = flame["flame_model"]["f"].long().to(dev)

    # base splat shell + load trained params
    splat = init_splat_from_faces(faces, torch.zeros(5023, 3, device=dev),
                                  sh_degree=args.sh_degree,
                                  splats_per_face=args.splats_per_face, device=dev)
    s = torch.load(args.ckpt, map_location=dev)
    sh = s["sh"]; opacity_logit = s["opacity_logit"]
    log_scale = s["log_scale"]; local_offset = s["local_offset"]
    local_quat = splat.local_quat
    scale = torch.exp(log_scale)
    opac = torch.sigmoid(opacity_logit)

    # a frontal camera: use the turntable frame whose yaw is closest to frontal
    with open(args.driver / "optim.pkl", "rb") as f:
        optim = pickle.load(f)
    import math
    def yaw_of(e):
        R = np.asarray(e["transform_matrix"])[:3, :3]
        return math.degrees(math.atan2(R[0, 2], R[2, 2]))
    # frontal ~ +/-180 (from earlier yaw analysis front.png was -178.6)
    front_key = min(optim, key=lambda k: abs(abs(yaw_of(optim[k])) - 180.0))
    cam = torch.as_tensor(optim[front_key]["transform_matrix"], device=dev).float()[:3, :4]
    base_pose = torch.as_tensor(optim[front_key]["posecode"], device=dev).float().reshape(-1)
    base_exp = torch.as_tensor(optim[front_key]["expcode"], device=dev).float().reshape(-1)[:100]

    def viewmat(c):
        V = torch.eye(4, device=dev); V[:3, :3] = c[:3, :3]; V[:3, 3] = c[:3, 3]
        V[:2, :] *= -1.0; return V[None]

    @torch.no_grad()
    def render_expr(exp100, jaw):
        po = base_pose.clone()
        po[3] = jaw  # jaw open amount (first jaw-pose component)
        motion = torch.cat([exp100, po[:3], po[3:6]])[None]
        b = rt.build_forward_batch(motion)
        verts = b["t_points"][0]
        bb, R, k = compute_triangle_frames(verts, faces)
        pos, quat, wscale = bind_gaussians(local_offset, local_quat, scale,
                                           splat.parent_idx, bb, R, k)
        out, _, _ = rasterization(pos, quat, wscale, opac, sh, viewmat(cam), K,
                                  width=W, height=H, sh_degree=args.sh_degree,
                                  render_mode="RGB")
        return (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # expression set: neutral, smile (exp[0] up), mouth-open (jaw), 'surprise'
    neutral = base_exp.clone()
    smile = base_exp.clone(); smile[0] += 2.0
    talk = base_exp.clone()
    strip = [
        render_expr(neutral, jaw=0.0),            # neutral
        render_expr(smile, jaw=0.05),             # slight smile+open
        render_expr(talk, jaw=0.25),              # mouth open (talking)
        render_expr(talk, jaw=0.45),              # wide open
    ]
    row = np.concatenate(strip, axis=1)
    imageio.imwrite(args.out / "anim_strip.png", row)
    print(f"[anim] wrote anim_strip.png  frontal key={front_key}  "
          f"[neutral | smile | talk | wide]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
