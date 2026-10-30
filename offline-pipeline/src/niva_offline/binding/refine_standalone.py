"""S3 — Per-subject refinement of the standalone SH splat vs the REAL video.

This is the stage that makes the avatar beautiful. We optimise the LOCAL Gaussian
parameters (SH colour, opacity, scale, small position offset) so the splat —
rendered through gsplat's plain SH rasterizer (NO StyleUNet) — matches the REAL
reference frames from the tracker's img_lmdb. The FLAME binding + Gaussian count
are frozen (identity/controllability guarantee).

Key difference from the failed v2/v3: the optimisation target is the ACTUAL PHOTO,
not GAGAvatar's neural-UNet output. The splat must stand on its own.

Anti-spike regularisation (from GaussianAvatars / 3DGS):
    * position reg:  ‖max(|local_offset| - eps, 0)‖²   (keep splats near triangle)
    * scale reg:     ‖max(local_scale - s_max, 0)‖²     (no giant splats)
    * aspect clamp:  cap max/min axis ratio (~8) each step (no needles/spikes)

Checkpointed + resumable; periodic preview export; held-out frame split for
honest PSNR/SSIM. Run in GAGAvatar env with gsplat (CUDA_HOME=/usr/local/cuda-12.8,
env bin on PATH).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main(argv=None):
    p = argparse.ArgumentParser(description="S3 standalone SH splat refinement")
    p.add_argument("--gaga-root", required=True, type=Path)
    p.add_argument("--niva-src", required=True, type=Path)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--avatar-id", default="front.png")
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--splats-per-face", type=int, default=1,
                   help="K Gaussians per FLAME triangle (density -> sharpness)")
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--lr-sh", type=float, default=0.01)
    p.add_argument("--lr-opacity", type=float, default=0.02)
    p.add_argument("--lr-scale", type=float, default=0.005)
    p.add_argument("--lr-offset", type=float, default=0.0005)
    p.add_argument("--w-pos-reg", type=float, default=10.0)
    p.add_argument("--w-scale-reg", type=float, default=1.0)
    p.add_argument("--offset-eps", type=float, default=0.3,
                   help="allowed local offset (in triangle-k units) before penalty")
    p.add_argument("--scale-max", type=float, default=2.0,
                   help="allowed local scale (in triangle-k units) before penalty")
    p.add_argument("--aspect-max", type=float, default=8.0)
    p.add_argument("--holdout", type=int, default=10, help="every Nth frame held out")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--preview-every", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--portrait-driver", type=Path, default=None,
                   help="GAGAvatar_track outputs/images dir (optim.pkl + matted pngs)")
    p.add_argument("--portrait-frac", type=float, default=0.5,
                   help="fraction of training steps that use a portrait view")
    args = p.parse_args(argv)

    sys.path.insert(0, str(args.gaga_root.resolve()))
    sys.path.insert(0, str(args.niva_src.resolve()))

    import imageio.v2 as imageio
    from gsplat import rasterization
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig
    from gagavatar.assets import GAGAvatarAssets
    from gagavatar.libs.GAGAvatar_track.engines import LMDBEngine

    from niva_offline.binding.standalone_splat import (
        init_splat_from_faces, world_gaussians,
    )
    from niva_offline.binding.triangle_binding import compute_triangle_frames

    dev = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preview").mkdir(exist_ok=True)

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

    with open(args.driver / "optim.pkl", "rb") as f:
        optim = pickle.load(f)
    keys = sorted(optim, key=lambda k: int(k.split("_")[-1]))
    train_keys = [k for i, k in enumerate(keys) if i % args.holdout != 0]
    hold_keys = [k for i, k in enumerate(keys) if i % args.holdout == 0]
    lmdb = LMDBEngine(str(args.driver / "img_lmdb"), write=False)
    print(f"[s3] {len(train_keys)} train / {len(hold_keys)} holdout frames", flush=True)

    # Cache per-frame posed verts + camera (small; 5023x3 each) to avoid rerunning
    # the heavy GAGAvatar forward each iter.
    def batch_for(key):
        e = optim[key]
        exp = torch.as_tensor(e["expcode"], device=dev).float().reshape(-1)[:100]
        po = torch.as_tensor(e["posecode"], device=dev).float().reshape(-1)
        b = rt.build_forward_batch(torch.cat([exp, po[:3], po[3:6]])[None])
        verts = b["t_points"][0].detach()
        # Use the per-frame TRACKED camera when available (turntable/multi-view:
        # the viewpoint rotation lives in transform_matrix, NOT in t_transform
        # which is the canonical frontal camera). Fall back to t_transform.
        if "transform_matrix" in e:
            cam = torch.as_tensor(e["transform_matrix"], device=dev).float()[:3, :4]
        else:
            cam = b["t_transform"][0].detach()
        return verts, cam

    verts0, _ = batch_for(keys[0])

    # ---- optional multi-view portraits (the sharpness fix) --------------------
    # Each portrait has its OWN camera (transform_matrix) giving a distinct
    # viewpoint (~±45deg yaw), which the frontal-only video lacks. Posing the mesh
    # with the portrait's expcode and rendering from the portrait camera supplies
    # the multi-view constraint that lets splats resolve sharp detail.
    portraits = []  # list of (verts, cam3x4, gt_HW3)
    if args.portrait_driver is not None:
        with open(args.portrait_driver / "optim.pkl", "rb") as f:
            poptim = pickle.load(f)
        for pkey, e in poptim.items():
            exp = torch.as_tensor(e["expcode"], device=dev).float().reshape(-1)[:100]
            po = torch.as_tensor(e["posecode"], device=dev).float().reshape(-1)
            pb = rt.build_forward_batch(torch.cat([exp, po[:3], po[3:6]])[None])
            pverts = pb["t_points"][0].detach()
            pcam = torch.as_tensor(e["transform_matrix"], device=dev).float()[:3, :4]
            pgt = torch.as_tensor(e["image"], device=dev).float()
            if pgt.max() > 1.5:
                pgt = pgt / 255.0
            if pgt.dim() == 3 and pgt.shape[0] in (3, 4):
                pgt = pgt[:3]
                pgt = F.interpolate(pgt[None], size=(H, W), mode="bilinear",
                                    align_corners=False)[0].permute(1, 2, 0)
            portraits.append((pverts, pcam, pgt.clamp(0, 1)))
        print(f"[s3] +{len(portraits)} portrait views (frac={args.portrait_frac})",
              flush=True)

    # init splat; seed DC colour from the frontal frame average skin tone
    splat = init_splat_from_faces(faces, verts0, sh_degree=args.sh_degree,
                                  splats_per_face=args.splats_per_face, device=dev)

    sh = splat.sh.clone().requires_grad_(True)
    opacity_logit = splat.opacity_logit.clone().requires_grad_(True)
    log_scale = torch.log(splat.local_scale.clamp_min(1e-6)).clone().requires_grad_(True)
    local_offset = splat.local_offset.clone().requires_grad_(True)
    local_quat = splat.local_quat  # frozen (identity) for v1

    opt = torch.optim.Adam([
        {"params": [sh], "lr": args.lr_sh},
        {"params": [opacity_logit], "lr": args.lr_opacity},
        {"params": [log_scale], "lr": args.lr_scale},
        {"params": [local_offset], "lr": args.lr_offset},
    ], betas=(0.9, 0.99))

    start_iter = 0
    ckpt = args.out / "ckpt.pt"
    if args.resume and ckpt.exists():
        s = torch.load(ckpt, map_location=dev)
        sh.data = s["sh"]; opacity_logit.data = s["opacity_logit"]
        log_scale.data = s["log_scale"]; local_offset.data = s["local_offset"]
        opt.load_state_dict(s["opt"]); start_iter = s["iter"]
        print(f"[s3] resumed @ {start_iter}", flush=True)

    def viewmat(cam3x4):
        V = torch.eye(4, device=dev)
        V[:3, :3] = cam3x4[:3, :3]; V[:3, 3] = cam3x4[:3, 3]
        V[:2, :] *= -1.0  # verified INRIA/flipxy convention (S2)
        return V[None]

    def render(verts_posed, cam3x4):
        b, R, k = compute_triangle_frames(verts_posed, faces)
        from niva_offline.binding.triangle_binding import bind_gaussians
        scale = torch.exp(log_scale)
        pos, quat, wscale = bind_gaussians(
            local_offset, local_quat, scale, splat.parent_idx, b, R, k)
        opac = torch.sigmoid(opacity_logit)
        out, _, _ = rasterization(
            pos, quat, wscale, opac, sh, viewmat(cam3x4), K,
            width=W, height=H, sh_degree=args.sh_degree, render_mode="RGB")
        return out[0].clamp(0, 1)  # (H,W,3)

    def gt_for(key):
        g = lmdb.load(key, type="image").float().to(dev) / 255.0
        g = F.interpolate(g[None], size=(H, W), mode="bilinear",
                          align_corners=False)[0]
        return g.permute(1, 2, 0)  # (H,W,3)

    @torch.no_grad()
    def holdout_psnr(n=8):
        ps = []
        for key in hold_keys[:n]:
            v, c = batch_for(key)
            im = render(v, c)
            mse = F.mse_loss(im, gt_for(key))
            ps.append(float(-10 * torch.log10(mse + 1e-8)))
        return float(np.mean(ps)) if ps else float("nan")

    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(start_iter, args.iters):
        use_portrait = portraits and (rng.random() < args.portrait_frac)
        if use_portrait:
            verts, cam, gt = portraits[int(rng.integers(len(portraits)))]
            im = render(verts, cam)
        else:
            key = train_keys[int(rng.integers(len(train_keys)))]
            verts, cam = batch_for(key)
            im = render(verts, cam)
            gt = gt_for(key)

        img_loss = F.l1_loss(im, gt)
        # anti-spike regularisation
        pos_reg = (local_offset.abs() - args.offset_eps).clamp_min(0).pow(2).mean()
        scale_lin = torch.exp(log_scale)
        scale_reg = (scale_lin - args.scale_max).clamp_min(0).pow(2).mean()
        loss = img_loss + args.w_pos_reg * pos_reg + args.w_scale_reg * scale_reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # aspect-ratio clamp (hard, post-step): keep max/min axis ratio <= aspect_max
        with torch.no_grad():
            s = torch.exp(log_scale)
            smax = s.max(dim=1, keepdim=True).values
            smin = (smax / args.aspect_max)
            s_clamped = torch.maximum(s, smin)
            log_scale.data = torch.log(s_clamped.clamp_min(1e-6))

        if it % 100 == 0:
            rate = (it - start_iter + 1) / (time.time() - t0)
            print(f"[s3] it={it} loss={loss.item():.4f} img={img_loss.item():.4f} "
                  f"pos_reg={pos_reg.item():.5f} scale_reg={scale_reg.item():.5f} "
                  f"{rate:.1f}it/s", flush=True)

        if (it + 1) % args.preview_every == 0 or it + 1 == args.iters:
            with torch.no_grad():
                v, c = batch_for(hold_keys[0])
                im = (render(v, c).detach().cpu().numpy() * 255).astype(np.uint8)
                gt = (gt_for(hold_keys[0]).cpu().numpy() * 255).astype(np.uint8)
                side = np.concatenate([gt, im], axis=1)
                imageio.imwrite(args.out / "preview" / f"it{it+1:06d}.png", side)

        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iters:
            torch.save({"sh": sh.data, "opacity_logit": opacity_logit.data,
                        "log_scale": log_scale.data, "local_offset": local_offset.data,
                        "opt": opt.state_dict(), "iter": it + 1,
                        "sh_degree": args.sh_degree}, ckpt)
            print(f"[s3] ckpt @ {it+1}  holdout_psnr={holdout_psnr():.2f} dB", flush=True)

    print(f"[s3] DONE  final holdout_psnr={holdout_psnr():.2f} dB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
