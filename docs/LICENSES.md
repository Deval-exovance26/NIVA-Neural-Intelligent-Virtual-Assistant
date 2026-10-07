# NIVA — Dependency License Audit

Scope: every third-party model, repo, and library the NIVA 3DGS avatar pipeline
depends on (offline build + browser runtime). This satisfies Task 2 in
`docs/plan.md` ("License & dependency audit — tracked, non-blocking").

**Commercial posture:** NIVA is a commercial product. The audit flags any
component whose license restricts commercial use. The single launch-gating item
is **GaussianAvatars** (Toyota non-commercial) — it is flagged, not a build
blocker, because two concrete escape hatches exist (see its row + the dedicated
section below).

> Licenses below are recorded from each project's public repository at the time
> of writing. **Verify the current `LICENSE` file in each repo before launch** —
> upstreams can relicense. Several model *weights* carry separate, often more
> restrictive terms than the *code* (noted per row).

## Audit table

| Component | Repo | License | Commercial-OK? | Notes / mitigation |
|---|---|---|---|---|
| **FLAME 2020** | https://flame.is.tue.mpg.de/ (model); code: `TimoBolkart/TF_FLAME` | MPI FLAME license (research/academic; **commercial use requires a separate MPI/MPG commercial license**) | ⚠️ **Conditional** | License-gated manual signup. `scripts/fetch_models.sh` refuses to proceed without a user-provided download and never bypasses the gate. **Action:** obtain a commercial FLAME license from MPI for Intelligent Systems before launch. The parametric model is foundational; no drop-in permissive substitute. |
| **GAGAvatar** | https://github.com/xg-chu/GAGAvatar | Code typically permissive (confirm repo `LICENSE`); **pretrained weights trained on research datasets (e.g. VFHQ/CelebV-HQ) carry dataset terms** | ⚠️ **Conditional** | Usable now as the feed-forward Stage-1 stand-in. **Action:** confirm the code license text and, critically, the **training-data provenance of the released weights** — research-dataset-trained weights may restrict commercial deployment even if the code is permissive. Consider retraining on licensed data if blocked. |
| **GAGAvatar_track** | https://github.com/xg-chu/GAGAvatar_track | Code per repo `LICENSE`; bundles EMICA/MICA + mediapipe assets, each with **own upstream licenses** (MICA is **non-commercial research**) | ⚠️ **Conditional** | Used for FLAME tracking (Task 4). **Action:** audit the embedded tracker stack — MICA/EMICA weights are commonly **non-commercial**. Tracking is an *offline preprocessing* step producing FLAME params (not shipped in the product), which lowers but does not eliminate exposure; confirm the specific sub-model terms. |
| **GaussianAvatars** | https://github.com/ShenhanQian/GaussianAvatars | **Toyota Motor Europe — NON-COMMERCIAL research license** | 🚫 **NO (launch-gating)** | **The one hard flag.** Do **not** ship GaussianAvatars code/weights in the commercial product. We only need its triangle-binding math (Eq. 1, §3.2). **Two escape hatches** — (1) **reimplement the triangle-binding transform** from standard FLAME mesh math (local frame → world pos/rot/scale; it is not Toyota-proprietary), or (2) use **FlashAvatar's permissive UV-embedding** for the refinement stage. See section below. |
| **ARTalk** | https://github.com/xg-chu/ARTalk | Per repo `LICENSE` (confirm); audio→FLAME motion model | ⚠️ **Conditional** | Audio→FLAME driver (Task 12). **Action:** confirm code + weight license and the training-audio dataset terms. |
| **UniLS** (ARTalk successor) | https://github.com/xg-chu/UniLS | Per repo `LICENSE` (confirm) | ⚠️ **Conditional** | Candidate upgrade over ARTalk; same audit obligations. Evaluate in Task 12. |
| **DINOv3** | https://github.com/facebookresearch/dinov3 | **DINOv3 License (Meta) — gated; permits commercial use under its terms, with use restrictions; weights require accepting Meta's terms** | ⚠️ **Conditional** | Only needed if a from-scratch SpatialAvatar-0 generator is built (not in current plan — GAGAvatar is the stand-in). **Action:** if used, accept Meta's DINOv3 terms and review the acceptable-use restrictions for commercial deployment. |
| **gsplat** | https://github.com/nerfstudio-project/gsplat | **Apache-2.0** | ✅ **Yes** | Permissive. Offline/validation rasterization backend. No restriction. |
| **Three.js** | https://github.com/mrdoob/three.js | **MIT** | ✅ **Yes** | Permissive. Browser WebGPU host/renderer. No restriction. |

Legend: ✅ clear for commercial use · ⚠️ allowed only after confirming/obtaining
terms (gated, dataset-dependent, or sub-component risk) · 🚫 not permitted as-is.

## GaussianAvatars — the launch-gating flag (detail)

GaussianAvatars is released under a **Toyota Motor Europe non-commercial research
license**. Shipping its code or weights in a commercial product is **not
permitted**. NIVA's dependency on it is narrow: the **rigid triangle-binding
transform** (SpatialAvatar-0 Eq. 1 / GaussianAvatars §3.2) that maps a Gaussian's
local-frame attributes to world space via its parent FLAME triangle.

Two escape hatches (either removes the block):

1. **Reimplement the triangle-binding math.** The transform
   `x_world = R_f (x_local · s_f) + c_f`, `q_world = q_f ⊗ q_local`,
   `s_world = s_local · s_f` is derived from standard FLAME mesh geometry (the
   parent triangle's centroid `c_f`, orientation `R_f`, and scale `s_f`). This is
   general computer-graphics math, not Toyota IP — implement it cleanly from the
   FLAME mesh (planned as Task 7, `niva_offline.binding`).
2. **FlashAvatar UV embedding.** Use FlashAvatar's permissively-licensed
   UV-space Gaussian embedding for the per-subject refinement stage instead of
   GaussianAvatars' binding, if a from-scratch reimplementation is undesirable.

**Recommendation:** take escape hatch (1) — it is the smallest, cleanest change
and keeps the one-Gaussian-per-valid-UV-pixel layout intact. Track it as a
pre-launch task; the research/dev build may reference GaussianAvatars for
validation only, never in shipped artifacts.

## Pre-launch checklist (non-blocking for the build; blocking for commercial launch)

- [ ] Obtain a **commercial FLAME license** from MPI (or confirm scope).
- [ ] Verify **GAGAvatar weight** training-data provenance for commercial use.
- [ ] Audit **GAGAvatar_track** embedded sub-models (MICA/EMICA are often
      non-commercial); confirm tracking-only/offline use is acceptable.
- [ ] Replace **GaussianAvatars** binding with escape hatch (1) or (2); ensure no
      GaussianAvatars code/weights ship in the product.
- [ ] Confirm **ARTalk/UniLS** code + weight + training-audio terms.
- [ ] If **DINOv3** is used, accept Meta terms + review use restrictions.
- [ ] gsplat (Apache-2.0) and Three.js (MIT): ✅ no action.
