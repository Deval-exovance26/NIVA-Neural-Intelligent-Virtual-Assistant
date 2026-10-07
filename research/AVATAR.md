# SpatialAvatar-0: High-Quality 4D Head Avatar with Multi-Stage Reconstruction

**Yiran Wang**1∗ **Zeyu Zhang**2∗ **Yuanming Li**2∗ **Ziming Wang**3 **Yang Zhao**4† 1USYD 2SpatialReal 3ZJU 4La Trobe ∗Equal contribution. †Corresponding author: y.zhao2@latrobe.edu.au.

## Abstract

High-quality 4D head avatars from one or a few source portraits are central to telepresence, AR/VR, and digital-human interaction. 3D Gaussian Splatting (3DGS) has emerged as the dominant representation, with two complementary regimes (generalizable feed-forward predictors and per-subject refiners) maturing in parallel. However, existing feed-forward predictors are trained on a single dataset family with a hard-coded source count, inheriting the corresponding domain bias. Per-subject refiners require 300K–600K iterations and rely on adaptive densification that destroys upstream Gaussian layouts, preventing the two regimes from sharing a representation end-to-end. To bridge both regimes we propose **SpatialAvatar-0** on a shared FLAME-mesh-bound Gaussian representation: a feed-forward generator with a parameter-free K-source mean-pool and a monocular-temporal → multi-view-spatial two-phase schedule that anchors against identity-prior collapse onto the smaller multi-view set. We further introduce a 10K-iter layout-preserving per-subject refinement loop that freezes the FLAMEbinding and Gaussian count and replaces densification with a three-component anti-spike regularization. On VFHQ/HDTF cross-domain zero-shot we surpass the in-domain leader GAGAvatar by +**1**.**5** dB PSNR despite never training on either test domain, and on the SplattingAvatar monocular benchmark we lead every reported metric, surpassing the 300K-iter GeoAvatar by +**1**.**3** dB PSNR at up to **60**× shorter per-subject schedule than common SOTA baselines. Website: https://spatialwalk.github.io/SpatialAvatar-0

## 1 Introduction

High-quality 4D head avatars are a building block for telepresence, AR/VR communication, and digital-human interaction. Within the explicit 3D Gaussian Splatting (3DGS) line of head-avatar research [28], two complementary regimes have matured: generalizable feed-forward predictors [10, 11, 13, 66] that emit an animatable face-bound Gaussian model in a single forward pass from one or a few source portraits, and per-subject 3DGS refiners [48, 62, 43, 52] that continue to optimize a face-bound Gaussian model against a held-out video of a single identity to recover subject-specific high-frequency detail. Production deployments of digital humans need both regimes, so the practical question has shifted from “feed-forward or per-subject” to whether the two regimes can share a common Gaussian representation end-to-end. Two long-standing gaps prevent such a unified pipeline. First, existing feed-forward Gaussian avatars are trained on a single dataset family (either monocular video [77] or multi-view capture [30]) and inherit the corresponding domain bias: monocular-trained predictors lack ground-truth multiview geometry, while multi-view-trained predictors see orders-of-magnitude fewer identities and degrade on in-the-wild portraits. Architectures further hard-code the source-image count, preventing a single network from leveraging however many source views happen to be available at deployment

Preprint.

time. Second, per-subject 3DGS refinement consumes 300K–600K iterations per identity (∼5–9 hours) [62, 48, 43] and relies on adaptive densification that destroys any spatial inductive bias inherited from a feed-forward initialization. Removing densification, however, exposes the wellknown 3DGS anisotropy failure: Gaussians elongate along a single axis to overfit individual training views, catastrophically breaking novel-view rendering. We observe that a Gaussian avatar trained on the union of monocular and multi-view supervision can in principle inherit wide-identity coverage from monocular pretraining and geometric grounding from multi-view post-training, provided the training signal does not collapse the broader identity prior onto the smaller multi-view set; we further want the same network to handle a variable source count without architectural switching. Separately, we want a per-subject refinement loop that is layoutpreserving: it freezes the FLAME-binding and Gaussian count of the feed-forward output while refining per-Gaussian attributes, so that the upstream identity prior remains reusable downstream and the refinement consumes orders of magnitude fewer iterations, provided we can suppress the 3DGS anisotropy spikes that emerge once densification is removed. We propose **SpatialAvatar-0**, a unified feed-forward and per-subject 4D head-avatar pipeline over a shared FLAME-mesh-bound Gaussian representation. First, addressing the feed-forward gap, we introduce a feed-forward image-to-Gaussian generator with a parameter-free K-source mean-pool aggregator that is the identity at K=1 and order-invariant for K>1, trained under a monocular-temporal → multi-view-spatial two-phase schedule with an L2-SP anchor [37] and a 25% NeRSemble crosstime mix that together prevent identity-prior collapse onto the smaller multi-view set. Second, addressing the per-subject gap, we propose a 10K-iter layout-preserving per-subject refinement loop that freezes the FLAME-triangle binding and the Gaussian count and replaces densification with a three-component anti-spike regularization (scale-freeze warmup, hard log-scale clamp, screen-space anti-anisotropy penalty) that keeps the maximum Gaussian aspect ratio bounded throughout optimization. Third, we conduct a comprehensive empirical evaluation: on cross-domain VFHQ and HDTF feed-forward zero-shot probes our model surpasses the in-domain leader GAGAvatar [10] by +**1**.**5** dB PSNR despite never observing either dataset during training; on the SplattingAvatar monocular benchmark our 10K-iter refinement surpasses the 300K-iter GeoAvatar [43] on every reported metric by +**1**.**3** dB PSNR while completing per-subject creation in ∼2 minutes (vs. 4.9 hours); ablations confirm each design choice is load-bearing. Our contributions are threefold:

- A K-source-variable feed-forward FLAME-mesh-bound Gaussian generator with a monocular → multi-view two-phase training schedule, anchored by L2-SP and a 25% NeRSemble cross-time mix against identity-prior collapse on the smaller multi-view set.
- A 10K-iter layout-preserving per-subject refinement loop with a three-component anti-spike regularization replacing densification, leading every reported metric on the SplattingAvatar leaderboard at up to **60**× shorter schedule than common SOTA per-subject baselines.
- Comprehensive cross-domain and per-subject experiments on VFHQ, HDTF, and the SplattingAvatar monocular benchmark (§4), with ablations validating every design choice.

## 2 Related Work

The creation of photorealistic and animatable head avatars has garnered significant attention within the computer vision and graphics communities, driven by applications in telepresence, gaming, and the metaverse. The fundamental objective of this field is to faithfully reconstruct or synthesize a source head from sparse inputs, such as a single portrait or a short monocular video, while enabling precise, fine-grained control over facial expressions, gaze, and head poses. Existing methodologies can be broadly categorized into 2D-based synthesis, implicit 3D reconstruction using radiance fields, and the emerging paradigm of explicit 3D Gaussian Splatting.

**2.1 2D-Based Talking Head Synthesis**

Early attempts to generate talking heads primarily relied on 2D generative models, such as GANs, to synthesize image sequences directly in the pixel space. A popular strategy among these works is the injection of expression and pose features from a driving image into a 2D generative backbone to achieve motion through feature modulation or latent space manipulation [69, 7, 57, 41, 40, 26, 9]. Another recent trend involves the estimation of dense 2D warp fields or optical flows to deform the source portrait into the target geometry [53, 50, 14, 23, 70, 54, 73]. To improve the physical realism and structural integrity of these deformations, several methods have integrated 3D Morphable Models (3DMMs) [6, 46, 34] to serve as a low-dimensional geometric prior [59, 67, 14, 70, 41, 51]. Some works further leverage the generative power of StyleGAN to produce highresolution textures conditioned on 3DMM coefficients [55, 68]. Although these methods produce visually compelling results, they inherently struggle with 3D consistency, often leading to unrealistic distortions, temporal flickering, and identity-shifting artifacts when encountering significant head pose variations. Furthermore, these approaches often lack explicit 3D geometry constraints, limiting their utility in free-viewpoint rendering and complex lighting interaction applications.

**2.2 3D-Based Head Avatar Reconstruction**

To overcome the view-consistency limitations of 2D methods, a paradigm shift toward 3D-aware representations has occurred. Early 3D approaches utilized mesh-based modeling driven by statistical face priors to provide explicit surfaces for rendering [29, 20, 60, 16, 31]. However, the rise of Neural Radiance Fields (NeRF) [42] has enabled more flexible, topology-agnostic head reconstruction. Numerous NeRF-based methods have been proposed to reconstruct personalized avatars from monocular or sparse-view videos by learning a canonical radiance field combined with a deformation field [17, 1, 18, 45, 78, 30, 4, 74, 2]. To facilitate one-shot synthesis and improve cross-identity robustness, researchers have explored learning tri-plane features or conditioned latent spaces to bypass the need for subject-specific training [36, 41, 11, 65, 12, 56, 35]. Despite their significant progress, NeRF-based avatars typically suffer from face rendering speed limitations due to the heavy computational cost of volumetric sampling through large MLPs, making them difficult to deploy in real-time environments. Moreover, many of these methods depend on identity-specific optimization or require thousands of frames for training, which significantly limits their generalization to unseen identities and poses significant privacy and storage concerns.

**2.3 3D Gaussian Splatting for Human Avatars**

Recently, 3D Gaussian Splatting (3DGS) [28] has emerged as a revolutionary representation, offering an optimal balance between rendering efficiency and geometric fidelity. Unlike implicit NeRFs, 3DGS utilizes anisotropic explicit primitives that allow for real-time rasterization via a tile-based approach. Recent research has successfully extended 3DGS to human head modeling by binding Gaussians to underlying meshes or learned deformation fields [52, 48, 32, 75, 24, 71, 64]. Some approaches map Gaussians onto UV coordinates [62, 3] or utilize neural parametric models to handle dynamic expressions and complex topologies [19, 64]. Most recently, generalizable one-shot Gaussian avatars have been proposed to enable immediate animation from a single image by predicting Gaussian attributes directly from pixel features [10, 76, 38, 21, 43, 8]. However, current Gaussian-based methods often lack robust generalization across diverse accessories (e.g., glasses, hats) and complex hairstyles, as they are frequently trained on datasets with limited diversity. Furthermore, many existing Gaussian avatars still require identity-specific video data to achieve high-frequency textural details, failing to bridge the gap toward a truly generalizable, zero-shot digital human system that achieves ultra-realistic results across the entire human population. In contrast to these prior works, our method leverages a hybrid representation that bypasses the redundant computations of volumetric rendering while maintaining superior generalization capabilities. By integrating a multi-scale feature fusion module with an enhanced rigging strategy, we achieve high-fidelity, view-consistent rendering without the need for time-consuming per-identity optimization.

## 3 Method

**3.1 Overview**

We frame head avatar reconstruction as two coupled stages (Fig. 1) over a shared FLAME-mesh-bound 3D Gaussian representation [48]. **Stage 1** (feed-forward): a single network fθ ingests K ∈ {1, 2, 3, 4} source portrait images and emits face-bound 3D Gaussians in one forward pass. **Stage 2** (optional persubject optimization): starting from fθ ’s output for a chosen reference frame, we run 10K iterations of photometric refinement against the target video. Stage 1 is trained in two phases on the same architecture: monocular-temporal pretraining (Phase 1, CelebV-HQ [77]) and multi-view-spatial post-training (Phase 2, NeRSemble [30]); the variable source count K ∈ {1, 2, 3, 4} during training exposes fθ to monocular and multi-view contexts within a single training distribution.

Figure 1: Overview of our two-stage avatar reconstruction pipeline.

**Design summary.** We organize Gaussians via the FLAME UV unwrap: one Gaussian per valid pixel of a 256×256 UV grid (∼ 58K per identity), each rigidly bound to its parent triangle. UValigned FLAME-bound Gaussian organizations have appeared in prior head-avatar work [62]; we treat this layout as fixed infrastructure rather than as a contribution. Our paper-specific contributions are (i) a feed-forward image-to-Gaussian generator over this layout with variable source count K ∼ Uniform{1, 2, 3, 4}, (ii) a monocular-temporal → multi-view-spatial two-phase schedule with an L2-SP anchor against the Phase-1 checkpoint to prevent identity-prior collapse onto the smaller multi-view set, and (iii) a per-subject refinement loop that freezes the binding manifold and the Gaussian count, preserving the one-to-one UV-pixel-to-Gaussian map of Stage 1. Comparison with feed-forward and per-subject head-avatar baselines is deferred to Related Work; quantitative comparisons appear in Experiments.

**3.2 Feed-Forward Reconstruction Stage**

**Architecture.** A frozen DINOv3 ViT-B/16 backbone with a trainable DPT head [49] produces a dense feature map and a global CLS token; the dense feature is barycentrically warped into UV space and passed to a StyleUNet UV generator [58] (StyleGAN2-based [27], with Spatial-Feature- Transform layers modulated by the source CLS). The 32-channel UV output is concatenated with a 27-d harmonic encoding γ(**d**) of the target camera direction **d** and decoded by five parallel heads (local position, rotation, scale, color, opacity). For K > 1 sources we mean-pool the per-source UV features and CLS tokens (parameter-free, identity at K=1, order-invariant; attention-pool baseline in Tab. 5). FLAME parameters are recovered with the GAGAvatar tracker [10]; full architecture, tracker stack, harmonic embedding form, and barycentric warp are in App. A.1.

**FLAME-conditioned residual head.** A FiLM [47]-modulated residual head adds per-attribute corrections to position, rotation, scale, and color (opacity excluded) using a 112-d conditioning vector formed from the target frame’s expression, pose, and eye codes. Shape β is intentionally excluded: it feeds mesh geometry directly through FLAME’s blendshape basis and is the load-bearing axis of cross-subject generalization. The FiLM and residual layers are zero-initialized so the residual contributes nothing at start. Conditioning-vector breakdown, the rationale for excluding shape and view-direction from the residual head, and ablation handles are in App. A.1.

**Mesh binding.** We adopt the rigid mesh-binding of [48] (§3.2). For each Gaussian we construct the local frame (**c**f , **R**f , sf ) of its bound triangle on the target FLAME mesh (full construction in

App. A.1); the world-space attributes of Gaussian g bound to triangle f (g) are

**x**world g = **R**f (g) **x**local g · sf (g) + **c**f (g), **q**world g = **q**f (g) ⊗ **q**local g , sworld g = slocal g · sf (g), (1)

where ⊗ denotes Hamilton-product quaternion multiplication. We modify the binding by indexing Gaussians via valid UV pixels rather than per-triangle (eliminating split/clone densification) and by freezing the binding during the per-subject refinement (§3.3). Rendering deforms the mesh to the target FLAME pose, transforms each Gaussian via Eq. 1, and rasterizes through a differentiable 3DGS splatter [28].

**Training objective.** We supervise the rendered image ˆ **I** against the target **I** with

L = λ1L1 + λmsLms + λlpLlp | {z } photometric

+ λboxLbox | {z } face-region

+ λpsLps + λssLss | {z } surface smoothness

+ λjacLjac + λδ Lδ | {z } architectural

. (2)

The photometric trio is pixel-wise L1, 1 − MS-SSIM [61], and LPIPS [72]; Lbox is an additional L1 on the face bounding box; Lps and Lss are UV-domain TV penalties on local position and log-scale; Lδ is an output-space L1 shrinkage on the residual head’s deltas (complementing parameter-space weight decay); Ljac enforces view-direction invariance on geometric heads (next paragraph). Explicit forms, the data-difficulty source-frame sampler, and final λ∗ values are in App. A.2.

**View-direction invariance via Jacobian penalty.** The harmonic view-direction input lets color and opacity heads model view-dependent appearance, but would also permit the decoder to leak view direction into geometric attributes. We enforce per-pixel view-direction invariance on the local-frame, pre-projection position and scale predictions via a Hutchinson-randomized Frobenius Jacobian penalty: with i.i.d. Rademacher **v**p ∈ {−1, +1}dout and g(γ) =

P p⟨**v**p, fgeo,p(γ)⟩, one has E**v**[∥∇γ g∥2 2] =

P p ∥Jp∥

2 F with Jp = ∂fgeo,p/∂γ(**d**) (proof in App. A.3). The penalty is

Ljac = log

 1 + ∥∇γ g**x**∥2 2 + ∥∇γ gs∥2 2

 , (3)

costing one backward pass against γ(**d**) per iteration. The estimator follows the single-backward Jacobian-regularization scheme of [22] adapted to the UV-pixel setting. The penalty does not eliminate rasterizer-induced perspective foreshortening, nor does it address pose-conditioning leakage through the residual head (head pose and source camera direction are coupled in monocular tracker outputs). Variance and bias analysis, head-selection rationale, and the rationale for the log(1 + ·) wrapping are in App. A.3.

**Two-phase training. Phase 1 (monocular-temporal):** we pretrain fθ on CelebV-HQ; from each clip we sample K ∼ Uniform{1, 2, 3, 4} frames as sources and a held-out frame as target. Phase 1 covers a wide identity manifold but provides no ground-truth multi-view geometry. **Phase 2 (multi-** **view-spatial):** we continue from the Phase-1 checkpoint on NeRSemble with no architectural change; each batch samples K +1 cameras from the synchronized 16-camera capture, mixing cross-camera and cross-time draws. Phase 2 uses encoder layer-wise LR decay and an L2-SP anchor [37] against the Phase-1 weights to prevent identity-prior collapse onto the smaller multi-view set. Optimizer schedule, sampler details, the cross-camera/cross-time mixing ratio, and the L2-SP weight are in App. A.2.

**3.3 Optimization-Based Reconstruction Stage**

**Per-subject refinement.** The feed-forward stage cannot recover subject-specific high-frequency content beyond the FLAME mesh’s parametric range. We therefore run a 10K-iter per-subject photometric refinement starting from fθ ’s output at a chosen reference frame, with the FLAME mesh and triangle bindings frozen (preserving the one-to-one UV-pixel-to-Gaussian map of Stage 1). At each iteration, we sample a random target frame, deform the FLAME mesh, transform every Gaussian via Eq. 1, and render. We deliberately omit the adaptive densification of GaussianAvatars [48] so as to preserve the spatial inductive bias of our representation. Reference-frame selection (a jawpose-magnitude heuristic with a noise-threshold fallback) and per-parameter learning rates are in App. A.4.

**Anti-spike regularization.** Without densification, the optimization is susceptible to the well-known 3DGS anisotropy failure [28, 48]: Gaussians elongate along a single axis to overfit individual training views, degrading reconstruction at non-training views. We combine three mechanisms (full forms in App. A.4): a scale-freeze warmup for the first 500 iterations, a hard log-scale clamp applied per iteration after scale unfreezes, and a soft screen-space anti-anisotropy penalty active from iteration 500: Laspect = Eg

" ReLU 1 2 log

λmax

 Σ2D g



λmin

 Σ2D g

 − τ

!# , (4)

where Σ2D g is the projected 2D covariance of Gaussian g at the target view (formed via the standard 3DGS projection of Σworld g to image space; full derivation in App. A.4) and τ is a log-aspect budget. Operating on the screen-space projection aligns the penalty with the perceptual quantity of interest (whether the rendered Gaussian appears as a thin streak) and is well-defined for any per-Gaussian local rotation.

**Background compositing and refinement objective.** The target frames carry a foreground alpha mask. To avoid the asymmetric matte-boundary gradient of single-side compositing, we composite both rendered and target images against a shared per-iteration random RGB background **b** ∼ U([0, 1]3) before computing the loss; the resulting symmetric boundary gradient pulls the splatting alpha toward the matte without an explicit silhouette loss (full derivation in App. A.4). The refinement objective is Lref = λ1 L1 + λssim (1 − SSIM) + ⊮[t ≥ 500] · λaspect Laspect. (5) We omit LPIPS at this iteration budget; rationale and per-parameter learning rates are in App. A.4.

## 4 Experiments

**4.1 Datasets and Evaluation Metrics**

**Datasets.** We train on CelebV-HQ [77] (Phase 1) and NeRSemble v2 [30] (Phase 2). For feedforward zero-shot evaluation we use VFHQ [63] with its default test split, and HDTF [73], following the test split used in [11, 10], including 19 video clips. For per-subject evaluation we use the SplattingAvatar [52] dataset, with the last 350 frames of each video reserved as the test set. All ablation tables (main and appendix) are evaluated on a held-out CelebV-HQ [77] test slice. VFHQ and HDTF do not appear in any supervised stage of our training, so they constitute a strict cross-domain zero-shot probe rather than an in-domain benchmark.

**Evaluation metrics.** Standard image-quality metrics PSNR ↑, SSIM ↑, and LPIPS ↓ [72] apply to both lanes. For the feed-forward lane we additionally report ArcFace identity similarity (CSIM ↑), average expression distance (AED ↓), average pose distance (APD ↓), and average keypoint distance (AKD ↓), all computed post-hoc on existing renders. We evaluate two reenactment settings: selfreenactment sets the source to the first frame and the remaining frames as driver/target; crossreenactment uses a different-identity driver, has no pixel ground truth, and is evaluated only by CSIM, AED, and APD. Refinement iteration count and per-subject wall-clock are reported in Tab. 3.

**4.2 Implementation Details**

For Phase 1 we use CelebV-HQ for to train for 200K iterations, and the encoder is initialized from DINOv3-B pretrain weight which is frozen during training. Phase 2 continues from the Phase-1 checkpoint on NeRSemble v2 multiview data and a 25% NeRSemble cross-time mix share. The persubject refinement runs the 10K-iter training on given videos. All FLAME parameters are recovered with the EMICA-based tracker [10]. We use Adamw for both Phase 1 and Phase 2 at a base learning rate of 1×10−4. Training runs on a single H100 NVL GPU with an AMD Ryzen Threadripper 3995X host for 14 days.

**4.3 Main Results**

**Feed-forward comparison (Tabs. 1–2).** We compare the feed-forward stage of our pipeline against eleven generalizable head-avatar baselines on VFHQ and HDTF. VFHQ and HDTF are out-ofdistribution for our model (Phase 1 trains on CelebV-HQ [77], Phase 2 on NeRSemble v2 [30]), while

Table 1: Feed-forward zero-shot reenactment on VFHQ. “Self” is one-shot self-reenactment; “Cross” uses a different-identity driver and reports only identity- and expression-similarity metrics. Best per-column in **bold**.

Self-reenactment Cross-reenactment Method PSNR ↑ SSIM ↑ LPIPS ↓ CSIM ↑ AED ↓ APD ↓ AKD ↓ CSIM ↑ AED ↓ APD ↓

| StyleHeat [67] | 19.95 | 0.726 | 0.211 | 0.537 | 0.199 | 0.385 | 7.659 | 0.407 | 0.279 | 0.551 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ROME [29] | 19.96 | 0.786 | 0.192 | 0.701 | 0.138 | 0.186 | 4.986 | 0.530 | 0.259 | 0.277 |
| OTAvatar [41] | 17.65 | 0.563 | 0.294 | 0.465 | 0.234 | 0.545 | 18.19 | 0.364 | 0.324 | 0.678 |
| HideNeRF [35] | 19.79 | 0.768 | 0.180 | 0.787 | 0.143 | 0.361 | 7.254 | 0.514 | 0.277 | 0.527 |
| CVTHead [39] | 18.43 | 0.706 | 0.317 | 0.504 | 0.186 | 0.224 | 5.678 | 0.374 | 0.261 | 0.311 |
| GPAvatar [11] | 21.04 | 0.807 | 0.150 | 0.772 | 0.132 | 0.189 | 4.226 | 0.564 | 0.255 | 0.328 |
| Real3D-Portrait [66] | 20.88 | 0.780 | 0.154 | 0.801 | 0.150 | 0.268 | 5.971 | 0.663 | 0.296 | 0.411 |
| Portrait4D [12] | 20.35 | 0.741 | 0.191 | 0.765 | 0.144 | 0.205 | 4.854 | 0.596 | 0.286 | 0.258 |
| Portrait4D-v2 [13] | 21.34 | 0.791 | 0.144 | 0.803 | 0.117 | 0.187 | 3.749 | 0.656 | 0.268 | 0.273 |
| GAGAvatar [10] | 21.83 | 0.818 | 0.122 | 0.816 | 0.111 | 0.135 | 3.349 | 0.633 | 0.253 | 0.247 |
| Ours-FF | 23.34 | 0.875 | 0.077 | 0.916 | 0.091 | 0.061 | 2.181 | 0.675 | 0.245 | 0.210 |

Table 2: Feed-forward zero-shot reenactment on HDTF. Same column structure as Tab. 1. Best per-column in **bold**.

Self-reenactment Cross-reenactment Method PSNR ↑ SSIM ↑ LPIPS ↓ CSIM ↑ AED ↓ APD ↓ AKD ↓ CSIM ↑ AED ↓ APD ↓

| StyleHeat [67] | 21.41 | 0.785 | 0.155 | 0.657 | 0.158 | 0.162 | 4.585 | 0.632 | 0.271 | 0.239 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ROME [29] | 20.51 | 0.803 | 0.145 | 0.738 | 0.133 | 0.123 | 4.763 | 0.726 | 0.268 | 0.191 |
| OTAvatar [41] | 20.52 | 0.696 | 0.166 | 0.662 | 0.180 | 0.170 | 8.295 | 0.643 | 0.292 | 0.222 |
| HideNeRF [35] | 21.08 | 0.811 | 0.117 | 0.858 | 0.120 | 0.247 | 5.837 | 0.843 | 0.276 | 0.288 |
| CVTHead [39] | 20.08 | 0.762 | 0.179 | 0.608 | 0.169 | 0.138 | 4.585 | 0.591 | 0.242 | 0.203 |
| GPAvatar [11] | 23.06 | 0.855 | 0.104 | 0.855 | 0.114 | 0.135 | 3.293 | 0.842 | 0.268 | 0.219 |
| Real3D-Portrait [66] | 22.82 | 0.835 | 0.103 | 0.851 | 0.138 | 0.137 | 4.640 | 0.903 | 0.299 | 0.238 |
| Portrait4D [12] | 20.81 | 0.786 | 0.137 | 0.810 | 0.134 | 0.131 | 4.151 | 0.793 | 0.291 | 0.240 |
| Portrait4D-v2 [13] | 22.87 | 0.860 | 0.105 | 0.860 | 0.111 | 0.111 | 3.292 | 0.857 | 0.262 | 0.183 |
| GAGAvatar [10] | 23.13 | 0.863 | 0.103 | 0.862 | 0.110 | 0.111 | 2.985 | 0.851 | 0.231 | 0.181 |
| Ours-FF | 24.67 | 0.916 | 0.076 | 0.871 | 0.103 | 0.078 | 2.326 | 0.905 | 0.224 | 0.147 |

several baselines, notably GAGAvatar [10] and GPAvatar [11], include one or both datasets in their training distribution. Ours-FF leads every column on both splits, surpassing the in-domain leader GAGAvatar by +1.5 dB PSNR on each, with consistent gains across all reported metrics. Qualitative renders are in Fig. 2, and the cross-identity companion in App. B.2. Ours-FF preserves high-frequency identity-discriminative cues such as forehead wrinkles, malar elevation, beard texture, nasolabial folds, and hair-edge silhouette. The detail preservation is consistent across identities and poses.

Table 3: Per-subject training efficiency on a single RTX 3090. Inference FPS at 5122 via the 3- channel diff_gaussian_rasterization backend (58,173 Gaussians/subject). Method Iters ↓ Train ↓ Lat. (ms) ↓ FPS ↑ FlashAvatar [62] 150K ∼1.66 h – 291.20 GaussianAvatars [48] 600K ∼9.25 h – 19.11 GeoAvatar [43] 300K ∼4.90 h – 71.52 **Ours+S3** 10**K** ∼2.0 **min 3.77 265.60**

**Per-subject comparison (Tab. 4).** We compare against per-subject baselines on the SplattingAvatar [52] dataset, with GAGAvatar’s one-shot result included in the leftmost column as a feed-forward upper bound. Ours+S3 leads every reported metric (MSE, PSNR, SSIM, LPIPS), surpassing the prior PSNR/SSIM/MSE leader GeoAvatar [43] by +**1**.**3** dB PSNR at a 30× shorter iteration budget (10K vs 300K) and a substantially shorter wallclock (Tab. 3). Qualitative renders are in App. B.1, Fig. 3.

- (a) Self-reenactment samples from VFHQ

- (b) Self-reenactment samples from HDTF

Figure 2: Feed-forward qualitative comparison on self-reenactment (Tabs. 1–2). The source is set to the first frame and the target to a held-out frame from the same video. (a) Samples from VFHQ; (b) samples from HDTF. We compare against GAGAvatar [10], CVTHead [39], GPAvatar [11], and Portrait4D-v2 [13]; our method is highlighted in yellow.

**Per-subject creation efficiency (Tab. 3).** We compare per-subject training cost on the SplattingAvatar set against the per-subject baselines on a single RTX 3090. Tab. 3 shows our per-subject refinement completes in ∼2 min per subject, two orders of magnitude shorter than FlashAvatar (∼1.66 h), GeoAvatar (∼4.90 h), and GaussianAvatars (∼9.25 h) reported on RTX 3090. Our 10Kiter budget is 15×, 30×, and 60× shorter than the per-subject schedules of FlashAvatar, GeoAvatar, and GaussianAvatars respectively. Inference renders at 265.6 FPS at 5122 on a single RTX 3090, real-time and within the same throughput regime as the per-subject baselines; the feed-forward stage produces a complete avatar from a single image in ∼50 ms on the same RTX 3090.

Table 4: Per-subject monocular reenactment on the SplattingAvatar dataset. The first column reports GAGAvatar’s one-shot result as a feed-forward reference. Best per-row in **bold**.

Metric GAGAv∗ INSTA 3DGS SpA MonoGA FlashAv GA GA0 GeoAv **Ours+S3** MSE× 10−3 ↓ 4.425 1.555 1.761 2.929 1.314 1.173 1.223 1.075 0.545 **0.402** PSNR ↑ 23.541 28.083 27.543 25.333 28.813 29.306 29.124 29.686 32.635 **33.960** SSIM ↑ 0.875 0.938 0.923 0.933 0.937 0.943 0.938 0.934 0.965 **0.971** LPIPS× 10−1 ↓ 1.121 0.678 1.019 0.588 0.733 0.444 0.494 0.529 0.367 **0.355** ∗one-shot feed-forward reference (no per-subject training); SpA = SplattingAvatar, MonoGA = MonoGaussianAvatar, GA = GaussianAvatars, GeoAv = GeoAvatar. GaussianAvatars0 is GaussianAvatars with SH degree set to 0 (per [43] §4.4). The Ours-FF feed-forward column on this benchmark is omitted from the main paper for compactness.

Table 5: Pipeline and architecture ablation on a held-out CelebV-HQ test slice. Top block (rows A–D): cumulative stage-by-stage build-up; row D is our **Full** configuration. Bottom block: subtractive ablations of architectural components, evaluated on top of row D’s configuration.

Experiment PSNR ↑ SSIM ↑ LPIPS ↓ CSIM ↑ AED ↓ APD ↓ AKD ↓

| (A) Phase 1 only | 24.84 | 0.879 | 0.073 | 0.923 | 0.086 | 0.053 | 2.102 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (B) (A) + Phase 2 (no anchor, no cross-time mix) | 22.73 | 0.799 | 0.135 | 0.814 | 0.117 | 0.112 | 3.244 |
| (C) (B) + L2-SP anchor + 25% cross-time mix | 24.94 | 0.881 | 0.070 | 0.935 | 0.084 | 0.054 | 2.032 |
| (D) (C) + per-subject refinement (10K iters, Full) | 29.95 | 0.947 | 0.045 | 0.959 | 0.057 | 0.039 | 1.365 |
| (E) (D) w/o residual (delta) head | 28.36 | 0.933 | 0.060 | 0.931 | 0.081 | 0.041 | 1.412 |
| (F) (D) w/o Jacobian penalty (λjac=0) | 28.72 | 0.937 | 0.040 | 0.952 | 0.061 | 0.042 | 1.874 |
| (G) (D) attention pool (replaces mean-pool) | 29.46 | 0.948 | 0.047 | 0.907 | 0.062 | 0.036 | 1.374 |

**4.4 Ablation Study**

We report the headline pipeline-and-architecture ablation here on self-reenactment over a held-out CelebV-HQ test slice. Remaining ablations are deferred to App. B and App. A.2.

**Pipeline and architecture (Tab. 5).** The top block (rows A–D) builds the pipeline cumulatively, and the bottom block (rows E–G) holds (D) fixed and removes one architectural component at a time. In (A) we report Phase 1 alone, our feed-forward generator trained only on monocular CelebV- HQ. Adding the Phase-2 NeRSemble v2 post-training without controls (B) catastrophically forgets the CelebV-HQ distribution. Reintroducing the L2-SP anchor at λ=10−3 together with the 25% NeRSemble cross-time mix share (C) recovers in-domain quality at near-equal NeRSemble quality, indicating that the two controls are jointly sufficient. Adding the per-subject refinement on top (D) yields the largest single jump and we adopt (D) as our **Full** configuration throughout the paper. (E) removes the FLAME-conditioned residual head and regresses on all metrics, identifying it as broadly load-bearing. (F) removes the Hutchinson Jacobian penalty, which constrains the position and scale heads to be view-invariant: geometry should be an intrinsic property of the scene rather than depend on viewing direction. Without this constraint, AKD rises substantially (1.87 vs 1.37). (G) replaces K-source mean-pooling with attention-pooling, winning SSIM (tied) and APD by small margins but losing on other metrics. The three failure modes target distinct axes of the pipeline and support that the components are not redundant.

## 5 Conclusion

We presented **SpatialAvatar-0**, a unified end-to-end pipeline that bridges generalizable feed-forward and per-subject 3DGS head-avatar regimes through a shared FLAME-mesh-bound Gaussian representation. The K-source-variable feed-forward generator combined with a monocular → multi-view two-phase schedule produces a strong cross-domain identity prior that surpasses the in-domain leader by +1.5 dB PSNR on zero-shot VFHQ/HDTF benchmarks; the 10K-iter layout-preserving per-subject refinement, enabled by an anti-spike regularization replacing adaptive densification, surpasses the 300K-iter GeoAvatar leaderboard on every reported metric by +1.3 dB PSNR at 60× shorter per-subject wall-clock. Together these designs demonstrate that the two avatar regimes need not be developed in isolation: a coherent shared Gaussian representation, paired with appropriate cross-domain training and layout-preserving refinement, can deliver feed-forward generalization and per-subject fidelity within a single ∼2-minute pipeline.

## References

[1] ShahRukh Athar. Rignerf: Fully controllable neural 3d portraits. In Computer Vision and Pattern Recognition, pages 20364–20373, 2022. [2] ShahRukh Athar, Zhixin Shu, and Dimitris Samaras. Flame-in-nerf: Neural control of radiance fields for free view face animation. In IEEE International Conference on Automatic Face & Gesture Recognition, pages 1–8. IEEE, 2021. [3] Kelian Baert, Mae Younes, Francois Bourel, Marc Christie, and Adnane Boukhayma. Gtavatar: Bridging gaussian splatting and texture mapping for relightable and editable gaussian avatars. In Computer graphics forum (Print), page e70351. Wiley Online Library, 2025. [4] Ziqian Bai, Feitong Tan, Zeng Huang, Kripasindhu Sarkar, Danhang Tang, Di Qiu, Abhimitra Meka, Ruofei Du, Mingsong Dou, S. Orts-Escolano, et al. Learning personalized high quality volumetric head avatars from monocular rgb videos. In Computer Vision and Pattern Recognition, pages 16890–16900. IEEE, 2023. [5] Christopher M. Bishop. Training with noise is equivalent to Tikhonov regularization. Neural Computation, 7(1):108–116, 1995. [6] Volker Blanz and Thomas Vetter. A morphable model for the synthesis of 3d faces. In International Conference on Computer Graphics and Interactive Techniques, pages 187–194. ACM Press, 1999. [7] Egor Burkov, Igor Pasechnik, Artur Grigorev, and Victor Lempitsky. Neural head reenactment with latent pose descriptors. In Computer Vision and Pattern Recognition, pages 13783–13792. IEEE, 2020. [8] Hongrui Cai, Yuting Xiao, Xuan Wang, Jiafei Li, Yudong Guo, Yanbo Fan, Shenghua Gao, and Juyong Zhang. Hybrid explicit representation for ultra-realistic head avatars. arXiv preprint arXiv:2403.11453, 2024. [9] Lele Chen, Guofeng Cui, Celong Liu, Zhong Li, Ziyi Kou, Yi Xu, and Chenliang Xu. Talkinghead generation with rhythmic head motion. In European conference on computer vision, pages 35–51. Springer, Springer International Publishing, 2020. [10] Xuangeng Chu and Tatsuya Harada. Generalizable and animatable gaussian head avatar. Neural Information Processing Systems, 37:57642–57670, 2024. [11] Xuangeng Chu, Yu Li, Ailing Zeng, Tianyu Yang, Lijian Lin, Yunfei Liu, and Tatsuya Harada. Gpavatar: Generalizable and precise head avatar from image (s). International Conference on Learning Representations, 2024. [12] Yu Deng, Duomin Wang, Xiaohang Ren, Xingyu Chen, and Baoyuan Wang. Portrait4d: Learning one-shot 4d head avatar synthesis using synthetic data. In Computer Vision and Pattern Recognition, pages 7119–7130, 2023. [13] Yu Deng, Duomin Wang, and Baoyuan Wang. Portrait4D-v2: Pseudo multi-view data creates better 4D head synthesizer. In European Conference on Computer Vision, pages 316–333, 2024. [14] Nikita Drobyshev, Jenya Chelishev, Taras Khakhulin, Aleksei Ivakhnenko, Victor Lempitsky, and Egor Zakharov. Megaportraits: One-shot megapixel neural head avatars. In ACM Multimedia, pages 2663–2671. ACM, 2022. [15] Harris Drucker and Yann LeCun. Improving generalization performance using double backpropagation. IEEE Trans. Neural Networks, 3(6):991–997, 1992. [16] Yao Feng, Haiwen Feng, Michael J Black, and Timo Bolkart. Learning an animatable detailed 3d face model from in-the-wild images. ACM Transactions on Graphics, 40(4):1–13, 2020. [17] G. Gafni, Justus Thies, Michael Zollhofer, and M. Nießner. Dynamic neural radiance fields for monocular 4d facial avatar reconstruction. In Computer Vision and Pattern Recognition, pages 8649–8658, 2020.

[18] Xuan Gao, Chenglai Zhong, Jun Xiang, Yang Hong, Yudong Guo, and Juyong Zhang. Reconstructing personalized semantic facial nerf models from monocular video. ACM Transactions on Graphics, 41(6):1–12, 2022.

[19] Simon Giebenhain, Tobias Kirschstein, Martin Rünz, Lourdes Agapito, and Matthias Nießner. Npga: Neural parametric gaussian avatars. In ACM SIGGRAPH Conference and Exhibition on Computer Graphics and Interactive Techniques in Asia, pages 1–11. ACM, 2024.

[20] Philip-William Grassal, Malte Prinzler, Titus Leistner, C. Rother, M. Nießner, and Justus Thies. Neural head avatars from monocular rgb videos. In Computer Vision and Pattern Recognition, pages 18653–18664, 2021.

[21] Chen Guo, Zhuo Su, Jian Wang, Shuang Li, Xu Chang, Zhaohu Li, Yang Zhao, Guidong Wang, and Ruqi Huang. Sega: Drivable 3d gaussian head avatar from a single image. arXiv.org, 2025.

[22] Judy Hoffman, Daniel A. Roberts, and Sho Yaida. Robust learning with Jacobian regularization. arXiv.org, 2019.

[23] Fa-Ting Hong, Longhao Zhang, Li Shen, and Dan Xu. Depth-aware generative adversarial network for talking head video generation. In Computer Vision and Pattern Recognition, pages 3387–3396. IEEE, 2022.

[24] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In Computer Vision and Pattern Recognition, pages 634–644, 2023.

[25] Michael F. Hutchinson. A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. Communications in Statistics - Simulation and Computation, 18(3):1059– 1076, 1989.

[26] Xinya Ji, Hang Zhou, Kaisiyuan Wang, Wayne Wu, Chen Change Loy, Xun Cao, and Feng Xu. Audio-driven emotional video portraits. In Computer Vision and Pattern Recognition, pages 14075–14084. IEEE, 2021.

[27] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Computer Vision and Pattern Recognition, pages 8110–8119, 2019.

[28] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and G. Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.

[29] Taras Khakhulin, Vanessa Sklyarova, Victor Lempitsky, and Egor Zakharov. Realistic oneshot mesh-based head avatars. In European Conference on Computer Vision, pages 345–362. Springer, Springer Nature Switzerland, 2022.

[30] Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim Walter, and M. Nießner. Nersemble: Multi-view radiance field reconstruction of human heads. ACM Transactions on Graphics, 42(4):1–14, 2023.

[31] Alexandros Lattas, Stylianos Moschoglou, Stylianos Ploumpis, Baris Gecer, Jiankang Deng, and Stefanos Zafeiriou. Fitme: Deep photorealistic 3d morphable model avatars. In Computer Vision and Pattern Recognition, pages 8629–8640. IEEE, 2023.

[32] Jaeseong Lee, Taewoong Kang, Marcel C. Bühler, Minjeong Kim, Sungwon Hwang, Junha Hyung, Hyojin Jang, and J. Choo. Surfhead: Affine rig blending for geometrically accurate 2d gaussian surfel head avatars. International Conference on Learning Representations, 2024.

[33] Junxuan Li, Rawal Khirodkar, Chengan He, Zhongshi Jiang, Giljoo Nam, Lingchen Yang, Jihyun Lee, Egor Zakharov, Zhaoen Su, R. Abdrashitov, et al. Large-scale codec avatars: The unreasonable effectiveness of large-scale avatar pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[34] Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Transactions on Graphics, 36(6):1–17, 2017. [35] Weichuang Li, Longhao Zhang, Dong Wang, Bin Zhao, Zhigang Wang, Mulin Chen, Bang Zhang, Zhongjian Wang, Liefeng Bo, and Xuelong Li. One-shot high-fidelity talking-head synthesis with deformable neural radiance field. In Computer Vision and Pattern Recognition, pages 17969–17978. IEEE, 2023. [36] Xueting Li, Shalini De Mello, Sifei Liu, Koki Nagano, Umar Iqbal, and Jan Kautz. Generalizable one-shot 3d neural head avatar. Neural Information Processing Systems, 36:47239–47250, 2023. [37] Xuhong Li, Yves Grandvalet, and Franck Davoine. Explicit inductive bias for transfer learning with convolutional networks. In International Conference on Machine Learning, pages 2825– 2834. PMLR, 2018. [38] Weijie Lyu, Yi Zhou, Ming-Hsuan Yang, and Zhixin Shu. Facelift: Learning generalizable single image 3d face reconstruction from synthetic heads. In IEEE International Conference on Computer Vision, pages 12691–12701, 2024. [39] Haoyu Ma, Tong Zhang, Shanlin Sun, Xiangyi Yan, Kun Han, and Xiaohui Xie. CVTHead: One-shot controllable head avatar with vertex-feature transformer. In IEEE Workshop/Winter Conference on Applications of Computer Vision, pages 6131–6141, 2023. [40] Yifeng Ma, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yingya Zhang, and Zhidong Deng. Dreamtalk: When expressive talking head generation meets diffusion probabilistic models. arXiv.org, 2(3), 2023. [41] Zhiyuan Ma, Xiangyu Zhu, Guo-Jun Qi, Zhen Lei, and Lei Zhang. Otavatar: One-shot talking face avatar with controllable tri-plane rendering. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16901–16910. IEEE, 2023. [42] B. Mildenhall, Google Research, P. Srinivasan, Matthew Tancik, Jonathan T. Barron, and Ravi Ramamoorthi. Nerf: Representing scenes as neural radiance fields for view synthesis. Lecture Notes in Computer Science, 65(1):405–421, 2020. [43] SeungJun Moon, Hah Min Lew, Seungeun Lee, Ji-Su Kang, and Gyeong-Moon Park. Geoavatar: Adaptive geometrical gaussian splatting for 3d head avatar. In IEEE International Conference on Computer Vision, pages 12811–12821. IEEE, 2025. [44] T. Müller, Alex Evans, Christoph Schied, and A. Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics, 41(4):1–15, 2022. [45] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 5845–5854. IEEE, 2021. [46] Pascal Paysan, Reinhard Knothe, Brian Amberg, Sami Romdhani, and Thomas Vetter. A 3d face model for pose and illumination invariant face recognition. In 2009 sixth IEEE international conference on advanced video and signal based surveillance, pages 296–301. Ieee, IEEE, 2009. [47] Ethan Perez, Florian Strub, H. D. Vries, Vincent Dumoulin, and Aaron C. Courville. FiLM: Visual reasoning with a general conditioning layer. In AAAI Conference on Artificial Intelligence, pages 3942–3951, 2017. [48] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, and

- M) Nießner. Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In Computer Vision and Pattern Recognition, pages 20299–20309, 2023.

[49] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In IEEE International Conference on Computer Vision, pages 12159–12168. IEEE, 2021. [50] Yurui Ren, Ge Li, Yuanqi Chen, Thomas H Li, and Shan Liu. Pirenderer: Controllable portrait image generation via semantic neural rendering. In IEEE International Conference on Computer Vision, pages 13739–13748. IEEE, 2021.

[51] Soubhik Sanyal, Timo Bolkart, Haiwen Feng, and Michael J Black. Learning to regress 3d face shape and expression from an image without 3d supervision. In Computer Vision and Pattern Recognition, pages 7755–7764. IEEE, 2019.

[52] Zhijing Shao, Zhaolong Wang, Zhuang Li, Duotun Wang, Xiangru Lin, Yu Zhang, Mingming Fan, and Zeyu Wang. Splattingavatar: Realistic real-time human avatars with mesh-embedded gaussian splatting. In Computer Vision and Pattern Recognition, pages 1606–1616. IEEE, 2024.

[53] Aliaksandr Siarohin, Stéphane Lathuilière, S. Tulyakov, E. Ricci, and N. Sebe. First order motion model for image animation. Neural Information Processing Systems, 32, 2020.

[54] Aliaksandr Siarohin, Willi Menapace, Ivan Skorokhodov, Kyle Olszewski, Jian Ren, Hsin-Ying Lee, Menglei Chai, and Sergey Tulyakov. Unsupervised volumetric animation. In Computer Vision and Pattern Recognition, pages 458–469. IEEE, 2023.

[55] Jingxiang Sun, Xuan Wang, Lizhen Wang, Xiaoyu Li, Yong Zhang, Hongwen Zhang, and Yebin Liu. Next3d: Generative neural texture rasterization for 3d-aware head avatars. In Computer Vision and Pattern Recognition, pages 20991–21002, 2022.

[56] Felix Taubner, Ruihang Zhang, Mathieu Tuli, and David B Lindell. Cap4d: Creating animatable 4d portrait avatars with morphable multi-view diffusion models. In Computer Vision and Pattern Recognition, pages 5318–5330. IEEE Computer Society, 2024.

[57] Duomin Wang, Yu Deng, Zixin Yin, Heung-Yeung Shum, and Baoyuan Wang. Progressive disentangled representation learning for fine-grained controllable talking head synthesis. In Computer Vision and Pattern Recognition, pages 17979–17989, 2022.

[58] Lizhen Wang, Xiaochen Zhao, Jingxiang Sun, Yuxiang Zhang, Hongwen Zhang, Tao Yu, and Yebin Liu. StyleAvatar: Real-time photo-realistic portrait avatar from a single video. In International Conference on Computer Graphics and Interactive Techniques, pages 1–10. ACM, 2023.

[59] Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. One-shot free-view neural talking-head synthesis for video conferencing. In Computer Vision and Pattern Recognition, pages 10039– 10049, 2020.

[60] Xueying Wang, Yudong Guo, Zhongqi Yang, and Juyong Zhang. Prior-guided multi-view 3d head reconstruction. IEEE Transactions on Multimedia, 24:4028–4040, 2021.

[61] Zhou Wang, Eero P Simoncelli, and Alan C Bovik. Multiscale structural similarity for image quality assessment. In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003, volume 2, pages 1398–1402. IEEE, IEEE, 2003.

[62] Jun Xiang, Xuan Gao, Yudong Guo, and Juyong Zhang. Flashavatar: High-fidelity head avatar with efficient gaussian embedding. In Computer Vision and Pattern Recognition, pages 1802–1812, 2023.

[63] Liangbin Xie, Xintao Wang, Honglun Zhang, Chao Dong, and Ying Shan. VFHQ: A highquality dataset and benchmark for video face super-resolution. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 656–665. IEEE, 2022.

[64] Yuelang Xu, Benwang Chen, Zhe Li, Hongwen Zhang, Lizhen Wang, Zerong Zheng, and Yebin Liu. Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians. In Computer Vision and Pattern Recognition, pages 1931–1941. IEEE, 2024.

[65] Zhenhui Ye, Ziyue Jiang, Yi Ren, Jinglin Liu, Jinzheng He, and Zhou Zhao. Geneface: Generalized and high-fidelity audio-driven 3d talking face synthesis. International Conference on Learning Representations, 2023.

[66] Zhenhui Ye, Tianyun Zhong, Yi Ren, Jiaqi Yang, Weichuang Li, Jiawei Huang, Ziyue Jiang, Jinzheng He, Rongjie Huang, Jinglin Liu, et al. Real3D-Portrait: One-shot realistic 3D talking portrait synthesis. In International Conference on Learning Representations, 2024.

[67] Fei Yin, Yong Zhang, Xiaodong Cun, Mingdeng Cao, Yanbo Fan, Xuan Wang, Qingyan Bai, Baoyuan Wu, Jue Wang, and Yujiu Yang. Styleheat: One-shot high-resolution editable talking face generation via pre-trained stylegan. In European Conference on Computer Vision, pages 85–101. Springer, Springer Nature Switzerland, 2022. [68] Yu Yin, Kamran Ghasedi, HsiangTao Wu, Jiaolong Yang, Xin Tong, and Yun Fu. Nerfinvertor: High fidelity nerf-gan inversion for single-shot real image animation. In Computer Vision and Pattern Recognition, pages 8539–8548, 2022. [69] Egor Zakharov, Aliaksandra Shysheya, Egor Burkov, and Victor Lempitsky. Few-shot adversarial learning of realistic neural talking head models. In IEEE International Conference on Computer Vision, pages 9458–9467. IEEE, 2019. [70] Bowen Zhang, Chenyang Qi, Pan Zhang, Bo Zhang, HsiangTao Wu, Dong Chen, Qifeng Chen, Yong Wang, and Fang Wen. Metaportrait: Identity-preserving talking head generation with fast personalized adaptation. In Computer Vision and Pattern Recognition, pages 22096–22105, 2022. [71] Dongbin Zhang, Yunfei Liu, Lijian Lin, Ye Zhu, Kangjie Chen, Minghan Qin, Yu Li, and Haoqian Wang. Hravatar: High-quality and relightable gaussian head avatar. In Computer Vision and Pattern Recognition, pages 26285–26296. IEEE, 2025. [72] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595. IEEE, 2018. [73] Zhimeng Zhang, Lincheng Li, and Yu Ding. Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. In Computer Vision and Pattern Recognition, pages 3660–3669. IEEE, 2021. [74] Xiaochen Zhao, Lizhen Wang, Jingxiang Sun, Hongwen Zhang, Jinli Suo, and Yebin Liu. Havatar: High-fidelity head avatar via facial model conditioned neural radiance field. ACM Transactions on Graphics, 43(1):1–16, 2023. [75] Zhongyuan Zhao, Zhenyu Bao, Qing Li, Guoping Qiu, and Kanglin Liu. Psavatar: A point-based shape model for real-time head avatar animation with 3d gaussian splatting. IEEE Transactions on Visualization and Computer Graphics, 2024. [76] Xiaozheng Zheng, Chao Wen, Zhaohu Li, Weiyi Zhang, Zhuo Su, Xu Chang, Yang Zhao, Zheng Lv, Xiaoyuan Zhang, Yongjie Zhang, et al. Headgap: Few-shot 3d head avatar via generalizable gaussian priors. In International Conference on 3D Vision, pages 946–957. IEEE, 2024. [77] Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy. CelebV-HQ: A large-scale video facial attributes dataset. In European Conference on Computer Vision, pages 650–667. Springer Nature Switzerland, 2022. [78] Wojciech Zielonka, Timo Bolkart, and Justus Thies. Instant volumetric head avatars. In Computer Vision and Pattern Recognition, pages 4574–4584, 2022.

## A Additional method details

This appendix collects the derivations, exact functional forms, and training-recipe specifics referenced from Section 3.

**A.1 Feed-forward representation and architecture**

**UV-aligned representation.** The UV-aligned per-pixel representation rests on three properties. First, the UV unwrap is topology-respecting: neighboring UV pixels correspond to neighboring mesh triangles, so a 2D convolutional generator inherits the inductive bias for face-bound primitives, and smoothness in the UV domain corresponds directly to smoothness on the mesh surface. Second, the Gaussian count is determined by the UV grid topology and is independent of the input image content; alternative representations must instead supply this information through explicit prediction or densification heuristics. Third, the UV alignment yields a spatially-local mapping from image features to per-Gaussian attributes: each UV pixel governs a single Gaussian, and no cross-pixel attention or order-agnostic pooling is required. The architecture and conditioning interface are identical across the two training phases; only the data distribution differs.

**Barycentric feature warp.** The UV feature map is constructed by a parameter-free barycentric warp from the encoder output. For each UV pixel, we identify its FLAME triangle from a precomputed map, interpolate the corresponding 3D position from the source-frame mesh vertices using the triangle’s barycentric weights, project the 3D point through the source camera, and bilinearly sample the encoder feature and the source RGB at the projected location.

**Harmonic view-direction encoding.** The target camera direction **d** is the unit-normalized third column of the target view’s world-to-camera matrix, encoded as the 27-dimensional harmonic embedding [42] γ(**d**) =

 **d**, sin(2kπ**d**), cos(2kπ**d**)

 k=0..3 ∈ R

27. (6)

**FLAME version, tracker, and UV layout.** Source and target FLAME parameters are produced by the GAGAvatar tracker stack [10] (xg-chu/GAGAvatar_track; an EMICA-style monocular tracker built on the EMOCA / SMIRK expression-aware lineage, combined with a VGGHead detector, StyleMatte matting, and a 300-step Adam OptimEngine). Per frame, the tracker emits β ∈ R300 (shape), ψ ∈ R100 (EXPCODE, FLAME 2020 expression blendshape coefficients), and a tracker-specific subset of FLAME pose: θpose ∈ R6 (POSECODE, = global head rotation ⊕ jaw axis-angle) and θeye ∈ R6 (EYECODE, bilateral eye pose). The standard FLAME pose vector [34] is θ ∈ R3K+3 = R15 for K = 4 joints (neck, jaw, two eyeballs) plus global rotation; the EMICA tracker absorbs the neck joint into the camera transform and exposes the eye pose separately, yielding the 6+6 subset above. The tracker also emits a 3×4 camera transform and a face bounding box. The mesh is FLAME 2020 [34] extended via patch_teeth.py to 5143 vertices (10144 faces), unwrapped onto a 256×256 UV grid via flame_uv.npz (5118 UV vertices, 9976 UV faces, no UV coordinates for teeth); the teeth-patched mesh and the UV unwrap are released as data assets of the GAGAvatar tracker [10]. The valid-UV mask Ω ⊂ {1, . . . , 256}2 has |Ω| ≈ 58,173 pixels, fixed across all subjects and frames; this is the count of Gaussians per identity.

**Residual head conditioning vector.** The residual head consumes a 112-dimensional conditioning vector formed by concatenating the target frame’s expression, pose, and eye codes: **c**t = ψt ⊕ θpose t ⊕ θeye t ∈ R100+6+6 = R112. (7) A 2-layer MLP produces per-attribute FiLM [47] coefficients (γa, βa) for a ∈ {position, rotation, scale, color}; the modulated features pass through a 1 × 1 convolutional head whose output is added to the post-decoder UV maps for those four attributes. Opacity is excluded (we observed alpha-pumping during animation in early experiments, an empirical observation rather than a derived property). Shape β is intentionally excluded from **c**t: shape is fixed-per-subject, feeds mesh geometry directly through FLAME’s blendshape basis (Eq. 1), and is the load-bearing axis of cross-subject generalization; conditioning the residual on β would let the head learn per-identity displacements and overfit to training subjects rather than identity-agnostic expression-conditioned residuals. Head pose θpose [0:3] is included in **c**t but does not feed the harmonic view encoder γ(**d**), so the residual head is structurally independent of view direction (App. A.3). The residual head’s contribution is evaluated in our ablation study (Tab. 5).

**Per-triangle frame construction.** The per-triangle local frame (**c**f , **R**f , sf ) for a triangle with vertices (**v**0, **v**1, **v**2) is constructed as follows. The center **c**f is the triangle centroid. The rotation **R**f is built from the edge vectors via Gram-Schmidt-like orthonormalization: **a**0 = (**v**1 − **v**0)/∥**v**1 − **v**0∥ along the first edge, **a**1 along the unit triangle normal, and **a**2 = **a**1 × **a**0. The isotropic scale is the average of one edge length and one altitude. This averaging follows the textual description in GaussianAvatars [48] §3.2 (“the mean length of one of the edges and its perpendicular”) and matches the public reference implementation; the resulting scalar inherits the FLAME mesh’s fixed vertex ordering, and the thin-triangle fallback below bounds the residual conditioning:

sf = 1 2 (∥**v**1 − **v**0∥ + |**a**2 · (**v**2 − **v**0)|) . (8)

All **R**f and sf are computed with respect to the fixed FLAME 2020 vertex ordering (5143 vertices via patch_teeth.py); changing the mesh re-permutes vertex indices and invalidates the predicted local quaternions **q**local g , since the residual head’s quaternion correction is interpreted in this fixed local basis.

**Numerical stability on degenerate triangles.** The FLAME UV unwrap contains a small number of near-degenerate triangles at eye corners, lip seams, and ear seams. We rely on safe normalization in the Gram-Schmidt construction (rather than an explicit detection-and-fallback branch) to prevent division-by-zero on near-zero edge or altitude lengths; the per-triangle scale sf tends to ∥**v**1 − **v**0∥/2 in the zero-altitude limit and remains finite. Empirically the residual conditioning has not driven divergent training or visible artifacts.

**A.2 Training procedure**

**Data-difficulty-weighted source-frame sampler.** We bias the source distribution against trivial near-identity source-target pairs by sampling source[0] from a softmax over a weighted combination of camera-direction and expression-code dissimilarity from the target frame. The softmax temperature is annealed monotonically during the early phase of training: at high temperature the softmax is near-uniform and the sampler draws candidates approximately uniformly across the pool, while at low temperature the sampler concentrates on candidates with maximum dissimilarity. The temperature is annealed from τ0 = 2.0 (near-uniform) to τT = 0.5 (concentrated on maximum-dissimilarity candidates) over the first quarter of training via a piecewise-linear schedule; only source[0] is drawn from this softmax, the remaining K −1 sources are uniform.

**Phase 2 specifics.** Each Phase 2 batch samples K +1 cameras from the synchronized 16-camera NeRSemble capture, mixing cross-camera draws (same identity, same instant, K+1 distinct cameras; default 75% of batches) with cross-time draws (same identity, K +1 distinct time-shifted views from the available cameras, reducing to monocular same-camera-different-time when K = 1; default 25%). The cross-camera majority provides multi-view supervision absent from Phase 1; the cross-time minority preserves the wide-identity prior acquired during Phase 1 and prevents collapse onto the smaller multi-view training set. Phase 2 uses layer-wise learning-rate decay following the LCA Appendix C scheme [33] (we denote this rate γLR = 0.65 to avoid notational collision with γ(**d**)) on the encoder layers, with normalization-layer running statistics continued (not reset, not frozen) across the Phase 1 → Phase 2 boundary, with both decoders held at base LR, supplemented by an L2-SP anchor [37] against the Phase 1 checkpoint as a regularization-to-Phase-1-prior. The motivating concern is identity-prior collapse onto the 17 NeRSemble subjects [30] we train on (a subset of the released 220-subject pool) relative to CelebV-HQ’s wide-identity Phase-1 manifold (∼ 15,653 identities); our ablation study (Tab. 5) reports whether without-anchor exhibits regression on CelebV-HQ-held-out subjects. The cross-cam-mix ratio, LR-decay coefficient, and L2-SP weight are not claimed to be optimal. Remaining optimizer, batch-size, and hardware details are in the “Optimizer, precision, and batch sizes” paragraph below.

**Loss-form table.** Let Ω ⊂ {1, . . . , 256}2 denote the valid-UV mask (|Ω| ≈ 58,173). The seven currently-undefined loss terms in Eq. 2 have explicit forms:

L1 = HW 1 ˆ **I** − **I** 1, Lms = 1 − MS-SSIMσ∈{0.5,1,2,4,8}(ˆ **I**, **I**), Llp = LPIPS-AlexNet(ˆ **I**, **I**), Lbox = |bbox 1 | ˆ **I**bbox − **I**bbox 1, bbox: EMICA face bbox dilated × 1.65,

Lps = |Ω 1 |

X

(u,v)∈Ω

 ∇u**x**local 1 + ∇v **x**local 1

 ,

Lss = |Ω 1 |

X

(u,v)∈Ω

 ∇uslog 1 + ∇v slog 1

 ,

Lδ = |Ω 1 |

X

a∈{p,r,s,c}

∆Ω a 1 (omits opacity),

Ljac as in body Eq. 3.

The reduction is mean over valid pixels for all UV-domain terms. The TV penalties (Lps, Lss) are defined on the UV grid; at FLAME UV seams (face↔ear, neck) UV-adjacent pixels are not mesh-adjacent, and the TV imposes an unintended smoothing across the seam. The cross-seam contribution to |Ω| is small in our setting. Order-of-magnitude weights are λ1, λms, λlp, λbox ∼ 100, λps, λss ∼ 10−2, λjac ∼ 10−1, and λδ ∼ 10−2 in Phase 2 (the residual-shrinkage term is inactive in Phase 1, where the head trains under wide-identity exposure with weight decay only). The two-phase contribution, the L2-SP anchor, and cross-domain CelebV-HQ behavior are reported in our ablation study (Tab. 5).

**Optimizer, precision, and batch sizes.** Phase 1 uses plain Adam at 2.5×10−4 with FP32 precision, batch size 48 on a single H100 NVL, and gradient clipping at norm 5.0. Phase 2 uses AdamW with cosine warmup, bf16 mixed precision, and batch size 12; the L2-SP weight is λsp=10−3 and the NeRSemble cross-time mix share is 25% (the layer-wise LR decay γLR, anchor mechanism, and the cross-camera majority are described in Phase 2 specifics above). The residual-shrinkage term Lδ is inactive in Phase 1 and activated in Phase 2; the remaining photometric loss weights match the Phase-1 magnitudes (Loss-form table above). The per-subject refinement sets λaspect=0.5 for the screen-space anti-anisotropy penalty active over iterations 500–10,000; per-parameter learning rates and the global cosine schedule are described in App. A.4.

**A.3 Jacobian penalty derivation**

**Goal: per-pixel view-direction invariance.** Let fgeo denote the position and scale heads. We require the per-pixel Jacobian Jp = ∂fgeo,p/∂γ(**d**) ∈ Rdout×27 to vanish for every UV pixel p ∈ Ω. The natural penalty is the per-pixel Frobenius energy

F = |Ω 1 |

X

p∈Ω

∥Jp∥2 F = |Ω 1 |

X

p

X

i,j

Jp,i,j 2 . (9)

A direct evaluation of F requires either |Ω| ≈ 58K backward passes against γ(**d**), or a vmap-jacrev materialization of the |Ω| × dout × 27 per-pixel Jacobian tensor; both inflate training-time autograd cost by one to two orders of magnitude.

**Discrete two-point alternative.** A finite-difference proxy

fgeo(**x**src, **d**) − fgeo(**x**src, **d**ref) 1 (10)

between the actual view direction **d** and a fixed reference **d**ref avoids autograd cost but, at the moderate-to-large angular separations typical of our training videos, confounds first-order viewdirection sensitivity with higher-order curvature; we prefer the differential limit below.

**Hutchinson per-pixel estimator.** Let fgeo,p ∈ Rdout denote the position or scale prediction at UV pixel p, with per-pixel Jacobian Jp = ∂fgeo,p/∂γ(**d**) ∈ Rdout×27 (dout = 3 for both heads). Sample i.i.d. Rademacher **v**p ∈ {−1, +1}dout and define g(γ) =

P p⟨**v**p, fgeo,p(γ)⟩. Then

E**v**

 ∇γ g

2 2

 =

X

p

Jp

2 F . (11)

Proof. ∇γ g =

P p J

⊤ p **v**p; by independence and zero mean, E[**v**p**v**

⊤ q ] = δpq **I**, so E[∥∇γ g∥

2 P 2] = p,q E[**v**

⊤ p JpJ

⊤ q **v**q ] =

P p tr(JpJ

⊤ p ) =

P p ∥Jp∥

2 F . □

**Estimator variance and bias.** The lineage anchor is Hutchinson [25] (Rademacher trace estimator); the single-backward Jacobian-regularization scheme follows Hoffman et al. [22] (whose official implementation uses Gaussian-unit-norm projections, agreeing with Rademacher in expectation up to a dout factor, with comparable but non-identical per-sample variance); per-feature variants are due to Drucker and LeCun [15] and Bishop [5]. We use Rademacher per Hutchinson 1989, with dout = 3 for both position and scale heads applied independently. In the body penalty (Eq. 3), g**x** =

P p⟨**v**

**x** p , **x**

local p ⟩ and gs =

P p⟨**v**

s p, s

local p ⟩ are the per-head Hutchinson scalars, with independently sampled Rademacher projections. **Estimator-target gap.** The Hutchinson identity (Eq. 11) shows E**v**[∥∇γ g∥2 2] =

P p ∥Jp∥

2 F on the inner argument; the wrapped loss is biased relative to log(1 +

P p ∥Jp∥

2 F ) by Jensen’s inequality (strict whenever Var[∥∇γ g∥2] > 0). By Taylor expansion of log(1 + ·) around μ = E[∥∇γ g∥2], the wrapped-loss bias scales as − 1 2 Var[∥∇γ g∥2]/(1 + μ)2 at leading order; the bias decays toward zero as training drives μ toward zero. The penalty therefore targets the per-pixel Frobenius energy via a Hutchinson estimate of the inner argument, with a second-order bias from the log(1 + ·) wrapping. The leading-order Taylor expansion above is per-head; the wrapped loss applies a single log(1 + ·) over the sum of the position and scale contributions, so the joint Jensen bias is governed by Var[∥∇γ g**x**∥2] + Var[∥∇γ gs∥2] rather than the sum of two independently-wrapped per-head log biases. **Per-iteration variance.** The relative variance of the inner argument scales as O(

P p̸=q ∥JpJ

⊤ q ∥

2 F /(

P p ∥Jp∥

2 F )

2. = Θ(1) in |Ω| (the cross-pixel sum is Θ(|Ω|2), the

squared-Frobenius normalizer is also Θ(|Ω|2), leading-order in |Ω|, with diagonal contribution Θ(|Ω|) dominated for |Ω| ≫ 1). The estimator is therefore not asymptotically variance-shrinking with pixel count; the regularizer contributes a controlled-variance inductive bias whose per-step gradient is dominated by the photometric trio at the chosen λjac ∼ 10−1 (consistent with the magnitude in the loss-form table above). The Θ(1) relative-variance scaling above holds under approximate cross-pixel independence; correlated Jp across UV pixels (e.g., overlapping CNN receptive fields) would re-introduce a cross-pixel scaling factor. We verify training stability empirically. **Why Rademacher per-pixel and not unprojected pixel-sum.** Per the Hutchinson identity (Eq. 11), the per-pixel Rademacher projection makes the cross-pixel cancellation regime independent of any sign correlation in {Jp}, regardless of training equilibrium: an unprojected pixel-sum

P p ∇γ fp

2 2 would admit a benign minimizer at sign-cancelled {Jp}; the per-pixel projection removes this failure mode by construction. The penalty matches the autograd cost of an unprojected pixel-sum (one backward pass against γ(**d**)).

**Why we wrap in** log(1 + ·)**.** We wrap the inner argument in log(1 + ·) to cap the regularizer’s per-step magnitude during early training: at random initialization the unwrapped argument is on the order of 102 and would otherwise dominate the photometric signal. At convergence the wrapper behaves as the identity (gradient → 1/(1 + ∥ · ∥2), vanishing as the argument shrinks). This is one option among multiple stability mechanisms (a λ-warmup schedule, gradient clipping, or smaller λ).

**Application to which heads.** We apply Ljac to position and scale heads only; color and opacity remain view-dependent. The residual head’s input vector does not include the harmonic view encoding γ(**d**) (App. A.1), so it is structurally independent of γ(**d**) and is not penalized by Ljac. Rotation is excluded because the local quaternion is constrained to the unit sphere; a Frobenius penalty on the raw 4-d quaternion components does not equal angular-Jacobian energy under the non-trivial Riemannian metric, making a Hutchinson-projection penalty ill-posed for **q**local g .

**A.4 Per-subject optimization details**

**Log-scale hard clamp and screen-space soft penalty.** The hard clamp operates in parameter space: each iteration the log-scale tensor **s**log is clamped to a fixed absolute range [slog min, slog max], then the per-Gaussian deviation from the log-scale mean is clamped so that maxa(**s**log g,a) − mina(**s**log g,a) stays within a fixed log-spread bound across the three axes; this provides a hard upper bound on world-space anisotropy. The soft penalty Laspect operates in screen space. The per-Gaussian world-space covariance Σworld g = Rg world diag

 (**s**world g )2

 (Rg world )⊤ is projected to the camera by Σ2D g = Jg W Σworld g W ⊤Jg⊤ , where W is the world-to-camera rotation and Jg is the projection Jacobian at the Gaussian’s center, following the standard 3DGS projection [28]. The closed-form 2×2 eigenvalues λmax, λmin of Σ2D g define the projected log-aspect 1 2 (log λmax − log λmin), which Laspect penalizes above τ . The two controls operate on different quantities (the hard clamp on parameterspace log-scale, the soft penalty on screen-space projected aspect) and are therefore not redundant: the hard clamp guarantees an absolute world-space bound, while the soft penalty supplies a smooth perceptually-aligned gradient toward isotropic projected footprints.

**Dual-side background compositing.** The rendered image ˆ **I** exits the differentiable 32-channel splatter together with a per-pixel splatting alpha α ˆ(u) = 1 −

Y

i∈N (u)

 1 − αi Gi(u)

 , (12)

where αi is the per-Gaussian opacity, Gi(u) is the projected 2D Gaussian footprint at pixel u, and N (u) is the set of Gaussians overlapping u. This is the standard 3DGS opacity accumulation [28], equivalent to one minus the unoccluded transmittance through N (u); the product is order-independent (front-to-back ordering matters for color compositing but not for the alpha-only product). The matte **m** is produced by the StyleMatte branch of the EMICA tracker (xg-chu/GAGAvatar_track; see App. A.1) as in the feed-forward stage. We use premultiplied alpha compositing on both sides: ˆ **I**comp = **m** ⊙ ˆ **I** + (1 − **m**) ⊙ **b**, **I**comp = **m** ⊙ **I** + (1 − **m**) ⊙ **b**, **b** ∼ U([0, 1]3), (13) with **b** resampled at every iteration. We trace where **b** enters the gradient. (i) Pure-L1 in pixels where α ˆ = **m** = 1. The L1 pixel residual reduces to ∥ˆ **I**comp − **I**comp∥1 = ∥**m** ⊙ (ˆ **I** − **I**)∥1 on the matted target side. On the rendered side, the splatter’s actual output is α ˆˆ **I** + (1 − α ˆ)**b** rather than **m**ˆ **I** + (1 − **m**)**b**; the random **b** cancels exactly only at pixels where α ˆ = **m** = 1 simultaneously. Pixels where α ˆ ̸= **m** are handled by mechanism (iii) below. (ii) SSIM-mediated boundary band. The SSIM term in Lref is computed on a Gaussian window of size w ×w (w = 11 as in the standard SSIM implementation). Because the window crosses the silhouette, the local-window means μˆ **I**comp , μ**I**comp and variances depend on **b** in a band of width ⌊w/2⌋ pixels around the silhouette, letting the randomization take effect at and near the boundary. (iii) α ˆ−**m** disagreement at the boundary. When the splatter’s α ˆ disagrees with the matte **m** (silhouette overspill or partial transparency), the realized rendering at pixel u is C(u) = ˆ α(u)ˆ **I**(u) + (1−α ˆ(u))**b**. Treating **mI** + (1 − **m**)**b** as the supervision target y(u), the L1 pixel loss ∥y − C∥1 has gradient ∂∥y − C∥1/∂**b** = −(1 − α ˆ) sign(y − C), which is non-zero on partially transparent pixels and switches sign with **b**; the same **b** on both sides yields a symmetric (rather than asymmetric) gradient that pulls α ˆ → **m** without an explicit silhouette loss. The dual-side trick of using the same random **b** on both sides, a long-standing differentiable-rasterization technique [42, 44], is here applied with α ˆ−**m** symmetrization in the per-subject Gaussian-refinement context.

**Reference-frame selection.** We initialize per-subject refinement from fθ ’s output at a reference frame r chosen by minimum jaw-pose magnitude, r = arg mint ∥θjaw t ∥1, with ties broken by earliest frame index. If the minimum jaw-pose magnitude across all T frames is below a noise threshold τjaw = 10−3 (silent monologue, occluded jaw, or severe-pose-only video), we fall back to a uniformly random frame from the bottom-decile pool {t : ∥θjaw t ∥1 ≤ quantile10%}. This is a jaw-only heuristic and does not select a fully canonical pose.

Figure 3: **Per-subject monocular qualitative comparison (visualization sample of** 4 **subjects; full** **numerics in Tab. 4).** Each row shows one held-out target frame; columns are GT, GeoAvatar [43], GaussianAvatars [48], SplattingAvatar [52], INSTA [78], FlashAvatar [62], and **Ours+S3** (highlighted).

**Per-parameter learning rates and schedule.** Per-parameter learning rates are tuned individually for each Gaussian attribute family (position, rotation, color, opacity, and scale, the last unfrozen at iteration 500), with a single global cosine decay over the full ten thousand iterations. We omit LPIPS at this iteration budget and restrict it to the feed-forward stage: at 10K steps with the perparameter learning rates above, the LPIPS gradient produces VGG-edge-filter speckle artifacts on the frozen Gaussian set before convergence; the L1+SSIM combination recovers high-frequency content directly via Gaussian color and position.

## B Additional experimental results

**B.1 Per-subject monocular qualitative comparison**

Fig. 3 shows qualitative renders accompanying the per-subject monocular numerics of Tab. 4 on a 4-subject visualization sample. Each row places a held-out target frame next to the six candidate renders (five per-subject baselines and Ours+S3). Ours+S3 follows the target head pose closely with no visible blur or dropped regions, and subject-specific high-frequency content (forehead wrinkles, malar contour, beard or mustache texture, hair-edge silhouette, eye specular highlights) remains sharp at the rendered scale.

**B.2 Cross-identity feed-forward qualitative comparison**

The cross-block metrics (CSIM/AED/APD) of Tabs. 1–2 characterize the cross-identity lane numerically; Fig. 4 visualizes whether high CSIM corresponds to recognizable identity transfer or to a low-frequency average. The cross-identity lane (Fig. 4) places source-driver pairs whose head yaw and expression states differ substantially. The second VFHQ row pairs a profile-view source with a near-frontal driver under a strong open-mouth expression, and analogous yaw-and-expression discrepancies recur in the remaining rows. Ours-FF retargets the source identity to the driver pose with the geometric face structure (eye contour, nose-bridge orientation, jaw outline) remaining consistent with the source identity even under these extreme yaw and expression conditions.

- (a) VFHQ test split (4 stratified-random cross-identity pairs)

- (b) HDTF test split (4 stratified-random cross-identity pairs, GPAvatar split)

Figure 4: **Cross-identity feed-forward qualitative comparison (Tabs. 1–2, Cross block, columns 9–** **11).** Each row pairs an identity-A source (col 1) with an identity-B driver (col 2); the remaining columns are reenactments of identity A under driver B’s expression and pose. **(a)** VFHQ test split, **(b)** HDTF test split. Each panel: 4 stratified-random disjoint-identity pairs × 7 columns (source-A, driver-B, **Ours-FF** (highlighted), GAGAvatar [10], CVTHead [39], GPAvatar [11], and Portrait4Dv2 [13]). Real3D-Portrait and Portrait4D dropped on the same grounds as Fig. 2.

**B.3 Backbone capacity and source-image count**

The top block of Tab. 6 compares the three publicly released DINOv3 variants; the encoder is frozen, so backbone capacity affects feature quality and forward-pass cost but not the trainable-parameter count of the pipeline. The bottom block evaluates the source-image count K ∈ {1, 2, 3, 4} at inference time on the single feed-forward checkpoint.

Table 6: **Model-parameter ablation on CelebV-HQ.** Same self-reenactment column structure as Tab. 5. Top block: backbone-capacity comparison across the three publicly released DINOv3 variants (encoder frozen; bold row marks the default). Bottom block: inference-time source-image count K on the single feed-forward checkpoint. Row K=4 is the largest training-time condition.

Variant PSNR ↑ SSIM ↑ LPIPS ↓ CSIM ↑ AED ↓ APD ↓ AKD ↓

| DINOv3-S | 24.48 | 0.863 | 0.083 | 0.917 | 0.095 | 0.064 | 2.296 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DINOv3-B | 24.94 | 0.881 | 0.070 | 0.935 | 0.084 | 0.054 | 2.032 |
| DINOv3-L | 25.02 | 0.884 | 0.067 | 0.941 | 0.085 | 0.053 | 1.943 |
| K=1 | 23.01 | 0.806 | 0.126 | 0.828 | 0.079 | 0.101 | 3.098 |
| K=2 | 24.21 | 0.856 | 0.090 | 0.908 | 0.092 | 0.071 | 2.396 |
| K=3 | 24.75 | 0.869 | 0.079 | 0.914 | 0.082 | 0.058 | 2.118 |
| K=4 | 24.94 | 0.881 | 0.070 | 0.935 | 0.084 | 0.054 | 2.032 |

Table 7: **Phase 1 and Phase 2 training hyperparameter ablation on CelebV-HQ.** Same selfreenactment column structure as Tab. 5. Each block varies one training-time hyperparameter with the others fixed at the default. Row (a) reproduces Tab. 5 row (C); the other Default rows (e),(h),(k) are independent reruns of the same configuration and differ by small evaluation-order noise.

Variant PSNR ↑ SSIM ↑ LPIPS ↓ CSIM ↑ AED ↓ APD ↓ AKD ↓ Phase 1 LR schedule

| (a) LinearLR 1.0→0.1, 160K+40K (Default) | 24.94 | 0.881 | 0.070 | 0.935 | 0.084 | 0.054 | 2.032 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (b) cosine decay over 200K iters | 24.81 | 0.879 | 0.072 | 0.929 | 0.082 | 0.056 | 2.118 |
| (c) constant LR (2.5×10−4) | 24.51 | 0.871 | 0.077 | 0.922 | 0.091 | 0.061 | 2.207 |

Phase 1 base LR

| (d) 1×10−4 (under) | 24.71 | 0.875 | 0.074 | 0.928 | 0.083 | 0.058 | 2.122 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (e) 2.5×10−4 (Default) | 24.92 | 0.880 | 0.071 | 0.934 | 0.085 | 0.055 | 2.041 |
| (f) 5×10−4 (over) | 24.42 | 0.866 | 0.082 | 0.916 | 0.095 | 0.064 | 2.273 |

Phase 2 encoder layer-wise LR decay γLR

| (g) γLR=0.5 (more frozen) | 24.84 | 0.878 | 0.072 | 0.926 | 0.087 | 0.057 | 2.011 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (h) γLR=0.65 (Default) | 24.96 | 0.882 | 0.069 | 0.937 | 0.083 | 0.054 | 2.024 |
| (i) γLR=0.8 (less decay) | 24.80 | 0.878 | 0.071 | 0.928 | 0.090 | 0.057 | 2.058 |

Jacobian penalty weight λjac

| (j) λjac=0 (no penalty) | 24.30 | 0.876 | 0.067 | 0.929 | 0.092 | 0.063 | 2.301 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (k) λjac=10−1 (Default) | 24.93 | 0.880 | 0.070 | 0.933 | 0.084 | 0.055 | 2.046 |
| (l) λjac=1 (over-regularized) | 24.46 | 0.867 | 0.082 | 0.917 | 0.097 | 0.066 | 2.005 |

The backbone comparison traces the expected monotone improvement from DINOv3-S to DINOv3-L, with DINOv3-B on the knee of the quality–parameter curve. The K-source comparison shows the largest jump from K=1 to K=2 and flattens thereafter.

**B.4 Training hyperparameter ablation**

Tab. 7 probes the sensitivity of the feed-forward checkpoint to four training-time hyperparameters: the Phase 1 learning-rate schedule, the Phase 1 base learning rate, the Phase 2 encoder layer-wise LR decay γLR, and the Jacobian penalty weight λjac. Each block varies one hyperparameter with the remaining hyperparameters held at the default; row (a) reproduces Tab. 5 row (C), and the other Default rows (e),(h),(k) are independent reruns of the same configuration. The Phase 1 LR schedule rows (a)–(c) show the LinearLR default outperforming both cosine (b) and constant (c) by ∼0.1–0.5 dB PSNR, with constant LR the worst. The base-LR rows (d)–(f) form a single peak around 2.5×10−4 where halving or doubling each lose ∼0.2–0.5 dB. The Phase 2 layer-wise decay γLR (g)–(i) is the least sensitive setting on PSNR: deviations either way regress by under 0.2 dB, while the L2-SP anchor and cross-time mix (Tab. 8) carry the Phase 2 cross-domain protection. The λjac rows (j)–(l) form a sharp peak at λjac=10−1: setting it to 0 (j) or 1 (l) each lose ∼0.4–0.7 dB.

Table 8: **L2-SP anchor and cross-time mix ablation on CelebV-HQ.** Same self-reenactment column structure as Tab. 5. Top block (rows a–d): 2×2 factorial over the L2-SP anchor weight λsp and the NeRSemble cross-time mix share. Bottom block (rows e–f): λsp sensitivity at the chosen mix share. Row (d) is the default Phase 2 configuration used elsewhere in the paper. Rows (a) and (d) duplicate Tab. 5 rows (B) and (C).

Variant PSNR ↑ SSIM ↑ LPIPS ↓ CSIM ↑ AED ↓ APD ↓ AKD ↓

| (a) λsp=0, cross-time 0% | 22.73 | 0.799 | 0.135 | 0.814 | 0.117 | 0.112 | 3.244 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (b) λsp=10−3, cross-time 0% | 24.11 | 0.854 | 0.097 | 0.945 | 0.100 | 0.079 | 2.459 |
| (c) λsp=0, cross-time 25% | 23.81 | 0.840 | 0.104 | 0.868 | 0.107 | 0.082 | 2.573 |
| (d) λsp=10−3, cross-time 25% (Default) | 24.94 | 0.881 | 0.070 | 0.935 | 0.084 | 0.054 | 2.032 |
| (e) λsp=10−2, cross-time 25% | 24.37 | 0.862 | 0.089 | 0.909 | 0.077 | 0.069 | 2.302 |
| (f) λsp=10−4, cross-time 25% | 24.52 | 0.869 | 0.077 | 0.916 | 0.090 | 0.063 | 2.283 |

Table 9: **Per-subject anti-spike component ablation on CelebV-HQ.** Same self-reenactment column structure as Tab. 5. Row (a) is the full anti-spike regularization (matches Tab. 5 row (D)); rows (b)–(d) each remove one component, holding the other two fixed.

Variant PSNR ↑ SSIM ↑ LPIPS ↓ CSIM ↑ AED ↓ APD ↓ AKD ↓

| (a) Full anti-spike (scale-freeze + log-clamp + Laspect) | 29.95 | 0.947 | 0.045 | 0.959 | 0.057 | 0.039 | 1.365 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (b) (a) w/o scale-freeze warmup | 27.88 | 0.941 | 0.070 | 0.945 | 0.051 | 0.043 | 1.623 |
| (c) (a) w/o log-scale clamp | 28.39 | 0.938 | 0.043 | 0.947 | 0.063 | 0.038 | 1.644 |
| (d) (a) w/o Laspect (λaspect=0) | 28.92 | 0.930 | 0.051 | 0.954 | 0.062 | 0.035 | 1.579 |

**B.5 L2-SP anchor and cross-time mix ablation**

Tab. 8 probes the two Phase 2 controls introduced in §3.2: the L2-SP anchor weight λsp and the NeRSemble cross-time mix share. Rows (a)–(d) form a 2×2 factorial over λsp ∈ {0, 10−3} and cross-time mix share∈ {0%, 25%}, isolating each control’s individual contribution; rows (e)–(f) vary λsp ∈ {10−2, 10−4} at the chosen mix share. Rows (a) and (d) duplicate Tab. 5 rows (B) and (C) for self-containment. The 2×2 factorial shows that the anchor alone (b) and cross-time mix alone (c) each provide partial cross-domain protection, while the joint configuration (d) recovers the full effect, indicating the two mechanisms are complementary rather than redundant. The λsp rows (e)–(f) show the expected sensitivity: under-anchoring (f) approaches cross-time-mix-only (c), while over-anchoring (e) reduces Phase 2’s ability to absorb multi-view signal. We adopt λsp=10−3 with 25% cross-time mix (row d) as the default.

**B.6 Anti-spike component ablation**

Tab. 9 probes the three components of the per-subject refinement anti-spike regularization described in §3.3: the scale-freeze warmup over iterations 0–500, the per-iteration log-scale clamp applied after iteration 500, and the screen-space anti-anisotropy penalty Laspect active from iteration 500. Row (a) is the full configuration (matching Tab. 5 row (D)); rows (b)–(d) each remove one component while holding the other two fixed. Removing the scale-freeze warmup (b) restores early-iteration anisotropy spikes that Laspect alone (active after iter 500) cannot retroactively arrest. Removing the log-clamp (c) lets accumulated log-scale drift past the budget between iterations, which Laspect as a soft penalty cannot fully reabsorb. Removing Laspect (d) lets gaussians regrow into thin screen-space streaks once the projection direction shifts away from training views, against which the warmup-then-clamp schedule provides no viewdependent protection. These failure modes do not overlap: the full regularization keeps maxAR ≈ 8 at iter 10K, removing Laspect drifts to ≈ 20, and removing the log-clamp causes catastrophic elongation (maxAR > 106, runtime status=1). The refinement photometric loss largely compensates at the rendered output (PSNR spread under 2.1 dB), so maxAR rather than rendered appearance is the more informative signal: each of the three components addresses a distinct failure mode.

## C Limitations and Future Work

Our method inherits two limitations from its FLAME-mesh foundation: large-scale accessories such as glasses or hats are not explicitly modeled and may be folded into the head’s surface texture, and the upstream FLAME tracker assumes a successful frontal face detection on at least one source frame. The per-subject refinement loop also remains a ∼2-minute step, so true real-time avatar creation is restricted to the feed-forward stage. Two natural extensions follow: (i) coupling with audio- and motion-driven control signals to form a complete digital-human stack beyond the still-portrait input regime, and (ii) extending the FLAME-mesh-bound Gaussian representation with body and hand sub-meshes for full-body 4D digital humans.