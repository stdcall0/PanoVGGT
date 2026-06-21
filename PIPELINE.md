# PIPELINE — Pano Feed-Forward 3DGS Branch

## 1. Goal

Predict a feed-forward 3D Gaussian-Splatting reconstruction directly
from one or more equirectangular panoramas, using PanoVGGT's predicted
world-frame point cloud as Gaussian centers and a pano-aware photometric
render loss for supervision.

Forward pass:

```
ERP images (B, S, 3, H, W)
        │
        ▼
   PanoVGGT backbone  (aggregator + RoPE + camera/depth/point/global heads)
        │
        ├── camera_poses (B, S, 4, 4)        # c2w
        ├── depth        (B, S, H, W, 1)
        ├── local_points (B, S, H, W, 3)
        ├── world_points (B, S, H, W, 3)     # Gaussian centers (patch-pooled)
        │
        ▼
  global_points_decoder hidden state (B*S, 1+4+Hp*Wp, 1024)
        │ (patch tokens only)
        ▼
   LinearGaussianHead  (per-token MLP: 6 small Linear layers)
        │
        ▼
  Gaussian params      (B, S, Hp, Wp, *)
   ├── offset (3)      — added to centers at Stage D
   ├── scale (3)       — softplus, init from depth footprint
   ├── rotation (4)    — L2-normalized quaternion
   ├── opacity (1)     — sigmoid
   ├── sh_dc (3)       — DC color (PixelSplat convention)
   └── sh_rest (3*K-1) — higher-order SH, zero-init
        │
        ▼
   Cube renderer (gsplat) at every input view's pose
        │  (95° FoV per face, 6 faces, w2c = inverse(c2w))
        ▼
   Cube → ERP (3-D grid_sample)
        │
        ▼
   Photometric loss (RGB L1 + SSIM, masked)
```

## 2. Components

### 2.1 `LinearGaussianHead`
File: `panovggt/layers/gaussian_head.py`

Six independent `nn.Linear(1024, dim)` projections — one per parameter
group. Per-stage freezing is therefore a regex match on
`gaussian_head.<group>` in `optim.frozen_module_names`.

Activations:

| param | activation | init bias |
|------|------------|-----------|
| offset | `reg_dense_offsets` (Splatt3R/PixelSplat) | 0 |
| scale  | `softplus(.) + 1e-6`                      | `softplus_inv(0.01)` |
| rotation | L2-normalize                            | (1, 0, 0, 0) |
| opacity | sigmoid                                  | logit(0.1) |
| sh_dc   | identity (PixelSplat: stored as (RGB-0.5)/C0) | 0 |
| sh_rest | identity                                 | 0, weight 0 |

### 2.2 `GSBranch`
File: `panovggt/render/gs_branch.py`

Stateless orchestrator that:

1. Patch-pools `world_points` to obtain per-token centers.
2. Optionally adds a per-Gaussian offset (Stage D).
3. Bootstraps DC color from a patch-pooled GT image — applied as a
   detached additive prior to the network's DC output, so the network
   gradually overrides it as `sh_dc` learns.
4. Selects scale/rotation per stage: trainable head outputs vs.
   detached initializers (depth-footprint scale, identity rotation).
5. Calls the configured renderer once per batch element.

### 2.3 Cube renderer
File: `panovggt/render/cube_renderer.py`

For each input view i:

- `w2c_i = se3_inverse(camera_poses_i)`
- For each of 6 face rotations `R_face`:
  - face camera = `face_R^T @ R_w2c`, `face_R^T @ t_w2c`
  - perspective intrinsics with FoV = 95° (`face_res` controlled per stage)
- Single batched gsplat call with all 6V cameras at once.
- Faces are stacked into `(V, C, 6, h, w)` and converted to ERP via the
  `Cube2Equirec` 3-D grid_sample module.
- A 4-px per-face boundary mask is converted alongside, multiplied by
  the cube-validity mask in `Cube2Equirec`, and used to weight the
  photometric loss.

The 95° FoV plus 4-px boundary mask is the standard
seam-suppression technique used in TPGS. We skip soft cosine fading to
keep the loss simple; if seam artifacts persist after Stage C we can add
it.

### 2.4 Cube ↔ ERP module
File: `panovggt/render/cube_to_equi.py`

A differentiable map from a cube tensor `(B, C, 6, h, w)` to ERP
`(B, C, equ_h, equ_w)` using `F.grid_sample` in 3-D mode:

- For every ERP pixel, compute its world-frame ray direction.
- Express that ray in each face's camera frame; pick the face whose
  +z component is largest.
- Within that face, map the ray to normalized image coordinates using
  the face's FoV-based intrinsic (`tan(fov/2)`).
- Use the face index (rescaled to [-1, 1]) as the depth axis of a
  3-D grid_sample.

Using `align_corners=True` and `padding_mode='border'` makes the seam
behaviour explicit: any ray landing outside the chosen face's image
plane is clamped to the border, but those rays are also masked out by
the explicit `valid_mask` so they contribute zero to the loss.

The face order matches `opencv_face_rotations()`: front, right, back,
left, up, down. Coordinate convention is OpenCV (+x right, +y down,
+z forward), aligned with PanoVGGT's `_get_direction_vectors` (where
`dir_y = -sin(theta)`).

### 2.5 Photometric loss
File: `panovggt/render/losses.py`

Composite loss = `rgb_w * masked_l1 + ssim_w * (1 - masked_ssim) + depth_w * masked_l1_depth`.

The 11×11 SSIM is implemented inline (no kornia dependency) so the
panovggt env stays minimal.

### 2.6 Loss integration
File: `panovggt/models/loss.py`

`Loss` accepts a `gs:` config dict at construction. When
`gs.enabled=True`:

- `prepare_gt` runs as usual; `images` is preserved on `pred` by the
  model (gated on `enable_gaussian`).
- `normalize_pred` runs unchanged. The world_points and camera_poses
  it returns live in the same cam0-frame, so the GS render and the
  photometric loss are in a consistent frame.
- After `point_loss + camera_loss`, the GS branch renders predicted
  Gaussians and adds RGB+SSIM(+depth) loss to `loss_objective`.
- `loss_dict` exposes `loss_gs`, `loss_gs_rgb`, `loss_gs_ssim`,
  `loss_gs_depth` for tensorboard logging.

When `gs.enabled=False`, the legacy behaviour is preserved exactly.

## 3. Stage progression

| Stage | trainable | render res | sh_degree | use offset | detach centers | backbone |
|-------|-----------|------------|-----------|------------|----------------|----------|
| A | dc, opacity | 256² | 0 | no  | yes | frozen |
| B | dc, opacity | 256² | 0 | no  | yes | frozen |
| C | dc, opacity, scale, rotation | 384² | 0 | no | yes | frozen |
| D | all + sh_rest + offset | 512² | 1 | yes | no  | LR 1e-6 |

The reasoning is straight from the user spec: prove the loss can
provide correct color before letting it touch geometry; let scale and
rotation absorb pose / shape mismatch before opening up high-frequency
SH; only let the loss reach the backbone once everything else is
stable.

## 4. Why a cubemap renderer instead of a native pano rasterizer?

- gsplat is a maintained, fast CUDA rasterizer with a stable Python
  API. It only emits perspective views, but the cube → ERP grid_sample
  is differentiable and cheap.
- The 95° FoV + boundary mask combo eliminates visible seam artifacts
  in practice (TPGS reports the same).
- Native pano rasterizers avoid the cubemap detour, but the previous ODGS
  stub was not implemented. The active code path is cube rendering only.

## 5. Inference path

`inference.py --enable_gaussian --gs_ply out.ply` runs the model with
the GS branch on, then aggregates patch-level Gaussians from every
input view into world frame and writes a standard 3DGS PLY:

- Means: predicted `world_points` patch-pool + offset.
- Scales: log-scaled before write (3DGS convention).
- Rotations: pre-normalized quaternion (wxyz).
- Opacities: logit-inverse'd before write.
- SH DC + rest: stored in inria order.

The viewer-side convention follows the official 3DGS PLY spec (verify
in SuperSplat or the inria viewer).

## 6. Files added or touched

```
PLAN.md                                      (new)
GUIDE.md                                     (new)
PIPELINE.md                                  (this file)

panovggt/layers/gaussian_head.py             (new)
panovggt/render/__init__.py                  (new)
panovggt/render/gs_utils.py                  (new)
panovggt/render/cube_to_equi.py              (new)
panovggt/render/cube_renderer.py             (new)
panovggt/render/losses.py                    (new)
panovggt/render/gs_branch.py                 (new)
panovggt/utils/gs_export.py                  (new)
panovggt/models/panovggt_model.py            (mod: enable_gaussian flag)
panovggt/models/loss.py                      (mod: gs branch)
inference.py                                 (mod: --enable_gaussian, --gs_ply)

training/config/gs/stage_a.yaml              (new)
training/config/gs/stage_b.yaml              (new)
training/config/gs/stage_c.yaml              (new)
training/config/gs/stage_d.yaml              (new)

references/                                  (gitignored — design refs only)
  splatt3r/
  OmniGS/
  TPGS/
```

## 7. Limitations and follow-ups

- The active renderer is the cube path; native pano rasterizers would need
  a fresh implementation rather than the removed ODGS stub.
- Centers are detached at Stages A-C, so `point_loss` continues to
  drive the world_points head. At Stage D the GS render contributes
  its own gradient to centers; tune `loss.gs.point_loss_weight` if you
  see the geometry head drift.
- Patch-level density (one Gaussian per 14×14 patch) loses high-freq
  detail. A future Stage E could split each patch into 2×2 sub-Gaussians
  to recover that without changing the head input.
- Render-resolution scheduling is currently config-driven; an automatic
  step scheduler that bumps `face_res` mid-stage is straightforward to
  add.
