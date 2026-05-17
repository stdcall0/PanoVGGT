# PLAN — Pano Feed-Forward 3DGS Branch (work/gs-test)

> Branch: `work/gs-test` (cut from `main`, **not** `gaussian`).
> Goal: predict 3D Gaussians directly from a multi-view ERP pano input
> using PanoVGGT's `world_points` as Gaussian centers, and supervise with
> a pano-aware photometric render loss. Stage A→D progressive unlocking.

---

## 1. Decisions (locked from clarifying Q&A)

| Topic | Choice |
|------|--------|
| Center density | **Patch-level** — one Gaussian per ViT patch (37×74 = 2 738 / frame) |
| Renderer | **Cubemap via gsplat** as primary; native pano (ODGS rasterizer) as optional second backend behind a config switch |
| GT supervision views | All `S` input frames (auto-encoding RGB; depth optional) |
| Center source | Predicted `world_points` (detached at A-C; offset+grad enabled at D) |
| Render res schedule | Centers full-res; cube faces 256² (A/B) → 384² (C) → 512² (D) |
| Cube seam handling | Wider FoV (95°) + 4-px boundary mask in loss |
| Reference repos | Splatt3R, ODGS, OmniGS, TPGS cloned under `references/` |
| Hardware target | RTX 4070 12 GB for debugging; **H100** for full retraining |

---

## 2. Architecture

### 2.1 Backbone — unchanged

`PanoVGGTModel(images)` already returns:

| key | shape | meaning |
|-----|-------|---------|
| `images` | `(B, S, 3, H, W)` | input ERP, [-1,1] |
| `local_points` | `(B, S, H, W, 3)` | per-pixel cam-frame xyz |
| `camera_poses` | `(B, S, 4, 4)` | **c2w** |
| `world_points` | `(B, S, H, W, 3)` | per-pixel world xyz |
| `depth` | `(B, S, H, W, 1)` | depth (m) |

Plus inside the model we keep `global_point_hidden` (shape `(B*S, P, 1024)` after the global-points decoder, where `P = 1 + 4 + Hp*Wp = 1 + 4 + 37*74 = 2 743` and the first 5 are register tokens). We will **read this hidden state**, not re-compute it, and feed it to the GS head.

### 2.2 New head — `GaussianTokenHead`

File: `panovggt/layers/gaussian_head.py`

```
in:   tokens       (B*S, Hp*Wp, 1024)        # global_points hidden, patch tokens only
      world_points (B,    S,    H, W, 3)
      images       (B,    S,    3, H, W)     # for color init
out:  centers      (B, S, N, 3)              # N = S * Hp * Wp
      offsets      (B, S, N, 3)              # used only at Stage D
      scales       (B, S, N, 3)              # log-scale via softplus
      rotations    (B, S, N, 4)              # quat, normalized
      opacity      (B, S, N, 1)              # sigmoid
      sh_dc        (B, S, N, 3)              # = RGB - 0.5 (Splatt3R/PixelSplat conv.)
      sh_rest      (B, S, N, 3, sh_extra)    # zero-init, frozen until Stage D
```

Module:

```
LinearGaussianHead(nn.Module):
    proj = Linear(1024, out_dim_per_token)
    out_dim_per_token = 3 (offset) + 3 (scale) + 4 (quat) + 1 (op) + 3 (DC) + 3*sh_extra (rest)
```

(We use a *single* linear projection per patch token rather than the DPT pixel-shuffle from Splatt3R, because at patch granularity we already have one prediction per Gaussian. This matches LinearPts3d's pattern in this repo.)

Init biases (Splatt3R-style):

| param | bias | weight scale |
|-------|------|--------------|
| offset | 0 | 1e-4 |
| scale | log(s0) (`softplus_inv(s0)`) | 1e-4 |
| quat | (1, 0, 0, 0) | 1e-4 |
| opacity | logit(0.1) ≈ -2.197 | 1e-4 |
| DC | 0 (overwritten by image init at Stage A) | 1e-4 |
| SH rest | 0 | 0 |

### 2.3 Center construction

```
patch_center = avg_pool(world_points, kernel=14)        # (B,S,Hp,Wp,3)
patch_color  = avg_pool(images,       kernel=14)        # (B,S,3,Hp,Wp)
                                                        # used only as DC init signal
mean = patch_center [+ offset (Stage D)]
```

Stage A–C: `mean = patch_center.detach()`.
Stage D:   `mean = patch_center + reg_dense_offsets(offset)` (Splatt3R activation).

### 2.4 Scale init (kNN distance vs depth footprint)

Two paths, controlled by `gs.scale_init`:

- `knn`: `s0 = 0.5 * mean_dist_to_3_nn(patch_center)` (per frame, no_grad).
- `depth_footprint`: `s0 = depth * pixel_angular_size`,
  where `pixel_angular_size = (π/Hp) ≈ patch height in radians`.
  At ERP this gives a tangent-plane Gaussian footprint that scales with depth — pole inflation expected and acceptable for init.

Default = `depth_footprint` (cheaper than kNN every step; we keep it computed once per forward).

### 2.5 Color init (Stage A)

`sh_dc` is **not** the network's prediction at Stage A — instead we **overwrite** it with `(patch_color - 0.5)` in the GS rasterization call. The network's DC output is initialized to zero so no real overwrite occurs, and the network gradually takes over at Stage B.

### 2.6 Rotation init

Identity quat at init. Optionally, build a per-Gaussian local frame using `viewing_ray = patch_center - cam_center` and align z-axis to it; **deferred** — identity is simpler and Stage C will adjust.

---

## 3. Renderer

### 3.1 Cubemap path (default)

Module: `panovggt/render/cube_renderer.py`

For each input view `i` in `[0..S-1]`:

1. **w2c** = `inverse(camera_poses[:, i])`.
2. For each of 6 cube faces with rotation `R_face`:
   - Compose face camera: `R_face_w2c = R_face @ R_w2c` , `t = R_face @ t_w2c`.
   - Build perspective intrinsics with FoV = **95°** (wider) and resolution `face_res` (per-stage).
   - Call **gsplat** `rasterization(...)` with all aggregated Gaussians (all S frames concatenated). Output `rgb (H,W,3)`, `alpha`, `depth`.
3. Concatenate the 6 face outputs and grid-sample them back to ERP using `panovggt.Projection.Cube2Equirec` (already in repo) to obtain a rendered ERP at original `(H, W)`.
4. Mask = (cube valid mask) ∩ (4-px shrink in each face) — implemented by recording a binary mask per face and converting to ERP via the same Cube2Equirec.
5. Loss = `L1 + λ_ssim * (1-SSIM)` between rendered and GT ERP, masked.

Default λ_ssim = 0.2 (Splatt3R/3DGS standard).

### 3.2 ODGS native pano path (optional)

Switch: `gs.renderer = "cube" | "odgs"`. ODGS path requires building `references/ODGS/submodules/odgs-gaussian-rasterization` as an editable pip install — done lazily at first use; if build fails, fall back to cube. We only wire the call signature now; full enablement in Stage C+ as time permits.

### 3.3 SSIM / depth losses

- Use `kornia.metrics.ssim` if available, else a small 11×11 SSIM impl.
- Optional masked depth L1 (predicted-depth from rasterizer vs GT depth) controlled by `gs.depth_weight` (default 0).

---

## 4. Stage controller

YAML (`training/config/gs/stage_a.yaml` etc.) flips:

```
gs:
  enabled: true
  stage: A | B | C | D
  freeze_backbone: true        # All A-C; False at D
  scale_init: depth_footprint  # or knn
  scale_init_value: 0.005      # only used as init prior
  opacity_init: 0.1            # logit space
  use_offset: false            # True at D
  train_dc: true               # True from Stage A
  train_opacity: true          # True from Stage A
  train_scale: false           # False A/B; True C/D
  train_rotation: false        # False A/B; True C/D
  train_sh_rest: false         # True only at D
  sh_degree: 1                 # extra SH bands beyond DC; 0 at A/B
  renderer: cube
  cube:
    fov_deg: 95
    face_res: 256              # 256 (A/B), 384 (C), 512 (D)
    boundary_px: 4
  loss:
    rgb_weight: 1.0
    ssim_weight: 0.2
    depth_weight: 0.0
    point_weight: 0.0          # disable existing pts loss in A/B/C; re-enable at D blend
```

Freezing is applied via parameter-group masking in `setup_optimizer`:

| Stage | Trainable params |
|-------|------------------|
| A | dc + opacity (with init); rest frozen |
| B | dc + opacity (offsets unfrozen logically; init still helpful) |
| C | + scale + rotation |
| D | + sh_rest + offset; backbone unfrozen with low LR |

### 4.1 Single-scene overfit mode (for Stage A/B/C debugging)

Flag `gs.overfit_one_scene: true` plus `gs.overfit_seq_id`. The dataset wrapper returns the same scene every iteration. Used to validate the loss before multi-scene training.

---

## 5. Data path & training integration

- **Reuse existing datasets**: Stanford2D3D, PanoCity (already in `training/data/datasets/`). They already provide `images`, `world_points`, `camera_poses`, `depths`, `point_masks`. No new dataset code at first.
- **Loss integration**: extend `panovggt/models/loss.py` with `GaussianRenderLoss`, returned alongside the existing `point_loss` / `depth_loss`. The trainer adds `loss_gs = stage_weights * (rgb + ssim + depth)`.
- **Trainer** (`training/trainer.py`): add `gs_branch` initialization, parameter-group LR setup driven by stage config, and `loss_gs` accumulation.

---

## 6. Inference

`inference.py` already returns the prediction dict. We add a CLI flag `--gs <out.ply>` and a small helper `panovggt/utils/gs_export.py` that:

1. Aggregates patch-level Gaussians from all `S` frames into a single set in world frame.
2. Writes a PLY in the standard 3DGS format (means, scales-log, rotations, opacities, sh).

Optionally, `--render-novel <pose.json>` triggers the cube renderer at a user-supplied novel pose for sanity check.

---

## 7. File map

```
panovggt/
  layers/gaussian_head.py            (new)
  render/
    __init__.py                      (new)
    gs_utils.py                      (init helpers, kNN, depth footprint)
    cube_renderer.py                 (new — gsplat path)
    odgs_renderer.py                 (new — ODGS path, lazy import)
    losses.py                        (SSIM + masked L1 + composite loss)
    cube_to_equi.py                  (thin wrapper around existing Cube2Equirec)
  utils/
    gs_export.py                     (PLY writer)
  models/
    panovggt_model.py                (+ optional GS head, gated by enable_gaussian)
    loss.py                          (+ GaussianRenderLoss wrapper)

training/
  config/gs/
    stage_a.yaml, stage_b.yaml, stage_c.yaml, stage_d.yaml
  trainer.py                         (wire stage_weights & param groups)

inference.py                         (+ --gs export & --render-novel)

GUIDE.md                             (write at end)
PIPELINE.md                          (write at end)
```

---

## 8. Step-by-step implementation order

1. **Add `enable_gaussian` flag** to `PanoVGGTModel.__init__` and forward, returning `global_point_hidden_patches` in predictions when set (no head yet).
2. **Implement `LinearGaussianHead`** (single linear projection; init schedule).
3. **Implement `gs_utils.py`**: `kNN_scale`, `depth_footprint_scale`, `patch_pool`, `init_dc_from_image`.
4. **Implement `cube_renderer.py`**:
   - install gsplat in `panovggt` env (`pip install gsplat==1.5.3` — prebuilt CUDA 12.4 wheel exists).
   - per-face rasterization → cube tensor → Cube2Equirec → masked loss.
5. **Implement `losses.py`** SSIM + mask helpers.
6. **Stage YAMLs** + parameter-group masking in trainer.
7. **Smoke test**: single-scene overfit, Stage A, batch=1, S=2, 256² faces. Verify rendered ERP changes from gray to scene colors within 50 iterations.
8. **Stage B / C / D** integration tests (each runs ~200 iters on the 4070 against a saved Stanford2D3D mini-batch).
9. **Inference export**: PLY round-trip in a viewer (e.g. SuperSplat) for visual sanity.
10. **ODGS path** wiring (build attempted in CI; fall back if extension unavailable).
11. Write `GUIDE.md` and `PIPELINE.md`.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| gsplat CUDA ext not installable on WSL | Ship pre-built wheel install instructions; cube renderer can also be exercised on CPU at tiny res for unit test |
| Memory blow-up at S=4 full-res D | Activation checkpoint render loop; allow `face_res` schedule down |
| Patch-pooling washes out high-freq depth | Acceptable for first loop; later we can subdivide each patch into 2×2 sub-Gaussians (planned Stage E, **out of scope**) |
| ODGS extension build (gcc/CUDA mismatch) | Treat ODGS path as best-effort; cube remains canonical |
| Cube seams visible | Already mitigated with FoV=95° + boundary mask. Soft fade can be added if visible artifacts persist |
| Backbone collapse at Stage D | Lower backbone LR (1e-6) when unfrozen; keep `point_loss` & `depth_loss` weights at original values |

---

## 10. Acceptance criteria

- `python training/launch.py --config training/config/gs/stage_a.yaml --overfit-one` reduces RGB L1 below 0.05 within 200 iters on Stanford2D3D scene 1.
- `python inference.py --image_folder ... --gs out.ply` produces a viewable PLY with the right axes.
- Stage B and C runs do not crash on either renderer; train loss decreases monotonically (smoothed) over 1k iters.
- All baseline (non-GS) training entrypoints still work (regression test on `training/config/default.yaml`).
