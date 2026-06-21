# Gaussian Head Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** stabilize PanoVGGT Gaussian-head training, remove novel-view target leakage, improve sharpness beyond the current one-Gaussian-per-patch ceiling, and make rotation/SH residuals safe to unlock.

**Architecture:** keep the frozen PanoVGGT backbone as the geometry/token provider; put all Gaussian training contracts in a small number of explicit places: trainer view split, loss render contract, Gaussian materialization, renderer math, and staged configs.

**Tech Stack:** Python, PyTorch, pytest, Hydra YAML configs, WSL conda environment `panovggt`.

---

## Locked Decisions

- Use 2x2 sub-Gaussians per patch. Do not implement K=2 duplicate Gaussians.
- Accept a deterministic tangent-frame base rotation before training residual rotation.
- Keep the training manual concise: `training/train_stages.md`.
- Use WSL for tests and smoke runs:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests -q'
```

- After each feature: run focused tests, inspect `git status`, then commit only that feature.

---

## Implementation Order

### 1. Test Harness And Safety Invariants

- [ ] Create `tests/` with focused CPU-sized tests.
- [ ] Add synthetic tensor builders for Gaussian params, panorama-shaped images, depth, masks, and camera poses.
- [ ] Add import tests for the Gaussian branch and loss without launching a full trainer.
- [ ] Add a WSL test command note to the test README or docstring if needed.

Files:

- `tests/conftest.py`
- `tests/test_gaussian_materialization.py`
- `tests/test_gaussian_loss_contract.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gaussian_materialization.py tests/test_gaussian_loss_contract.py -q'
```

Commit:

```bash
git add tests
git commit -m "test: add gaussian head safety harness"
```

### 2. Make Training And Export Materialization Match

Current problem: `GaussianBranch.materialize(...)` can use `point_masks`, but the free `materialize_gaussians(...)` path calls it without masks. This makes training and export/inference disagree about valid Gaussians.

- [ ] Change `materialize_gaussians(...)` to accept and pass `point_masks`.
- [ ] Update all export/inference callers to pass the same mask used during training.
- [ ] Make missing masks explicit: either require masks for training/export parity or log a clear fallback path for inference-only tools.
- [ ] Add a parity test where masked patches are removed identically in both paths.

Files:

- `panovggt/render/gs_branch.py`
- likely `panovggt/utils/gs_export.py` or inference/export caller files discovered by `rg "materialize_gaussians"`
- `tests/test_gaussian_materialization.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gaussian_materialization.py -q'
```

Commit:

```bash
git add panovggt tests
git commit -m "fix: keep gaussian materialization masks consistent"
```

### 3. Split RGB, Depth, And Gaussian-Existence Masks

Current problem: `point_masks` is depth-valid, but it is also used for RGB masking and Gaussian existence. That can train the renderer to ignore valid RGB pixels and can remove Gaussians for reasons unrelated to visibility.

- [ ] Add explicit mask names in the Gaussian loss path:
  - `rgb_masks`: image supervision mask, default all ones if the dataset has no image mask.
  - `depth_masks`: depth supervision mask, from valid depth.
  - `source_gs_masks`: Gaussian existence/materialization mask.
- [ ] Keep `point_masks` as the backward-compatible source for `depth_masks` and `source_gs_masks`.
- [ ] Make `mask_rgb_by_valid` default false in new staged configs. If enabled, it should use an explicit intersection, not silently reuse depth validity everywhere.
- [ ] Add tests where RGB is valid but depth is invalid, and assert RGB loss still sees the pixel when configured that way.

Files:

- `panovggt/models/loss.py`
- `panovggt/render/gs_branch.py`
- `training/config/gs/*.yaml`
- `tests/test_gaussian_loss_contract.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gaussian_loss_contract.py -q'
```

Commit:

```bash
git add panovggt training tests
git commit -m "fix: separate gaussian rgb depth and existence masks"
```

### 4. Move Novel-View Source/Target Split Before Model Forward

Current problem: novel-view mode selects targets inside `Loss._compute_gs_loss`, after `PanoVGGTModel.forward` and the aggregator have already seen all views. That leaks target tokens into the Gaussian head during bootstrap.

- [ ] Add a trainer-side view split helper for Gaussian training.
- [ ] For `photometric_mode: novel_view`, call `model(images=batch["images"][:, source_idx])`.
- [ ] Attach `gs_source_indices` and `gs_target_indices` to predictions before calling the loss.
- [ ] Teach loss normalization/rendering to use source predictions and full GT batch targets.
- [ ] Allow bootstrap render poses/centers from GT with an explicit config, for example:

```yaml
loss:
  gs:
    photometric_mode: novel_view
    view_split_location: trainer
    bootstrap_geometry_source: gt
    target_policy: last
    num_target_views: 1
```

- [ ] Add a test with a fake model that records the input view count and proves target views are not forwarded.
- [ ] Add a loss test proving target images are only used as render targets, not as source images or DC bootstrap.

Files:

- `training/trainer.py`
- `panovggt/models/loss.py`
- optional helper: `training/train_utils/view_split.py`
- `tests/test_gaussian_loss_contract.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gaussian_loss_contract.py -q'
```

Commit:

```bash
git add training panovggt tests
git commit -m "fix: split gaussian novel views before model forward"
```

### 5. Fix Renderer Background And Expected Depth

Current problem: `CubeGaussianRenderer` constructs a background but does not pass it to rasterization, and ERP depth appears to use the raw depth channel rather than alpha-normalized expected depth.

- [ ] Pass configured backgrounds into the rasterizer.
- [ ] Treat rasterized depth as an accumulated moment and divide by alpha before ERP conversion, with safe epsilon.
- [ ] Keep alpha as the render support mask.
- [ ] Add tests by monkeypatching rasterization or isolating the moment-to-depth math.

Files:

- `panovggt/render/cube_renderer.py`
- `tests/test_cube_renderer.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_cube_renderer.py -q'
```

Commit:

```bash
git add panovggt/render/cube_renderer.py tests
git commit -m "fix: compute gaussian render depth from alpha moments"
```

### 6. Make RGB Loss More Robust For Panorama Training

Current problem: current MSE/L1 behavior is too sensitive to blur/holes in bootstrap and does not account for ERP solid-angle distortion.

- [ ] Add Charbonnier RGB loss option.
- [ ] Add optional ERP solid-angle weighting by latitude.
- [ ] Keep old modes for config compatibility.
- [ ] Add tests for Charbonnier value, mask normalization, and solid-angle shape/broadcasting.

Files:

- `panovggt/models/loss.py`
- optional helper: `panovggt/render/losses.py`
- `training/config/gs/*.yaml`
- `tests/test_gaussian_loss_contract.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gaussian_loss_contract.py -q'
```

Commit:

```bash
git add panovggt training tests
git commit -m "feat: add robust panorama gaussian rgb loss"
```

### 7. Replace Hard Scale Clamp With Smooth Bounded Scale

Current problem: hard clamp hides gradients at the bounds, while one-sided scale regularization only penalizes oversized splats. This can plateau blurry renders and then explode when more params open.

- [ ] Change scale head materialization to log-space residuals around `scale_init`.
- [ ] Bound scale smoothly with `tanh` or sigmoid-mapped min/max.
- [ ] Log and regularize both too-small and too-large scale multipliers.
- [ ] Remove hard forward clamp from staged configs after compatibility defaults are in place.
- [ ] Add gradient tests proving the bounded scale still has nonzero gradients near old clamp values.

Files:

- `panovggt/layers/gaussian_head.py`
- `panovggt/render/gs_branch.py`
- `panovggt/models/loss.py`
- `training/config/gs/*.yaml`
- `tests/test_gaussian_parametrization.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gaussian_parametrization.py tests/test_gaussian_materialization.py -q'
```

Commit:

```bash
git add panovggt training tests
git commit -m "feat: use smooth bounded gaussian scales"
```

### 8. Replace Saturating Offset Mapping With Scale-Relative Offset

Current problem: `reg_dense_offsets(..., shift=6.0)` starts with tiny derivatives and can produce large raw offset regularization. Offset should be expressed relative to local patch/subpatch scale.

- [ ] Materialize offset as `offset_ratio * scale_ref`, where `offset_ratio = max_ratio * tanh(raw)`.
- [ ] Use the local tangent frame or local world axes consistently.
- [ ] Keep default max ratio conservative for source reconstruction and slightly larger for novel-view bootstrap.
- [ ] Update offset regularization to penalize the ratio directly.
- [ ] Add tests for bounds, gradients, and zero initialization.

Files:

- `panovggt/layers/gaussian_head.py`
- `panovggt/render/gs_branch.py`
- `panovggt/models/loss.py`
- `training/config/gs/*.yaml`
- `tests/test_gaussian_parametrization.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gaussian_parametrization.py -q'
```

Commit:

```bash
git add panovggt training tests
git commit -m "feat: bound gaussian offsets by local scale"
```

### 9. Add Tangent-Frame Base Rotation

Current problem: opening free rotation from identity destabilizes training. The model needs a geometric base frame and only a residual rotation to learn.

- [ ] Add a utility that builds an orthonormal tangent frame for each source patch or subpatch.
- [ ] Convert tangent frames to quaternions.
- [ ] Compose learned residual quaternion with the deterministic base rotation.
- [ ] Keep residual rotation frozen until Stage 3.
- [ ] Add tests for orthonormality, quaternion normalization, identity residual behavior, and finite gradients.

Files:

- new helper: `panovggt/render/gs_geometry.py`
- `panovggt/render/gs_branch.py`
- `panovggt/layers/gaussian_head.py`
- `tests/test_gaussian_geometry.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gaussian_geometry.py tests/test_gaussian_materialization.py -q'
```

Commit:

```bash
git add panovggt tests
git commit -m "feat: initialize gaussian rotation from tangent frames"
```

### 10. Add 2x2 Sub-Gaussians Per Patch

Current problem: one Gaussian per 14x14 patch explains the current blur ceiling. K duplicate Gaussians would add capacity without spatial meaning, so use a real 2x2 subgrid.

- [ ] Add config `subgrid_size: 2` under `loss.gs` or model GS config.
- [ ] Update `LinearGaussianHead` to emit four parameter slots per patch when enabled.
- [ ] Add `subpatch_pool(...)` for centers, source image color bootstrap, depth, scale footprint, and masks.
- [ ] Preserve a clear tensor contract, preferably:

```text
B, S, Hp, Wp, Q, ...
Q = subgrid_size * subgrid_size
```

- [ ] Flatten `Q` only at render/export boundaries.
- [ ] Make PLY export count become `B * S * Hp * Wp * 4` for 2x2.
- [ ] Add tests for shape, count, mask handling, and color bootstrap using distinct subpatch colors.

Files:

- `panovggt/layers/gaussian_head.py`
- `panovggt/render/gs_branch.py`
- `panovggt/render/gs_geometry.py`
- export/inference callers
- `training/config/gs/*.yaml`
- `tests/test_gaussian_subgrid.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gaussian_subgrid.py tests/test_gaussian_materialization.py -q'
```

Commit:

```bash
git add panovggt training tests
git commit -m "feat: add 2x2 gaussian subgrid capacity"
```

### 11. Stage Configs And Concise Training Manual

- [ ] Update staged configs to reflect the new order:
  - Stage 1: source self-reconstruction, DC/opacity only.
  - Stage 2: bounded scale/offset.
  - Stage 2b: coverage/floater cleanup.
  - Stage 2c: novel-view bootstrap with trainer-side source split, optional GT geometry bootstrap, 2x2 subgrid.
  - Stage 3: tangent-frame residual rotation, then SH residual.
- [ ] Keep `training/train_stages.md` concise and aligned with config names.
- [ ] Add config smoke tests if Hydra config import is lightweight enough.

Files:

- `training/config/gs/*.yaml`
- `training/train_stages.md`
- optional `tests/test_gs_configs.py`

Tests:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests/test_gs_configs.py -q'
```

Commit:

```bash
git add training tests
git commit -m "docs: document gaussian head training stages"
```

### 12. End-To-End Smoke And Visual Gate

- [ ] Run the full unit suite.
- [ ] Run a tiny train smoke on a very small batch limit if dataset paths are available.
- [ ] Export one checkpoint or synthetic sample and render a diagnostic image.
- [ ] Compare:
  - source RGB grid
  - target novel-view grid
  - alpha map
  - scale/offset histograms
  - PLY Gaussian count

Commands:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests -q'
```

Example smoke shape, adjusted after checking launcher arguments:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python training/launch.py --config-name gs/stage2c_novel_view limit_train_batches=2 limit_val_batches=1 max_epochs=1'
```

Commit only if a code/config fix is needed. Otherwise record the result in the final implementation summary.

---

## Review Checklist Before Opening Rotation Or SH Rest

- [ ] Novel-view model input excludes target images.
- [ ] Color bootstrap uses only source images.
- [ ] Train/export materialization uses the same masks.
- [ ] RGB loss is not accidentally masked by missing depth.
- [ ] Scale and offset are bounded smoothly and keep gradients.
- [ ] 2x2 subgrid increases spatial capacity without duplicating identical patch centers.
- [ ] Tangent-frame residual rotation is stable with identity residual.
- [ ] Stage 3 starts from a visually stable Stage 2c checkpoint.
