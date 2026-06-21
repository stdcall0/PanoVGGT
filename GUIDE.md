# GUIDE — Pano Feed-Forward 3DGS Branch

This guide covers training and inference for the GS branch added in
`work/gs-test`. It assumes:

- Conda env: `panovggt` (CUDA 12.4, torch 2.5.0, gsplat 1.5.3 — verified).
- Repo root: `/mnt/g/CS/1-PanoGS-stable/PanoVGGT`.
- Branch: `work/gs-test`.

If `gsplat` is missing in your env, install it once:

```bash
conda activate panovggt
pip install gsplat==1.5.3
```

---

## 1. Training

The four stage configs live in `training/config/gs/`:

```
training/config/gs/stage_a.yaml   # color + opacity, single-scene overfit
training/config/gs/stage_b.yaml   # color + opacity, multi-scene
training/config/gs/stage_c.yaml   # + scale + rotation
training/config/gs/stage_d.yaml   # + SH rest + center offset, backbone unfrozen
```

### 1.1 Stage A — single-scene overfit (debugging)

Validates that the photometric render loss drives color in the right
direction before any geometry parameter is unfrozen.

```bash
conda activate panovggt
cd training
python launch.py \
    --config gs/stage_a \
    +data.train.dataset.dataset.datasets.0.fix_seq_index=0 \
    +limit_train_batches=200
```

The `+fix_seq_index=0` override pins the loader to a single sequence so
you can watch one scene transition from gray to colored over ~50 iters.

### 1.2 Stage B — color + opacity, multi-scene

```bash
python launch.py --config gs/stage_b
```

### 1.3 Stage C — open scale + rotation

```bash
python launch.py --config gs/stage_c
```

Cube faces are rendered at 384² (vs 256² in A/B) for finer photometric
supervision now that scale/rotation are trainable.

### 1.4 Stage D — full feed-forward GS

```bash
python launch.py --config gs/stage_d
```

The backbone is unfrozen at LR 1e-6 so the existing PanoVGGT heads can
re-tune slightly under the GS loss. Cube faces render at 512².

### 1.5 Resuming or warm-starting from a previous stage

The stage configs do not auto-load checkpoints from a previous stage.
Pass `+checkpoint.resume_from=<path>` to start from a stage-A/B/C
checkpoint:

```bash
python launch.py --config gs/stage_b \
    +checkpoint.resume_from=runs/gs_stage_a/checkpoints/last.pt
```

### 1.6 Tuning common knobs

Override loss weights or face resolution at the CLI:

```bash
python launch.py --config gs/stage_b \
    loss.gs.rgb_weight=1.0 \
    loss.gs.ssim_weight=0.5 \
    loss.gs.face_res=320
```

---

## 2. Inference

Inference now exposes `--enable_gaussian` and `--gs_ply` flags. Either
flag implies the GS branch must be turned on in the model.

### 2.1 Export Gaussians as a 3DGS .ply

```bash
conda activate panovggt
python inference.py \
    --config training/config/gs/stage_d.yaml \
    --checkpoint training/runs/gs_stage_d/checkpoints/last.pt \
    --image_dir examples/stanford2d3d/area1 \
    --output_dir results/area1 \
    --gs_ply  results/area1/gaussians.ply
```

The `.ply` follows the standard inria 3DGS layout (vertex header
`x y z nx ny nz f_dc_0..2 f_rest_… opacity scale_… rot_…`) and opens
in:

- [SuperSplat](https://playcanvas.com/supersplat/editor) (web viewer)
- The official 3DGS viewer
- Any 3DGS-compatible toolchain that consumes inria-format PLY

### 2.2 Standard inference still works

The original `--config training/config/default.yaml` and pre-existing
non-GS checkpoints continue to run unchanged — the GS branch is gated
by `--enable_gaussian` / `--gs_ply` and the model config flag
`model.enable_gaussian`.

---

## 3. Smoke tests

Quick CPU/GPU sanity tests bundled in the changes:

```bash
conda activate panovggt
cd /mnt/g/CS/1-PanoGS-stable/PanoVGGT
PYTHONPATH=$PWD python -c "
import torch
from panovggt.layers.gaussian_head import LinearGaussianHead
h = LinearGaussianHead(dec_embed_dim=1024, sh_degree=1)
out = h(torch.randn(2, 32, 1024), Hp=4, Wp=8, B=1, S=2)
for k,v in out.items(): print(k, tuple(v.shape))
"

PYTHONPATH=$PWD python -c "
import torch
from panovggt.render.cube_renderer import CubePanoRenderer
r = CubePanoRenderer(equ_h=128, face_res=128, fov_deg=95.0, sh_degree=1).to('cuda')
N = 1000; means = torch.randn(N, 3, device='cuda')
quats = torch.zeros(N, 4, device='cuda'); quats[:, 0] = 1
scales = torch.full((N, 3), 0.05, device='cuda')
opacities = torch.full((N,), 0.5, device='cuda')
colors = torch.zeros(N, 4, 3, device='cuda'); colors[:, 0] = 1
w2c = torch.eye(4, device='cuda').unsqueeze(0)
out = r.render(means, quats, scales, opacities, colors, w2c)
print({k: tuple(v.shape) for k, v in out.items()})
"
```

---

## 4. Memory budget

The cube renderer at S=2, face_res=256, sh_degree=0 fits in
~10.5 GB on the RTX 4070 (verified). For larger configurations:

| Stage | face_res | sh_degree | S | peak VRAM |
|-------|----------|-----------|---|-----------|
| A     | 256      | 0         | 2 | ~10.5 GB  |
| B     | 256      | 0         | 4 | ~17 GB    |
| C     | 384      | 0         | 4 | ~22 GB    |
| D     | 512      | 1         | 4 | ~35 GB    |

Stage D requires an H100 (or equivalent ≥40 GB) for full multi-scene
training. Drop `face_res` or batch size on smaller GPUs.

---

## 5. Troubleshooting

- **`omegaconf.errors.ConfigAttributeError: Key 'loss' is not in struct`**
  — your stage YAML is missing `# @package _global_` at the top.
- **`AssertionError: enable_gaussian=True requires enable_global_points=True`**
  — set `model.enable_global_points: True` in the YAML.
- **`backgrounds` shape error from gsplat** — you're on a gsplat
  version older than 1.5.x. Upgrade with `pip install gsplat==1.5.3`.
