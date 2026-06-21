# PanoVGGT Gaussian Head Training Stages

This note is a compact training map for the Gaussian head. The main rule is:
open capacity only after the current stage has a clean supervision contract,
stable gradients, and visually useful renders.

## Shared Rules

- Run tests and smoke jobs in WSL with `conda run -n panovggt`.
- Keep the PanoVGGT backbone frozen unless a stage explicitly says otherwise.
- Do not let novel-view target images enter model forward or color bootstrap.
- Treat RGB masks, depth masks, and Gaussian-existence masks as different things.
- Prefer smooth bounded parameterizations over hard clamps.
- Move to the next stage by validation images and parameter histograms, not train loss alone.

Useful command shape:

```bash
wsl bash -lc 'cd /mnt/g/CS/3-PanoVGGT-workspace/src && conda run -n panovggt python -m pytest tests -q'
```

## Stage 0: Renderer And Data Contract Checks

Purpose: make sure training and inference materialize the same Gaussians.

Trainable parameters: none.

Required checks:

- train/export Gaussian materialization parity
- renderer alpha/depth/background behavior
- source/target split does not leak target views into model forward
- one small inference render or diagnostic splat preview

Advance when these tests pass on CPU-sized synthetic tensors and one real sample.

## Stage 1: Source Self-Reconstruction Bootstrap

Purpose: give the head stable opacity and DC color before learning geometry.

Config: `training/config/gs/stage1_bootstrap.yaml`.

Trainable parameters: `sh_dc`, `opacity`.

Frozen parameters: `offset`, `scale`, `rotation`, `sh_rest`, backbone.

Supervision: each source view reconstructs itself from its own Gaussians.

Expected signs:

- RGB loss drops without opacity immediately saturating to 1 everywhere
- rendered alpha covers valid source pixels
- color residual stays small because DC bootstrap carries patch color

## Stage 2: Bounded Geometry Refinement

Purpose: let the head sharpen using local offsets and scales without exploding.

Config: `training/config/gs/stage2_refine.yaml`.

Trainable parameters: `sh_dc`, `opacity`, bounded `scale`, bounded `offset`.

Frozen parameters: `rotation`, `sh_rest`, backbone.

Key changes:

- scale is optimized in smooth log space
- offset is bounded relative to the local scale or subpatch footprint
- no hard scale clamp in the forward path

Advance when validation renders become sharper and scale/offset histograms stay away
from their bounds.

## Stage 2b: Coverage And Floater Cleanup

Purpose: improve holes and front floaters after Stage 2 plateaus.

Config: `training/config/gs/stage2b_coverage.yaml`.

Trainable parameters: same as Stage 2.

Loss changes:

- gentle coverage term on alpha
- gentle front-floater penalty
- RGB remains the main signal

Stop or roll back if coverage pushes opacity to uniform saturation or scale expands
just to paint holes.

## Stage 2c: Novel-View Bootstrap

Purpose: make the head learn source-to-target rendering instead of only source
reconstruction.

Config: `training/config/gs/stage2c_novel_view.yaml`.

Trainable parameters: same as Stage 2, optionally with 2x2 sub-Gaussians enabled.

Contract:

- model forward receives only source views
- target images are used only as render loss targets
- target poses use explicit GT bootstrap in this stage
- color bootstrap is computed only from source images
- 2x2 sub-Gaussians are enabled through `model.gs_subgrid_size: 2`

Advance when held-out target views improve without source reconstruction regressing
badly.

## Stage 3: Tangent-Frame Rotation And SH Residuals

Purpose: open angular/color capacity after geometry is already stable.

Config: `training/config/gs/stage3_full_head.yaml`.

Trainable parameters:

- first: residual rotation in a tangent-frame base
- then: low-LR `sh_rest`

Frozen parameters: backbone.

Rules:

- do not train free quaternions from scratch
- start from the deterministic tangent frame
- use smaller LR for `rotation` and `sh_rest` than for DC/opacity
- keep scale/offset regularization active

Roll back if gradients spike, SH residual dominates DC color, or renders become
speckled.

## Stage 4: Final Polish And Export

Purpose: validate that the trained head works in inference/export, not only in the
training loss.

Required checks:

- train/export materialization parity on a checkpoint
- source reconstruction and novel-view validation grids
- opacity, scale, offset, rotation, and SH histograms
- exported PLY opens and renders with expected Gaussian count

Only keep a stage checkpoint if it improves visual quality and does not break these
checks.
