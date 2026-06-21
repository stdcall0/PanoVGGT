# PanoVGGT Gaussian Head Training Stages

This is the compact training map for the 3DGS head. The active configs keep
`model.gs_subgrid_size: 2` from Stage 1 onward, so every stage uses the same
2x2 Gaussian-head topology.

## Resume Rule

- Stage 1 starts from a PanoVGGT checkpoint without `gaussian_head.*` weights.
- Do not resume these stages from old 1x Gaussian-head checkpoints; the 2x2
  head has different parameter shapes.
- Each stage config uses `init_checkpoint_path` to initialize model weights from
  the base model or previous stage while starting fresh epoch/optimizer/scaler
  state.
- `resume_checkpoint_path` is only for continuing an interrupted run of the same
  stage. Keep previous-stage checkpoints out of `resume_checkpoint_path`.
- Auto-resume checks the stage `save_dir` before `init_checkpoint_path`. To
  restart a stage from the handoff checkpoint, delete, move, or change that
  stage's output directory first.

## Stages

| Stage | Config | Trainable GS Parameters | Init/Handoff Source |
| --- | --- | --- | --- |
| 1 bootstrap | `training/config/gs/stage1_bootstrap.yaml` | `sh_dc`, `opacity` | base PanoVGGT checkpoint |
| 2 refine | `training/config/gs/stage2_refine.yaml` | `sh_dc`, `opacity`, bounded `scale`, bounded `offset` | Stage 1 |
| 2b coverage | `training/config/gs/stage2b_coverage.yaml` | same as Stage 2 | Stage 2 |
| 2c novel view | `training/config/gs/stage2c_novel_view.yaml` | same as Stage 2 | Stage 2b |
| 3 full head | `training/config/gs/stage3_full_head.yaml` | add low-LR `rotation` and `sh_rest` | Stage 2c |

## Stage Notes

Stage 1 bootstraps stable DC color and opacity with source self-reconstruction.
Keep scale, offset, rotation, and SH residuals closed here.

Stage 2 opens bounded scale and offset after color/opacity are usable. Watch
scale and offset histograms; if they sit on bounds, reduce freedom before
raising loss weights.

Stage 2b adds gentle coverage and front-floater pressure. Roll back if alpha
turns uniformly saturated or scale grows just to paint holes.

Stage 2c is the novel-view bootstrap. Model forward receives only source views;
target views are used only by the render loss. The target camera/depth bootstrap
must stay in the same normalized scale as the predicted source geometry.

Stage 3 opens residual rotation and `sh_rest` only after bounded geometry is
visually stable. Keep their learning rates below the DC/opacity rate and keep
scale/offset regularization active.

## Keep Checkpoints By Evidence

Keep a stage checkpoint only when validation renders improve and parameter
histograms remain stable. Train loss alone is not enough for this head.
