# Staged Training Guide for 3DGS Head

目标是从官方 PanoVGGT 权重启动，只训练 3DGS head，避免旧的坏 Matterport RGB checkpoint 继续污染训练。所有命令默认从仓库根目录 `PanoVGGT-new` 执行。

## 0. 前置检查

先确认 Matterport RGB 已修复：

```bash
find ../dataset/Matterport3D/processed -path '*/pano_skybox_color_fixed/*.png' -type f | wc -l
```

如果数量明显少于 processed 里的 Matterport pano 总数，先跑：

```bash
conda run --no-capture-output -n panovggt python guides/matterport_data_preprocess/rebuild_matterport_skybox_color.py --all -j 8
```

当前 loader 会优先读取 `pano_skybox_color_fixed`，缺失时才回退到旧 `pano_skybox_color`。

## 1. 配置文件

已创建 3 个训练 stage cfg：

```text
training/config/gs/stage1_bootstrap.yaml
training/config/gs/stage2_refine.yaml
training/config/gs/stage3_full_head.yaml
```

它们继承 `training/config/gs/local.yaml` 的数据集和渲染设置，只覆盖 stage 名称、冻结模块、学习率、checkpoint 来源。

## 2. Stage 策略

Stage 1 `gs/stage1_bootstrap`：从 `./checkpoints/model.pt` 加载官方 PanoVGGT 权重，只训练 `gaussian_head.sh_dc` 和 `gaussian_head.opacity`。这个阶段让颜色 DC 和透明度先适配 3DGS renderer，降低后续 scale/offset 一起动时的发散风险。

Stage 2 `gs/stage2_refine`：从 Stage 1 的 model-only checkpoint 继续，训练 `gaussian_head.offset`、`scale`、`opacity`、`sh_dc`，仍然冻结 rotation 和 SH rest。这个是主训练阶段。

Stage 3 `gs/stage3_full_head`：可选低学习率 full GS-head finetune，解冻 rotation 和 SH rest。只有 Stage 2 loss 已经平台化、可视化没有明显漂移时再进入；否则容易把 rotation/SH rest 学成补偿数据噪声的自由度。

## 3. 启动环境

```bash
conda activate panovggt
export PYTHONPATH=$PWD:$PWD/training
export TORCH_HOME=$PWD/outputs/torch_cache
export XDG_CACHE_HOME=$PWD/outputs/cache
export HF_HOME=$PWD/outputs/hf_cache
export TORCH_EXTENSIONS_DIR=$PWD/outputs/torch_extensions
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

如果当前机器 conda 环境名是 `PanoVGGT` 而不是 `panovggt`，把命令里的环境名替换掉。

## 4. 清理旧训练输出

不要继续使用这些旧目录作为最终训练来源，因为它们包含错位 Matterport RGB 上训练过的权重：

```text
outputs/gs_local
outputs/gs_local_stage2
```

新 cfg 使用独立目录：

```text
outputs/gs_stage1_bootstrap
outputs/gs_stage2_refine
outputs/gs_stage3_full_head
```

如果要从头重跑某个 stage，先删除或移动对应的新目录，否则 trainer 会从 `checkpoint.save_dir` 自动续训：

```bash
rm -rf outputs/gs_stage1_bootstrap outputs/tensorboard/gs_stage1_bootstrap
rm -rf outputs/gs_stage2_refine outputs/tensorboard/gs_stage2_refine
rm -rf outputs/gs_stage3_full_head outputs/tensorboard/gs_stage3_full_head
```

## 5. 推荐启动方式

在 tmux 中启动，session 名固定为 `mch-train`：

```bash
tmux new -s mch-train
```

Stage 1：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 torchrun --standalone --nproc_per_node=8 training/launch.py --config gs/stage1_bootstrap
```

Stage 1 结束后，抽出 model-only checkpoint，避免 Stage 2 误读旧 optimizer/scaler state：

```bash
conda run --no-capture-output -n panovggt python guides/staged_training/extract_model_only.py   --input outputs/gs_stage1_bootstrap/ckpts/checkpoint.pt   --output outputs/gs_stage1_bootstrap/ckpts/stage1_model_only.pt
```

Stage 2：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 torchrun --standalone --nproc_per_node=8 training/launch.py --config gs/stage2_refine
```

Stage 3 可选：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 torchrun --standalone --nproc_per_node=8 training/launch.py --config gs/stage3_full_head
```

如果 Stage 2 的最佳 checkpoint 不是最后一个，可以覆盖 Stage 3 的启动权重：

```bash
torchrun --standalone --nproc_per_node=8 training/launch.py --config gs/stage3_full_head   checkpoint.resume_checkpoint_path=./outputs/gs_stage2_refine/ckpts/checkpoint_75.pt
```

## 6. 监控训练

日志目录：

```text
outputs/gs_stage*/gs_stage*/log*.txt
outputs/tensorboard/gs_stage*
```

常用检查：

```bash
tmux capture-pane -pt mch-train -S -80
tail -n 80 outputs/gs_stage2_refine/gs_stage2_refine/log_rank0.txt
```

正常趋势应该是 `loss_objective/loss_gs/loss_gs_rgb` 在前几个 epoch 明显下降，随后缓慢下降或平台化。若出现以下情况，优先回查 Matterport 数据而不是继续调学习率：

```text
loss 长时间不降或突然大幅升高
可视化 input panorama 仍有水平错位
valid mask 比例异常低
Stage 2 offset/scale 一解冻就快速发散
```

## 7. 当前建议

先完整跑 Stage 1 + Stage 2。Stage 3 不作为默认必跑阶段；它只在 Stage 2 的 infer 结果已经稳定，但仍需要更高阶外观表达时启用。这样比一开始直接全解冻 GS head 更稳，因为当前 3DGS branch 依赖冻结的 PanoVGGT geometry/depth/camera 作为 bootstrap，rotation 和 SH rest 早期自由度太高，容易吸收数据/渲染误差。
