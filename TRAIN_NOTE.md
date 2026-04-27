# TRAIN_NOTE

## 1. 当前训练仓库

- 分支：`gaussian`
- 工作目录：`/home/featurize/PanoVGGT`
- conda 环境：`panovggt`

进入仓库：

```bash
cd /home/featurize/PanoVGGT
git checkout gaussian
```

## 2. 当前可用环境

这次训练实际使用的是 `panovggt` 环境，关键依赖版本如下：

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torchaudio 2.11.0+cu128`
- `xformers 0.0.35`
- `gsplat 1.5.3`
- `cuda-nvcc 12.8.93`
- `cuda-toolkit 12.8.1`
- `open3d 0.19.0`
- `hydra 1.3.2`
- `omegaconf 2.3.0`

另外训练入口还依赖这些包，环境里也已经补齐：

- `iopath`
- `fvcore`
- `wcmatch`

说明：

- 这台机器是 Blackwell 卡，旧版 `torch 2.5.0+cu124` 不能稳定训练。
- 如果环境被重建，至少要保证 `torch/xformers/gsplat` 这一组版本兼容，否则很容易出现 CUDA kernel、ABI 或 import 问题。
- 当前 conda 前缀实际是：`/environment/miniconda3/envs/panovggt`

### 2.1 这次环境侧的关键改动

为了让 `gsplat` 在 `torch 2.11.0+cu128` 和 Blackwell 机器上可用，这次额外做了下面几步：

- 在 `panovggt` 环境里安装了 CUDA 编译工具链：`cuda-nvcc 12.8.93`
- `gsplat` 最终使用的是官方仓库源码安装：`git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation`
- 训练入口增加了运行时 bootstrap，显式设置 `CUDA_HOME`、`CUDA_PATH`、`CUDACXX`、`CPATH`、`CPLUS_INCLUDE_PATH`、`LIBRARY_PATH`、`LD_LIBRARY_PATH`、`TORCH_EXTENSIONS_DIR`
- 为了避免 `gsplat_cuda.so` 在加载时误用系统 `/lib/x86_64-linux-gnu/libstdc++.so.6`，启动时会先预加载 conda 环境里的 `libgcc_s.so.1` 和 `libstdc++.so.6`

如果后面要重建环境，最少复现以下命令：

```bash
conda run -n panovggt conda install -y -c nvidia cuda-nvcc=12.8
conda run -n panovggt pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
```

验证方式：

```bash
conda run -n panovggt python -c "import os, torch; print(os.environ.get('CONDA_PREFIX')); print(torch.__version__); print(torch.cuda.is_available())"
conda run -n panovggt pip show gsplat
```

注意：

- 第一次真正调用 `gsplat` 时会在 `/tmp/torch_extensions` 下 JIT 编译扩展，首次耗时会明显更长。
- 不要在自定义脚本里先 `import torch` 再晚些才走 `gsplat` bootstrap；最稳的启动方式就是走 `training/launch.py`。
- 如果要写独立调试脚本，优先先 `import panovggt.utils.gaussian_render`，或者先执行 `from panovggt.utils.runtime_env import bootstrap_gsplat_runtime; bootstrap_gsplat_runtime()`，再 import `torch`。

## 3. 数据、split 和权重路径

本次训练直接使用：

- 数据集：`/home/featurize/PanoCity`
- 原版 PanoVGGT 权重：`/home/featurize/PanoVGGT/checkpoints/model.pt`
- split 配置：`/home/featurize/PanoVGGT/outputs/panocity_splits/splits_config.json`
- 固定 split cache：`/home/featurize/PanoVGGT/outputs/panocity_cache/fixed_split_indices_seed42.json`

这次数据比之前多，所以 split 已经按新数据重新生成过，当前统计是：

- `1327` 条 trajectory split
- 每条 split 固定 `24` 帧
- 总计 `31848` 张 pano
- 固定随机种子 `42` 下切成：`1194 train / 66 val / 67 test`
- train split 覆盖 `60` 个 scene-block，当前数据总共也是 `60` 个 scene-block

当前配置里的采样预算是：

- `len_train: 4000`
- `len_test: 256`

这两个值是每个 epoch 的采样长度，不是唯一轨迹数量。也就是说：

- 每个 train epoch 会从 `1194` 条真实 train trajectory 中采样 `4000` 个 sequence sample
- 当前动态 dataloader 下每个 epoch 约 `449` step
- 每个 epoch 大约喂入 `~2.06 万` 张 pano exposure
- 跑满 `50` epoch 大约对应 `~102.9 万` 次 pano exposure

PanoCity 的 `split` 语义还有一个要注意的点：

- 数据集代码里 `split="train"` 会映射到 `mode="train"`
- `split="val"` 和 `split="test"` 都会映射到 `mode="val"`
- `split="test_final"` 才会映射到真正的 `mode="test"`

所以当前配置里虽然写的是 `val.dataset.dataset_configs[0].split: test`，但它实际拿到的是固定 `val` 集，也就是日志里显示的 `66 trajectories`。

## 4. 从原版权重开始训 3DGS

这次不是 resume 旧训练，而是拿原版权重做初始化：

- 配置项：`checkpoint.init_checkpoint_path: /home/featurize/PanoVGGT/checkpoints/model.pt`
- 实际配置文件：`training/config/panocity_partial_gaussian_v2_base.yaml`

对应策略：

- 第一次启动时，如果输出目录里还没有可 resume 的 checkpoint，就只加载 `model.pt` 的模型参数
- 这种初始化不会恢复 optimizer、scaler 或 epoch 计数，所以等价于“从头开始训练新增头，但主干不随机初始化”
- 之后如果输出目录里已经有 checkpoint，`training/trainer.py` 会优先走 `resume_checkpoint_path` 或自动发现 `save_dir` 里的最新 checkpoint
- 也就是说：这次首启不 resume；下次重启默认还是 resume

当前输出目录里已经有连续保存的 checkpoint，例如：

- `checkpoint_40.pt`
- `checkpoint_41.pt`
- `checkpoint_42.pt`
- `checkpoint_43.pt`
- `checkpoint_44.pt`

从原版 `model.pt` 初始化时，日志会打印：

- 一长串 `Missing: gaussian_*`
- 一长串 `Unexpected: global_points_*`

这是正常现象，不代表加载失败：

- 你的 3DGS 头是新增模块，原版权重里没有
- 原版 checkpoint 里带有旧的 global-point 头，而当前 gaussian-only 训练不使用它

只要日志里有：

```text
Initializing model weights from /home/featurize/PanoVGGT/checkpoints/model.pt
```

并且后面能正常进入：

```text
Train Epoch: [0][  0/449]
```

就说明训练已经按预期启动。

## 5. 当前训练配置

实际启动配置：

- `training/config/gaussian_only.yaml`
- `training/config/panocity_partial_gaussian_v2_base.yaml`

关键配置：

```yaml
logging:
  log_dir: /home/featurize/PanoVGGT/outputs

checkpoint:
  save_dir: ${logging.log_dir}/${exp_name}/ckpts
  init_checkpoint_path: /home/featurize/PanoVGGT/checkpoints/model.pt
  save_freq: 1

data:
  train:
    dataset:
      dataset_configs:
        - PanoCity_DIR: /home/featurize/PanoCity
          splits_config_file: /home/featurize/PanoVGGT/outputs/panocity_splits/splits_config.json
          cache_dir: /home/featurize/PanoVGGT/outputs/panocity_cache
          len_train: 4000
  val:
    dataset:
      dataset_configs:
        - PanoCity_DIR: /home/featurize/PanoCity
          splits_config_file: /home/featurize/PanoVGGT/outputs/panocity_splits/splits_config.json
          cache_dir: /home/featurize/PanoVGGT/outputs/panocity_cache
          len_test: 256
```

`gaussian_only.yaml` 里固定了：

- `loss.point_loss_weight = 0.0`
- `loss.camera_loss_weight = 0.0`
- `loss.gaussian_loss_weight = 1.0`
- `model.enable_global_points = False`
- `model.enable_3dgs = True`

训练时已经冻结非 3DGS 分支，只训高斯相关模块：

- `aggregator`
- `point_decoder`
- `point_head`
- `camera_decoder`
- `camera_head`
- `pos_adapters.point`
- `pos_adapters.camera`

当前模型摘要：

- 总参数量：`987.6M`
- 可训练参数：`93.2M`
- 冻结参数：`894.5M`

当前 3DGS 渲染监督还做了这几处重要约定，后续复现要注意：

- 不再使用旧版 `source frame -> disjoint target frame` 的随机配对渲染监督
- 直接使用样本内全部帧的全局高斯去渲染全部帧
- alpha 监督不再是硬二值 BCE，而是 cubemap 软 mask + 类别平衡加权 BCE
- render loss 权重比旧版更高，proxy 类子项相对收敛得更保守
- 日志里额外记录了 active Gaussian 数、active ratio、平均 opacity/scale、target/render alpha mean 等诊断量

## 6. 启动训练

推荐前台启动方式：

```bash
cd /home/featurize/PanoVGGT
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PANOVGGT_SKIP_DINOV2_DOWNLOAD=1
conda run -n panovggt torchrun --standalone --nproc_per_node=1 training/launch.py --config panocity_partial_gaussian_v2_base
```

说明：

- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 主要是为了让分布式错误更早暴露
- `PANOVGGT_SKIP_DINOV2_DOWNLOAD=1` 是这次补上的保险开关，受限网络环境下建议显式打开，避免 DINOv2 预训练权重下载卡住启动

后台 `screen` 启动方式：

```bash
screen -dmS panocity_gaussian_v2_base -L -Logfile /home/featurize/PanoVGGT/outputs/panocity_partial_gaussian_v2_base_screen.log zsh -lc 'cd /home/featurize/PanoVGGT && export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && export PANOVGGT_SKIP_DINOV2_DOWNLOAD=1 && exec conda run -n panovggt torchrun --standalone --nproc_per_node=1 training/launch.py --config panocity_partial_gaussian_v2_base'
```

当前训练会话名：

- `panocity_gaussian_v2_base`

进入会话：

```bash
screen -r panocity_gaussian_v2_base
```

离开但不杀进程：

```bash
Ctrl+A 然后按 D
```

## 7. 监控训练

主训练日志：

```bash
tail -f /home/featurize/PanoVGGT/outputs/panocity_partial_gaussian_v2_base/log.txt
```

`screen` 原始输出：

```bash
tail -f /home/featurize/PanoVGGT/outputs/panocity_partial_gaussian_v2_base_screen.log
```

看 GPU：

```bash
nvidia-smi
```

看进程：

```bash
ps -ef | rg 'training/launch.py --config panocity_partial_gaussian_v2_base|torchrun --standalone --nproc_per_node=1|panocity_partial_gaussian_v2_base'
```

看 screen 会话：

```bash
screen -ls
```

看 TensorBoard：

```bash
conda run -n panovggt tensorboard --logdir /home/featurize/PanoVGGT/outputs/tensorboard --port 6006
```

输出目录：

- 日志：`/home/featurize/PanoVGGT/outputs/panocity_partial_gaussian_v2_base/log.txt`
- TensorBoard：`/home/featurize/PanoVGGT/outputs/tensorboard/panocity_partial_gaussian_v2_base`
- checkpoint：`/home/featurize/PanoVGGT/outputs/panocity_partial_gaussian_v2_base/ckpts`

## 8. 这次为训练做的代码修改和注意事项

### 8.1 支持“原版权重初始化但不 resume”

文件：

- `training/trainer.py`

作用：

- 新增 `init_checkpoint_path` 初始化逻辑
- 只加载模型参数，不恢复 optimizer、scaler、epoch
- 恢复训练时仍然优先 resume 最新 checkpoint

核心逻辑是：

- `ckpt_path = resume_checkpoint_path or get_resume_checkpoint(save_dir)`
- 只有 `ckpt_path is None` 的时候，才会用 `init_checkpoint_path`

### 8.2 固定为只训练 Gaussian 分支

文件：

- `training/config/gaussian_only.yaml`
- `training/config/panocity_partial_gaussian_v2_base.yaml`

作用：

- 冻结主干、point、camera 相关模块
- 把 point/camera loss 权重设为 `0`
- 只优化 gaussian 分支

### 8.3 修复训练入口和运行时依赖

文件：

- `training/launch.py`
- `panovggt/utils/runtime_env.py`
- `training/data/composed_dataset.py`

作用：

- 启动时补齐 repo 路径，避免本地模块导入失败
- 在导入 `gsplat` 前先处理运行时库环境，降低 `libstdc++` / `CXXABI` 冲突概率
- 某些可选数据依赖改成按需导入，避免无关分支阻塞训练

和复现直接相关的细节：

- `training/launch.py` 会在 import `trainer` 前调用 `panovggt.utils.runtime_env.bootstrap_gsplat_runtime()`
- `panovggt/utils/runtime_env.py` 会预加载 conda 环境里的 `libgcc_s.so.1` 和 `libstdc++.so.6`
- 如果绕开 `training/launch.py`，又在 bootstrap 前先 import 了 `torch`，很容易重新遇到 `CXXABI_1.3.15` 或 `libstdc++` 冲突

### 8.4 修复动态 dataloader 的 epoch 长度

文件：

- `training/data/dynamic_dataloader.py`

作用：

- 不再让每个 epoch 的长度假装是固定超大值
- 当前配置下每个 epoch 约 `449` step
- 这样学习率调度和 ETA 才是正常的

### 8.5 修复 Gaussian render loss 的设备和混精度问题

文件：

- `panovggt/utils/gaussian_render.py`

这次实际踩到并修掉的点：

- `_face_rotations` 的 CPU/CUDA device mismatch
- `torch.linalg.inv` 在 `bfloat16` 下不可用
- `binary_cross_entropy` 在 autocast 下不安全

处理方式：

- 使用前把 `_face_rotations` 明确搬到目标 device
- 相机矩阵构造和求逆放在 `autocast(enabled=False)` 里并强制 `float32`
- alpha BCE 分支单独关闭 autocast，并把输入显式转成 `float32`

如果后面这个文件再被改坏，训练大概率会重新卡在 render loss 这条路径上。

### 8.6 PanoCity 数据集这次补了 split/cache 逻辑

文件：

- `training/data/datasets/panocity.py`

作用：

- 支持显式 `cache_dir`
- 支持绝对路径的 `splits_config_file`
- 如果数据集规模变化，会自动重建固定 split
- 固定 `90/5/5` 的 train/val/test 切分，并缓存到 `fixed_split_indices_seed42.json`

这次就是因为 `~/PanoCity` 数据比以前多，所以必须重看 split，不能沿用旧 cache。

### 8.7 DINOv2 预下载现在可以显式跳过

文件：

- `panovggt/models/aggregator.py`

作用：

- `requests` 改成可选依赖
- 新增 `PANOVGGT_SKIP_DINOV2_DOWNLOAD=1`

如果机器网络受限，或者不想在启动阶段卡住，建议显式设置这个环境变量。

### 8.8 训练期间出现过一个坏 depth 文件，但不会直接打停训练

日志里反复出现：

```text
/home/featurize/PanoCity/ningbo/ningbo_block0/panodepth_images/pano_depth_0000243.png
```

当前数据读取逻辑会：

- 记录错误日志
- 返回零 depth，必要时重采样

所以训练可以继续跑，但这说明数据里确实有坏样本，后面整理数据时最好修掉。

## 9. 这次训练的经验结论

### 9.1 收敛状态整体是健康的

截至 `2026-04-27`，代表性指标大致如下：

- `epoch 31` train avg:
  - `train_loss_objective: 0.4736`
  - `train_loss_gaussian: 0.8368`
  - `train_loss_gaussian_render: 0.2664`
- `epoch 42` train avg:
  - `train_loss_objective: 0.4698`
  - `train_loss_gaussian: 0.8271`
  - `train_loss_gaussian_render: 0.2609`
- `epoch 42` val avg:
  - `val_loss_objective: 0.7861`
  - `val_loss_gaussian_render: 0.2529`

经验判断：

- 总体是在慢速变好，不是震荡失控
- 后期单个 epoch 内的 `gaussian`、`render` 子项会有轻微回弹，但大多属于正常噪声
- 目前看最有代表性的近期验证结果大概在 `epoch 42`

### 9.2 当前这批数据已经够把 3DGS 头训起来

按当前 split 统计：

- train 只有 `1194` 条唯一 trajectory
- 但对应 `28656` 张唯一 pano
- 跑满 `50` epoch 时，每条 trajectory 平均会被反复采样约 `167.5` 次
- 每张唯一 pano 平均大约会被看到 `35.9` 次

这说明现在已经不是“数据太少训不起来”的阶段，而是“已经能训稳，但泛化上限受场景多样性约束”。

### 9.3 如果要做最终版，优先加新场景，不要只加同一批 block 的近邻切片

基于这次结果，更合适的数据目标大概是：

- `8万-15万` 张唯一 pano
- `3k-6k` 条 24 帧 trajectory
- `150-300` 个真正不同的 scene-block

优先级建议：

- 第一优先级是新增 scene-block
- 第二优先级是新增不同城市/区域/布局
- 最后才是把同一批 block 切得更密

原因很直接：

- 当前 train 已经覆盖了现有全部 `60` 个 scene-block
- 继续在同一批 block 上增加高度相邻的轨迹，收益通常小于加入新 block
