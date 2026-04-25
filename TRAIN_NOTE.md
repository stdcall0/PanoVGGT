# TRAIN_NOTE

## 1. 当前训练仓库

- 分支：`gaussian`
- 工作目录：`/home/featurize/PanoVGGTv2`
- conda 环境：`panovggt`

进入仓库：

```bash
cd /home/featurize/PanoVGGTv2
git checkout gaussian
```

## 2. 当前可用环境

这次训练实际使用的是 `panovggt` 环境，关键依赖版本如下：

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torchaudio 2.11.0+cu128`
- `xformers 0.0.35`
- `gsplat 1.5.3`（官方 GitHub 源码安装，不是旧 wheel）
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
- 如果环境被重建，至少要保证 `torch/xformers/gsplat` 这一组版本兼容，否则容易出现 CUDA kernel、ABI 或 import 问题。
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

- 第一次真正调用 `gsplat` 时会在 `/tmp/torch_extensions` 下 JIT 编译扩展，首次耗时会明显更长
- 不要在自定义脚本里先 `import torch` 再晚些才走 `gsplat` bootstrap；最稳的启动方式就是走 `training/launch.py`
- 如果要写独立调试脚本，优先先 `import panovggt.utils.gaussian_render` 或 `from panovggt.utils.runtime_env import bootstrap_gsplat_runtime; bootstrap_gsplat_runtime()`，再 import `torch`

## 3. 数据和权重路径

本次训练直接使用：

- 数据集：`/home/featurize/PanoCity`
- 原版 PanoVGGT 权重：`/home/featurize/checkpoints/model.pt`

当前配置里：

- 训练集长度：`len_train: 4000`
- 验证集长度：`len_test: 256`

日志里已确认：

- `PanoCity_DIR is /home/featurize/PanoCity`
- `Training: PanoCity Data size: 528 trajectories`
- `Testing: PanoCity Data size: 29 trajectories`

## 4. 从原版权重开始训 3DGS

这次不是 resume 旧训练，而是拿原版权重做初始化：

- 配置项：`checkpoint.init_checkpoint_path: /home/featurize/checkpoints/model.pt`
- 实际配置文件：`training/config/panocity_partial_gaussian_v2_base.yaml`

对应策略：

- 前面的主干参数从原版 `model.pt` 初始化
- 新增的 `gaussian_*` 头因为原权重里没有，对应参数会显示为 `Missing`
- 原版 checkpoint 里的 global-point 相关参数在当前模型里不用，会显示为 `Unexpected`
- 这两类日志都是预期现象，不代表加载失败

训练时已经冻结非 3DGS 分支，只训高斯相关模块：

- 冻结：`aggregator`
- 冻结：`point_decoder`、`point_head`
- 冻结：`camera_decoder`、`camera_head`
- 冻结：`pos_adapters.point`
- 冻结：`pos_adapters.camera`

当前模型摘要：

- 总参数量：`987.6M`
- 可训练参数：`93.2M`
- 冻结参数：`894.5M`

这正是“沿用原版 PanoVGGT 权重，只训练你加的 3DGS 头和相关适配层”的状态。

## 5. 当前训练配置

实际启动配置：

- `training/config/gaussian_only.yaml`
- `training/config/panocity_partial_gaussian_v2_base.yaml`

关键配置：

```yaml
checkpoint:
  init_checkpoint_path: /home/featurize/checkpoints/model.pt

logging:
  log_dir: /home/featurize/PanoVGGTv2/outputs

data:
  train:
    dataset:
      dataset_configs:
        - PanoCity_DIR: /home/featurize/PanoCity
          len_train: 4000
  val:
    dataset:
      dataset_configs:
        - PanoCity_DIR: /home/featurize/PanoCity
          len_test: 256
```

`gaussian_only.yaml` 里还固定了：

- `loss.point_loss_weight = 0.0`
- `loss.camera_loss_weight = 0.0`
- `loss.gaussian_loss_weight = 1.0`
- `model.enable_global_points = False`
- `model.enable_3dgs = True`

虽然日志里还会打印 point/camera 的监控项，但它们不参与最终优化目标。

当前 3DGS 渲染监督还做了这几处重要约定，后续复现要注意：

- 不再使用旧版 `source frame -> disjoint target frame` 的随机配对渲染监督
- 现在直接使用样本内全部帧的全局高斯去渲染全部帧
- alpha 监督不再是硬二值 BCE，而是 cubemap 软 mask + 类别平衡加权 BCE
- render loss 的权重比旧版更高，proxy 类子项权重相对收敛得更保守
- 日志里额外记录了 active Gaussian 数、active ratio、平均 opacity/scale、target/render alpha mean 等诊断量

## 6. 启动训练

前台启动：

```bash
cd /home/featurize/PanoVGGTv2
conda run -n panovggt torchrun --standalone --nproc_per_node=1 training/launch.py --config panocity_partial_gaussian_v2_base
```

后台 `screen` 启动：

```bash
screen -dmS panocity_gaussian_v2 -L -Logfile /home/featurize/PanoVGGTv2/outputs/panocity_partial_gaussian_v2_base_screen.log zsh -lc 'cd /home/featurize/PanoVGGTv2 && exec conda run -n panovggt torchrun --standalone --nproc_per_node=1 training/launch.py --config panocity_partial_gaussian_v2_base'
```

当前就是用这个 `screen` 会话在跑：

- 会话名：`panocity_gaussian_v2`

进入会话：

```bash
screen -r panocity_gaussian_v2
```

离开但不杀进程：

```bash
Ctrl+A 然后按 D
```

## 7. 监控训练

训练日志：

```bash
tail -f /home/featurize/PanoVGGTv2/outputs/panocity_partial_gaussian_v2_base/log.txt
```

`screen` 原始输出：

```bash
tail -f /home/featurize/PanoVGGTv2/outputs/panocity_partial_gaussian_v2_base_screen.log
```

看 GPU：

```bash
conda run -n panovggt nvidia-smi
```

看 TensorBoard：

```bash
conda run -n panovggt tensorboard --logdir /home/featurize/PanoVGGTv2/outputs/tensorboard --port 6006
```

输出目录：

- 日志：`/home/featurize/PanoVGGTv2/outputs/panocity_partial_gaussian_v2_base/log.txt`
- TensorBoard：`/home/featurize/PanoVGGTv2/outputs/tensorboard/panocity_partial_gaussian_v2_base`
- checkpoint：`/home/featurize/PanoVGGTv2/outputs/panocity_partial_gaussian_v2_base/ckpts`

## 8. 这次为训练做的代码修改和注意事项

### 8.1 支持“原版权重初始化但不 resume”

文件：

- `training/trainer.py`

作用：

- 新增 `init_checkpoint_path` 初始化逻辑
- 只加载模型参数，不恢复 optimizer / scaler / epoch
- 适合“只新增 3DGS 头，不想重训前面大模型”的场景

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

### 8.6 训练日志中的加载提示是正常的

从原版 `model.pt` 初始化时，日志会打印：

- 一长串 `Missing: gaussian_*`
- 一长串 `Unexpected: global_points_*`

这是因为：

- 你的 3DGS 头是新增模块，原版权重里没有
- 原版 checkpoint 里带有旧的 global-point 头，而当前 gaussian-only 训练不使用它

只要日志里有：

```text
Initializing model weights from /home/featurize/checkpoints/model.pt
```

并且后面能正常进入：

```text
Train Epoch: [0][  0/449]
```

就说明训练已经按预期启动。
