# ENV

## 1. 适用范围

这份文档记录的是 **NVIDIA Blackwell 显卡** 上可复现的 `PanoVGGT` 环境配置流程，重点面向：

- `RTX 5090` 这一类 Blackwell GPU
- 需要训练或运行你当前这个带 `3DGS` 头的仓库
- Linux + conda 环境

这套组合已经在当前仓库上实际跑通过训练。和仓库 `README.md` 里的默认 quick start 相比，最大的区别是：

- 不使用 `torch 2.5.x + cu124`
- 不直接照抄 `requirements.txt` 里的旧 torch 版本
- `gsplat` 需要按 Blackwell 这套方式单独处理

## 2. 已验证组合

当前机器上实际验证通过的组合是：

- GPU: `NVIDIA GeForce RTX 5090`
- Driver: `580.95.05`
- Python: `3.11`
- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torchaudio 2.11.0+cu128`
- `xformers 0.0.35`
- `gsplat 1.5.3`
- `cuda-nvcc 12.8.93`
- `cuda-toolkit 12.8.1`

说明：

- Blackwell 上，仓库默认文档里的 `torch 2.5.x + cu124` 不建议继续用。
- 这台机器 `nvidia-smi` 显示运行时支持 `CUDA 13.0`，但用户态 PyTorch/编译链实际用的是 `cu128` 这套组合。
- 训练入口本身还依赖 `panovggt/utils/runtime_env.py` 里的运行时 bootstrap 来规避 `libstdc++` / `CXXABI` / `gsplat_cuda.so` 问题。

## 3. 前置条件

开始之前先确认：

```bash
nvidia-smi
```

至少要满足：

- 驱动已经正确识别 Blackwell 显卡
- 能看到 GPU 型号
- 驱动版本足够新

当前机器的参考输出是：

- `NVIDIA GeForce RTX 5090`
- `Driver Version: 580.95.05`

如果这里都不正常，先不要安装 Python 依赖，先把驱动/GPU 可见性修好。

## 4. 推荐安装流程

### 4.1 创建 conda 环境

```bash
conda create -n panovggt python=3.11 -y
conda activate panovggt
```

### 4.2 安装 Blackwell 兼容的 PyTorch 组合

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128
```

这里不要先跑仓库 `README.md` 里的 `cu124` 指令，也不要先 `pip install -r requirements.txt`，否则很容易把 torch 降回旧版本。

### 4.3 安装项目其余依赖，但跳过旧版 torch 相关包

仓库里的 `requirements.txt` 目前仍然固定了旧版本：

- `torch==2.5.0`
- `torchvision==0.20.0`
- `torchaudio==2.5.0`

所以这里不要直接执行：

```bash
pip install -r requirements.txt
```

推荐做法是先过滤掉 `torch / torchvision / torchaudio / xformers / gsplat`，再安装剩余依赖：

```bash
grep -Ev '^(torch|torchvision|torchaudio|xformers|gsplat)(==|$)' requirements.txt > /tmp/panovggt_requirements_no_torch.txt
pip install -r /tmp/panovggt_requirements_no_torch.txt
```

如果后面训练时报缺包，再补这些训练侧依赖：

```bash
pip install iopath fvcore wcmatch
```

### 4.4 在 conda 环境里补 CUDA 编译工具链

`gsplat` 在 Blackwell 上不要依赖系统里“碰巧可用”的 nvcc，直接把编译工具链装进当前 conda 环境：

```bash
conda install -y -c nvidia cuda-nvcc=12.8 cuda-toolkit=12.8
```

安装后确认：

```bash
which nvcc
nvcc --version
```

### 4.5 用源码方式安装 gsplat

不要依赖旧 wheel，直接走源码安装：

```bash
pip uninstall -y gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
```

说明：

- 当前验证通过的结果版本是 `gsplat 1.5.3`
- 第一次真正调用 `gsplat` 时，通常还会在 `/tmp/torch_extensions` 下进行一次 JIT 编译
- 首次编译明显比后续启动慢，这是正常现象

## 5. 安装完成后的验证

### 5.1 验证包版本

```bash
pip show torch torchvision torchaudio xformers gsplat
```

预期至少应当看到：

- `torch 2.11.0+cu128`
- `torchvision 0.26.0+cu128`
- `torchaudio 2.11.0+cu128`
- `xformers 0.0.35`
- `gsplat 1.5.3`

### 5.2 验证 PyTorch 能识别 GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

理想输出应当类似：

- `2.11.0+cu128`
- `12.8`
- `True`
- `NVIDIA GeForce RTX 5090`

### 5.3 验证 gsplat 运行时 bootstrap

对这个仓库，建议用下面的方式验证：

```bash
python -c "from panovggt.utils.runtime_env import bootstrap_gsplat_runtime; bootstrap_gsplat_runtime(); import gsplat; import torch; print('gsplat ok'); print(torch.__version__)"
```

如果这一步失败，优先检查：

- `cuda-nvcc` 是否真的装在当前 conda 环境里
- 是否误把系统旧版 `libstdc++.so.6` 抢先加载了
- 是否绕开了仓库自带的 runtime bootstrap

## 6. 训练和推理时的注意事项

### 6.1 尽量走仓库的 `training/launch.py`

训练入口会在导入训练主逻辑之前自动执行：

- `panovggt.utils.runtime_env.bootstrap_gsplat_runtime()`

这一步会：

- 设置 `CUDA_HOME`
- 设置 `CUDA_PATH`
- 设置 `CUDACXX`
- 补 `CPATH`、`CPLUS_INCLUDE_PATH`
- 补 `LIBRARY_PATH`、`LD_LIBRARY_PATH`
- 设置 `TORCH_EXTENSIONS_DIR`
- 预加载 conda 环境里的 `libgcc_s.so.1` 和 `libstdc++.so.6`

所以最稳的启动方式仍然是：

```bash
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 torchrun --standalone --nproc_per_node=1 training/launch.py --config panocity_partial_gaussian_v2_base
```

### 6.2 自定义脚本不要先 import torch

如果你写单独调试脚本，顺序尽量写成：

```python
from panovggt.utils.runtime_env import bootstrap_gsplat_runtime

bootstrap_gsplat_runtime()

import torch
import gsplat
```

不要先 `import torch` 再晚一点才做 bootstrap，否则很容易重新遇到：

- `CXXABI_*` not found
- `libstdc++.so.6` 冲突
- `gsplat_cuda.so` 加载失败

### 6.3 受限网络环境建议跳过 DINOv2 预下载

训练时可以显式设置：

```bash
export PANOVGGT_SKIP_DINOV2_DOWNLOAD=1
```

这样可以避免启动阶段卡在 DINOv2 权重下载上。

## 7. 常见问题

### 7.1 `pip install -r requirements.txt` 之后环境坏掉了

原因通常是：

- `requirements.txt` 里的 `torch==2.5.0` 把新装的 `2.11.0+cu128` 覆盖掉了

处理方式：

```bash
pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128
```

必要时再重装一次 `gsplat`：

```bash
pip uninstall -y gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
```

### 7.2 `torch.cuda.is_available()` 是 `False`

优先检查：

- `nvidia-smi` 是否正常
- 当前 shell 是否真的能看到宿主机 GPU
- 驱动是否太旧
- 是否在一个没有 GPU 权限的容器或 sandbox 里测试

### 7.3 `gsplat_cuda.so` / `libstdc++.so.6` / `CXXABI` 相关报错

优先检查：

- 是否通过 `training/launch.py` 启动
- 当前 conda 环境里是否装了 `cuda-nvcc`
- `CONDA_PREFIX/lib/libstdc++.so.6` 是否存在
- 是否在 bootstrap 之前就已经 import 了 `torch`

如果还有编译器兼容问题，优先尝试 `GCC 11`。这是当前机器上实际可用的组合。

### 7.4 第一次启动特别慢

常见原因：

- `gsplat` 首次 JIT 编译
- DINOv2 权重检查或下载

只要不是卡住几个小时，第一次慢一些通常是正常现象。

## 8. 一套可直接复现的命令

如果你只是想在 Blackwell 上快速得到一套能跑训练的环境，可以直接按这个顺序执行：

```bash
conda create -n panovggt python=3.11 -y
conda activate panovggt

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128

grep -Ev '^(torch|torchvision|torchaudio|xformers|gsplat)(==|$)' requirements.txt > /tmp/panovggt_requirements_no_torch.txt
pip install -r /tmp/panovggt_requirements_no_torch.txt
pip install iopath fvcore wcmatch

conda install -y -c nvidia cuda-nvcc=12.8 cuda-toolkit=12.8

pip uninstall -y gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "from panovggt.utils.runtime_env import bootstrap_gsplat_runtime; bootstrap_gsplat_runtime(); import gsplat; print('gsplat ok')"
```

## 9. 当前仓库的结论

对这个仓库来说，Blackwell 上最稳的经验结论就是：

- 用 `Python 3.11`
- 用 `torch 2.11.0+cu128`
- 用 `xformers 0.0.35`
- `gsplat` 用源码安装，不用旧 wheel
- `cuda-nvcc` 装到当前 conda 环境里
- 训练尽量走 `training/launch.py`
- 自定义脚本里先 bootstrap，再 import `torch`

如果只记住一件事，就是：**不要直接照抄 README 里的 `cu124` 环境，也不要直接把 `requirements.txt` 整份装进去。**
