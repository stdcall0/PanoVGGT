# Matterport3D Data Preprocess for PanoVGGT

这份 guide 固化当前项目需要的 Matterport3D 下载子文件、预处理脚本和已知的 skybox 修正。所有命令默认从仓库根目录 `PanoVGGT-new` 执行。

## 1. 需要下载的 Matterport3D 子文件

只需要下面 6 类，不需要下载全部 1.3TB release：

```text
matterport_camera_intrinsics
matterport_camera_poses
matterport_color_images
matterport_depth_images
matterport_skybox_images
house_segmentations
```

这些名字也写在 `required_filetypes.txt`，可以直接作为检查清单。

## 2. 下载原始数据

Matterport3D 需要先确认官方 TOS。脚本会在开始时要求按 Enter 确认。

```bash
mkdir -p ../dataset/Matterport3D
cp guides/matterport_data_preprocess/download_mp.py ../dataset/Matterport3D/download_mp.py
cd ../dataset/Matterport3D
python download_mp.py -o . --type matterport_camera_intrinsics matterport_camera_poses matterport_color_images matterport_depth_images matterport_skybox_images house_segmentations
```

下载后的目标结构应为：

```text
../dataset/Matterport3D/v1/scans/<scene_id>/*.zip
```

解压每个 scan 目录下的 zip：

```bash
cd ../dataset/Matterport3D
for z in v1/scans/*/*.zip; do unzip -n "$z" -d "$(dirname "$z")"; done
```

每个 scene 至少应包含：

```text
matterport_camera_intrinsics/
matterport_camera_poses/
matterport_color_images/
matterport_depth_images/
matterport_skybox_images/
house_segmentations/
```

## 3. 从原始数据生成 PanoVGGT processed 数据

完整重建 processed 数据：

```bash
cd /private/codes/exp/chenhang_workspace/homework/PanoVGGT-new
conda run --no-capture-output -n panovggt python guides/matterport_data_preprocess/matterport_preprocess.py --all -j 8
```

只处理单个 scene：

```bash
conda run --no-capture-output -n panovggt python guides/matterport_data_preprocess/matterport_preprocess.py --scene 5ZKStnWn8Zo -j 1
```

默认输入输出：

```text
raw scans:  ../dataset/Matterport3D/v1/scans
processed:  ../dataset/Matterport3D/processed
splits:     training/data/splits/matterport3d
```

输出结构：

```text
../dataset/Matterport3D/processed/
  parsed_json/<scene>.json
  train.txt
  val.txt
  test.txt
  <scene>/pano_skybox_color/<pano>.png
  <scene>/pano_depth/<pano>.png
  <scene>/pano_poses/<pano>.txt
```

## 4. 已有 processed 数据的 RGB 修复

如果 `pano_depth` 和 `pano_poses` 已经存在，只需要修正 RGB 拼接，不要覆盖原目录。使用 repair 脚本生成新目录：

```bash
conda run --no-capture-output -n panovggt python guides/matterport_data_preprocess/rebuild_matterport_skybox_color.py --all -j 8
```

默认写入：

```text
../dataset/Matterport3D/processed/<scene>/pano_skybox_color_fixed/<pano>.png
```

当前 `training/data/datasets/matterport3d.py` 会优先读取 `pano_skybox_color_fixed`，缺失时回退到旧的 `pano_skybox_color`，因此可以增量修复。

## 5. Skybox 修正点

Matterport raw skybox 的水平顺序应为：

```text
skybox1 -> skybox2 -> skybox3 -> skybox4 -> skybox1
front   -> right   -> back    -> left    -> front
```

修正后的 face 语义：

```text
0 top(+y), 1 front(-z), 2 right(+x), 3 back(+z), 4 left(-x), 5 down(-y)
```

之前的错误是把 `skybox2` 和 `skybox4` 的左右方向解释反了，生成的输入全景图会在拼接边界出现明显错位。训练前应随机抽几个 `pano_skybox_color_fixed` 检查横向接缝是否连续。

## 6. 快速检查

```bash
find ../dataset/Matterport3D/processed -path '*/pano_skybox_color_fixed/*.png' -type f | wc -l
find ../dataset/Matterport3D/processed -name pano_poses -type d | wc -l
find ../dataset/Matterport3D/processed -name pano_depth -type d | wc -l
```

完整 Matterport3D release 预期为 90 个 scenes；本项目 split JSON 为 train 72 scenes、val 8 scenes、test 10 scenes。
