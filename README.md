# 集装箱门把手盖缺陷检测与分割项目

<div align="center">

![Task](https://img.shields.io/badge/Task-Detection%20%26%20Segmentation-blue)
![License](https://img.shields.io/badge/LICENSE-Apache%202.0-green)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)
![Framework](https://img.shields.io/badge/Framework-Detectron2-brightgreen)

**集装箱门把手盖缺陷检测与实例分割项目**

</div>

---

## 项目概述

本项目面向集装箱门把手盖表面缺陷检测与实例分割，提供训练、评估、推理和导出流程。项目结合 RT-DETR 风格检测头与 Detectron2 评估能力，可用于目标检测和像素级缺陷分割任务。

### 主要特性

- 同时支持目标检测和实例分割
- 提供像素级分割掩模输出
- 基于 YAML 的配置系统，便于快速调整
- 集成 Detectron2 评估流程
- 支持多种骨干网络和模型配置
- 包含推理、导出和可视化脚本

---

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- torchvision
- CUDA 10.2+（推荐 GPU 环境）
- Detectron2

### 安装步骤

1. 创建 Conda 环境

```bash
conda create -n catch python=3.11 -y
conda activate catch
```

2. 安装 Detectron2 与项目依赖

```bash
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
cd ..
pip install -r requirements.txt
```

3. 验证安装

```bash
python -c "import detectron2; print(detectron2.__version__)"
python -c "import torch; print(torch.__version__)"
```

> Detectron2 在 Windows 上安装时可能需要额外配置，可参考 [Detectron2 安装文档](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)。

---

## 核心功能

### 目标检测

识别门把手盖上的缺陷位置，输出边界框与置信度。

### 实例分割

在检测基础上输出每个缺陷的像素级掩模。相关概念可参考 [Detectron2 模型文档](https://detectron2.readthedocs.io/en/latest/tutorials/models.html)。

### 评估指标

项目可使用 Detectron2 评估器统计以下指标：

- AP
- AP50 / AP75
- mAP
- AR
- Mask AP

---

## 项目结构

```text
Container door handle catch/
├── train.py
├── requirements.txt
├── configs/
│   ├── base/
│   ├── dataset/
│   ├── catch_dfine/
│   ├── catch_rtdetrv2/
│   └── catchv2/
├── engine/
│   ├── backbone/
│   ├── catch/
│   ├── core/
│   ├── data/
│   ├── misc/
│   ├── optim/
│   └── solver/
├── tools/
│   ├── benchmark/
│   ├── dataset/
│   ├── deployment/
│   ├── inference/
│   ├── reference/
│   └── visualization/
└── figures/
```

---

## 骨干网络准备

### 自动下载

以下骨干网络通常可在训练时自动下载：

- HGNetv2 系列
- ResNet 系列

### 需要手动下载

#### DINOv3 骨干网络

如果使用 `catchv2-L` 或 `catchv2-X`，需要手动准备 DINOv3 权重：

- `dinov3_vits16.pth`
- `dinov3_vitb14.pth`（可选）

参考仓库：[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

#### ViT-Tiny 权重

如果使用 `catchv2-S` 或 `catchv2-M`，需要准备蒸馏版 ViT-Tiny 权重：

- [ViT-Tiny](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing)
- [ViT-Tiny+](https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVgEOMdhpuVcG6Jt/view?usp=sharing)

### 权重目录示例

```text
ckpts/
├── dinov3_vits16.pth
├── dinov3_vitb14.pth
├── vitt_distill.pt
└── vittplus_distill.pt
```

配置示例：

```yaml
backbone:
  type: DINOv3STAs
  model_name: vits
  pretrained: true
  ckpt_path: ./ckpts/dinov3_vits16.pth
```

### 模型配置位置

完整模型配置可在以下目录中查看：

- `configs/catch_dfine/`
- `configs/catchv2/`
- `configs/catch_rtdetrv2/`

---

## 配置说明

常用参数如下：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `num_classes` | 检测类别数 | `1` |
| `batch_size` | 训练批大小 | `4-32` |
| `max_epochs` | 最大训练轮数 | `60` |
| `lr` | 初始学习率 | `0.0001` |
| `img_size` | 输入图像大小 | `640` |
| `task` | 任务类型 | `detection` / `segmentation` |

多任务配置示例：

```yaml
model:
  backbone: resnet50
  roi_heads:
    num_classes: 1
    mask_on: true

test:
  evaluator:
    type: detectron2
    metrics: ["bbox", "segm"]
```

---

## 数据准备

### 支持格式

- COCO 格式
- 按 COCO 组织的自定义数据集

### COCO2017 目录示例

```text
/path/to/COCO2017/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

配置示例：

```yaml
train_dataloader:
  img_folder: /path/to/COCO2017/train2017/
  ann_file: /path/to/COCO2017/annotations/instances_train2017.json

val_dataloader:
  img_folder: /path/to/COCO2017/val2017/
  ann_file: /path/to/COCO2017/annotations/instances_val2017.json
```

### 自定义数据集

如果使用自定义数据集，建议按 COCO 结构组织：

```text
custom_dataset/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

常用配置示例：

```yaml
task: detection

evaluator:
  type: Detectron2CocoEvaluator
  iou_types: ["bbox", "segm"]

num_classes: 1
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /path/to/custom_dataset/images/train
    ann_file: /path/to/custom_dataset/annotations/instances_train.json
    return_masks: true

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /path/to/custom_dataset/images/val
    ann_file: /path/to/custom_dataset/annotations/instances_val.json
    return_masks: true
```

---

## 训练与评估

### 基础训练

```bash
python train.py --config configs/catch_dfine/catch_hgnetv2_s_coco.yml

python train.py --config configs/catch_dfine/catch_hgnetv2_m_coco.yml \
  --output-dir ./outputs/experiment_001 \
  --device cuda:0
```

### 启用分割任务

```bash
python train.py --config configs/catch_dfine/catch_hgnetv2_m_coco.yml \
  --update model.roi_heads.mask_on=true \
  --update task=instance_segmentation
```

### 微调预训练模型

```bash
python train.py --config configs/catch_dfine/catch_hgnetv2_m_coco.yml \
  --resume ./checkpoints/model_pretrained.pth \
  --update max_epochs=20 \
  --update lr=0.00005
```

### 评估

```bash
python -m detectron2.tools.train_net \
  --config-file configs/catch_dfine/config.yaml \
  --eval-only \
  MODEL.WEIGHTS outputs/model_final.pth
```

示例输出：

```text
Average Precision  (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = 0.850
Average Precision  (AP) @[ IoU=0.50      | area=all | maxDets=100 ] = 0.920
Average Precision  (AP) @[ IoU=0.75      | area=all | maxDets=100 ] = 0.890
Average Recall     (AR) @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = 0.870
Mask Average Precision (mAP) = 0.820
```

### 分布式训练

```bash
python -m torch.distributed.launch --nproc_per_node 4 \
  train.py --config configs/catch_dfine/catch_hgnetv2_l_coco.yml \
  --distributed
```

---

## 训练监控

### TensorBoard

```bash
tensorboard --logdir=outputs/logs --port=6006
```

### 输出目录示例

```text
outputs/
├── model_final.pth
├── model_best.pth
├── log.txt
├── metrics.json
└── events.out.tfevents
```

---

## 推理与导出

### PyTorch 推理

```bash
python tools/inference/torch_inf.py
python tools/inference/torch_inf_vis.py
```

### ONNX 推理

```bash
python tools/inference/onnx_inf.py
```

### TensorRT 推理

```bash
python tools/inference/trt_inf.py
```

### OpenVINO 推理

```bash
python tools/inference/openvino_inf.py
```

### 模型导出

```bash
python tools/deployment/export_onnx.py
python tools/deployment/export_yolo_w_nms.py
```

---

## 可视化

当前仓库中的可视化脚本位于：

```bash
python tools/visualization/fiftyone_vis.py
```

---

## 关键依赖

| 库 | 用途 |
| --- | --- |
| Detectron2 | 检测和分割框架 |
| PyTorch | 深度学习框架 |
| torchvision | 计算机视觉工具 |
| OpenCV | 图像处理 |
| PyYAML | 配置管理 |
| TensorBoard | 训练监控 |

---

## 快速参考

```bash
# 训练
python train.py --config configs/catch_dfine/catch_hgnetv2_m_coco.yml

# 推理
python tools/inference/torch_inf.py

# 导出
python tools/deployment/export_onnx.py

# 训练监控
tensorboard --logdir=outputs/logs --port=6006
```

---

## 许可证

本项目采用 Apache 2.0 许可证。
