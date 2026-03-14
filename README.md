# AU-MIPGAN: Aleatoric Uncertainty-aware MIP-based GAN

基于论文《Aleatoric Uncertainty-Aware Maximum Intensity Projection-Based GAN for 7T-Like Generation From 3T TOF-MRA》的PyTorch实现。

## 项目概述

本项目实现了一个用于从3T TOF-MRA图像生成7T-like图像的深度学习模型。模型采用两阶段训练策略：
1. **Stage 1**: 训练三个教师模型，分别学习轴向、冠状和矢状方向的MIP特征
2. **Stage 2**: 通过知识蒸馏训练学生模型，融合多方向MIP信息

## 环境配置

### 系统要求
- Python 3.7+
- CUDA 10.2+ (推荐)
- GPU内存: 建议24GB以上 (支持多GPU训练)

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- torch >= 1.9.0
- nibabel >= 3.2.0 (NIfTI文件处理)
- scipy >= 1.5.0
- tensorboard >= 2.5.0
- tqdm >= 4.50.0

---

## 数据准备

### 1. 数据格式要求

支持的数据格式：
- NIfTI格式 (`.nii` 或 `.nii.gz`) - 推荐格式
- NRRD格式 (`.nrrd`) - 原始数据格式，需要预处理转换

### 2. 使用预处理脚本 (推荐)

本项目提供了完整的数据预处理脚本，可以自动处理NRRD格式的原始数据。

#### 2.1 预处理脚本功能

- 读取NRRD格式的3T和7T TOF-MRA图像
- 可选的刚性配准 (7T配准到3T)
- 可选的偏置场校正 (N4ITK)
- 可选的颅骨去除
- 重采样到统一大小 (默认512×512×320)
- 强度归一化到 [0, 255]
- 自动划分训练/验证/测试集
- 输出为NIfTI格式

#### 2.2 运行预处理

```bash
# 基本用法 (跳过配准和偏置场校正，速度较快)
python scripts/preprocess_data.py --input_dir origin_nrrd_data --output_dir data

# 完整预处理 (包含配准和偏置场校正)
python scripts/preprocess_data.py --input_dir origin_nrrd_data --output_dir data

# 自定义参数
python scripts/preprocess_data.py \
    --input_dir origin_nrrd_data \
    --output_dir data \
    --target_shape 512 512 320 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --test_ratio 0.2
```

#### 2.3 预处理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | origin_nrrd_data | 输入NRRD数据目录 |
| `--output_dir` | data | 输出目录 |
| `--target_shape` | 512 512 320 | 重采样目标大小 |
| `--no_registration` | - | 跳过刚性配准 |
| `--no_bias_correction` | - | 跳过偏置场校正 |
| `--skull_stripping` | - | 启用颅骨去除 |
| `--train_ratio` | 0.7 | 训练集比例 |
| `--val_ratio` | 0.1 | 验证集比例 |
| `--test_ratio` | 0.2 | 测试集比例 |
| `--seed` | 42 | 随机种子 |

#### 2.4 原始数据目录结构要求

```
origin_nrrd_data/
├── 101/
│   ├── subject_name_3T.nrrd
│   └── subject_name_7T.nrrd
├── 102/
│   ├── subject_name_3T.nrrd
│   └── subject_name_7T.nrrd
└── ...
```

**注意**：文件名必须以 `_3T.nrrd` 或 `_7T.nrrd` 结尾（不区分大小写）。

### 3. 手动预处理 (可选)

如果需要手动进行预处理，请参考以下步骤：

#### 3.1 图像配准
将7T图像刚性配准到对应的3T图像：
```python
# 推荐使用ANTs或FSL进行配准
# 示例使用ANTs:
antsRegistrationSyN.sh -d 3 -f 3T_image.nii -m 7T_image.nii -o output_
```

#### 2.2 偏置场校正
对图像进行偏置场校正：
```python
# 使用N4ITK (ITK/SimpleITK)
import SimpleITK as sitk

input_image = sitk.ReadImage("image.nii")
corrector = sitk.N4BiasFieldCorrectionImageFilter()
output_image = corrector.Execute(input_image)
sitk.WriteImage(output_image, "corrected_image.nii")
```

#### 2.3 颅骨去除
移除颅骨和非脑组织：
```python
# 推荐使用HD-BET或FSL BET
# 使用HD-BET:
# bet input.nii output.nii
```

#### 2.4 重采样和裁剪
统一图像矩阵大小以减少计算成本：
```python
import nibabel as nib
from scipy.ndimage import zoom

# 目标矩阵大小: 512 × 512 × 320
target_shape = (512, 512, 320)

img = nib.load("image.nii")
data = img.get_fdata()

# 计算缩放因子
zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
resampled = zoom(data, zoom_factors, order=1)
```

#### 2.5 强度归一化
将图像强度归一化到 [0, 255] 范围：
```python
import numpy as np

def normalize(data):
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min > 0:
        data = (data - data_min) / (data_max - data_min) * 255.0
    return data.astype(np.float32)
```

### 3. 数据目录结构

将预处理后的数据按以下结构组织：

```
data/
├── train/
│   ├── 3T/
│   │   ├── subject_001.nii.gz
│   │   ├── subject_002.nii.gz
│   │   └── ...
│   └── 7T/
│       ├── subject_001.nii.gz
│       ├── subject_002.nii.gz
│       └── ...
├── val/
│   ├── 3T/
│   │   ├── subject_033.nii.gz
│   │   └── ...
│   └── 7T/
│       ├── subject_033.nii.gz
│       └── ...
└── test/
    ├── 3T/
    │   ├── subject_041.nii.gz
    │   └── ...
    └── 7T/
        ├── subject_041.nii.gz
        └── ...
```

**重要说明**：
- 3T和7T目录中的文件名必须一一对应
- 同一受试者的3T和7T图像文件名必须相同
- 训练集、验证集、测试集的受试者不应重叠

### 4. 数据集划分建议

参考论文的数据划分：
- 训练集: 32对 3T-7T 图像
- 验证集: 8对 3T-7T 图像
- 测试集: 20对 3T-7T 图像

### 5. 创建测试数据 (可选)

如果需要测试代码，可以创建随机数据：

```bash
python setup.py --create_data
```

---

## 模型训练

### 训练配置

在 `configs/config.py` 中可以修改训练参数：

```python
# 关键参数
patch_size = 64          # 3D patch大小
stride = 32              # patch滑动步长
mip_thickness = 50       # MIP厚度 (voxels)
batch_size = 24          # 批大小
num_epochs = 160         # 训练轮数
learning_rate = 0.0002   # 学习率
lr_decay_start = 80      # 学习率衰减起始轮数

# 损失函数权重
alpha = 10.0             # MAE/AU-AE损失权重
beta = 1.0               # 重建损失权重
gamma = 1.0              # KD损失权重
```

### Stage 1: 训练教师模型

教师模型分别学习三个MIP方向（轴向、冠状、矢状）的3T到7T映射。

#### 训练轴向方向教师模型

```bash
python main.py train_teacher --direction axial
```

#### 训练冠状方向教师模型

```bash
python main.py train_teacher --direction coronal
```

#### 训练矢状方向教师模型

```bash
python main.py train_teacher --direction sagittal
```

#### 并行训练 (多GPU)

如果有多个GPU，可以并行训练三个教师模型：

```bash
# 终端1
python main.py train_teacher --direction axial --gpu 0

# 终端2
python main.py train_teacher --direction coronal --gpu 1

# 终端3
python main.py train_teacher --direction sagittal --gpu 2
```

#### 指定单个GPU训练

```bash
# 使用GPU 0训练轴向方向
python main.py train_teacher --direction axial --gpu 0

# 使用GPU 1训练冠状方向
python main.py train_teacher --direction coronal --gpu 1
```

#### 教师模型检查点

训练完成后，检查点保存在：
```
checkpoints/
├── teacher_axial/
│   ├── best_model.pth
│   └── latest_model.pth
├── teacher_coronal/
│   ├── best_model.pth
│   └── latest_model.pth
└── teacher_sagittal/
    ├── best_model.pth
    └── latest_model.pth
```

### Stage 2: 训练学生模型

学生模型通过知识蒸馏融合三个教师模型的知识。

#### 使用默认检查点路径

```bash
python main.py train_student --gpu 0
```

#### 指定教师模型检查点

```bash
python main.py train_student \
    --teacher_axial checkpoints/teacher_axial/best_model.pth \
    --teacher_coronal checkpoints/teacher_coronal/best_model.pth \
    --teacher_sagittal checkpoints/teacher_sagittal/best_model.pth \
    --gpu 0
```

#### 学生模型检查点

训练完成后，检查点保存在：
```
checkpoints/
└── student/
    ├── best_model.pth
    └── latest_model.pth
```

### 完整训练流程

```bash
# 1. 准备数据 (将数据放入data目录)

# 2. Stage 1: 训练三个教师模型
python main.py train_teacher --direction axial --gpu 0
python main.py train_teacher --direction coronal --gpu 1
python main.py train_teacher --direction sagittal --gpu 2

# 3. Stage 2: 训练学生模型
python main.py train_student --gpu 0
```

### 训练监控

使用TensorBoard监控训练过程：

```bash
# 查看教师模型训练日志
tensorboard --logdir logs/teacher_axial
tensorboard --logdir logs/teacher_coronal
tensorboard --logdir logs/teacher_sagittal

# 查看学生模型训练日志
tensorboard --logdir logs/student
```

---

## 模型推理

### 单文件推理

```bash
python main.py inference \
    --input path/to/3T_image.nii.gz \
    --output path/to/output_7t_like.nii.gz \
    --checkpoint checkpoints/student/best_model.pth \
    --mode file \
    --gpu 0
```

### 批量推理

```bash
python main.py inference \
    --input path/to/input_directory/ \
    --output path/to/output_directory/ \
    --checkpoint checkpoints/student/best_model.pth \
    --mode dir \
    --gpu 0
```

### 推理输出

推理会生成两个文件：
- `*_7t_like.nii.gz`: 生成的7T-like图像
- `*_uncertainty.nii.gz`: 不确定性图 (表示每个体素的置信度)

---

## 模型评估

### 评估指标

#### 图像质量指标
- **BBC (Blood-to-Background Contrast)**: 血管-背景对比度
- **CNR (Contrast-to-Noise Ratio)**: 对比度噪声比
- **SNR (Signal-to-Noise Ratio)**: 信噪比

#### 血管形态学指标
- 小血管总体积偏移
- 小血管总表面积偏移
- 全脑血管总长度偏移
- 全脑血管分支数偏移
- 最小血管平均半径偏移
- 最大血管平均半径偏移

### 评估脚本示例

```python
import numpy as np
import nibabel as nib

def calculate_mae(pred_path, gt_path):
    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    return np.mean(np.abs(pred - gt))

def calculate_psnr(pred_path, gt_path):
    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    mse = np.mean((pred - gt) ** 2)
    max_val = max(pred.max(), gt.max())
    return 20 * np.log10(max_val / np.sqrt(mse))

# 使用示例
mae = calculate_mae('output_7t_like.nii.gz', 'ground_truth_7t.nii.gz')
psnr = calculate_psnr('output_7t_like.nii.gz', 'ground_truth_7t.nii.gz')
print(f"MAE: {mae:.4f}, PSNR: {psnr:.2f} dB")
```

---

## 项目结构

```
TOF_3T_2_7T_new/
├── configs/
│   ├── __init__.py
│   └── config.py              # 配置文件
├── models/
│   ├── __init__.py
│   ├── generator.py           # 3D U-Net生成器
│   ├── discriminator.py       # 3D判别器
│   └── amalgamation.py        # 特征融合模块
├── utils/
│   ├── __init__.py
│   ├── mip_ops.py             # MIP操作
│   └── losses.py              # 损失函数
├── data/
│   ├── __init__.py
│   └── dataset.py             # 数据加载
├── scripts/
│   ├── __init__.py
│   ├── train_teacher.py       # 教师模型训练
│   ├── train_student.py       # 学生模型训练
│   └── inference.py           # 推理脚本
├── checkpoints/               # 模型检查点
├── logs/                      # TensorBoard日志
├── results/                   # 推理结果
├── main.py                    # 主入口
├── setup.py                   # 设置脚本
├── requirements.txt           # 依赖包
└── README.md                  # 说明文档
```

---

## 常见问题

### Q1: GPU内存不足
- 减小 `batch_size`
- 减小 `patch_size`
- 使用梯度累积

### Q2: 训练不稳定
- 检查学习率设置
- 确保数据归一化正确
- 调整损失函数权重

### Q3: 生成图像质量差
- 检查数据配准质量
- 增加训练轮数
- 调整MIP厚度参数

---

## 参考文献

```bibtex
@article{tan2025aumipgan,
  title={Aleatoric Uncertainty-Aware Maximum Intensity Projection-Based GAN for 7T-Like Generation From 3T TOF-MRA},
  author={Tan, et al.},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={29},
  number={9},
  pages={6665--6678},
  year={2025}
}
```

---

## 许可证

本项目仅供学术研究使用。
