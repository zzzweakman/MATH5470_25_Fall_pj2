# Stock CNN 训练报告

## 1. 实验概述

本实验基于论文 **(Re-)Imag(in)ing Price Trends** 的方法，使用深度学习模型对股票价格图像进行分类，预测未来 20 天的涨跌方向。我们实现并对比了三种不同架构的模型：**Baseline CNN**、**Baseline Large CNN** 和 **Vision Transformer (ViT)**。

---

## 2. 硬件与环境配置

| 配置项 | 设置 |
|--------|------|
| GPU 数量 | 8 |
| GPU 设备 | CUDA 0-7 |
| 并行策略 | `nn.DataParallel` |
| PyTorch 版本 | 2.6.0 |
| CUDA 版本 | 12.6 |
| Python 版本 | 3.10 |

---

## 3. 数据集配置

| 配置项 | 设置 |
|--------|------|
| 训练+验证数据 | 1993-2000 年 |
| 测试数据 | 2001-2019 年 |
| 总样本数（训练+验证） | 793,019 |
| 图像尺寸 | 64 × 60 (H × W) |
| 标签 | 二分类（Ret_20d > 0） |
| 训练/验证划分比例 | 70% / 30% |
| 训练样本数 | 555,113 |
| 验证样本数 | 237,906 |
| 划分方式 | 时序划分（非随机） |

---

## 4. 模型架构对比

### 4.1 模型参数量

| 模型 | 参数量 | 描述 |
|------|--------|------|
| **Baseline** | 708,866 (0.71M) | 3 层 CNN + FC |
| **Baseline Large** | 10,233,602 (10.23M) | 扩大通道数的 CNN |
| **ViT** | 10,821,314 (10.82M) | Vision Transformer |

### 4.2 Baseline CNN 架构

| 层 | 输出通道 | 卷积核 | 步长 | 膨胀 | 填充 | 激活函数 |
|----|---------|--------|------|------|------|---------|
| Conv2d + BN + LeakyReLU + MaxPool | 64 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Conv2d + BN + LeakyReLU + MaxPool | 128 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Conv2d + BN + LeakyReLU + MaxPool | 256 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Dropout(0.5) + Linear | 2 | - | - | - | - | Softmax |

### 4.3 Baseline Large CNN 架构

与 Baseline 结构相同，但通道数从 64→128→256 扩大到 96→192→384，并增加了隐藏层。

### 4.4 Vision Transformer 架构

| 配置项 | 设置 |
|--------|------|
| Patch Size | 4×4 |
| Embedding Dim | 384 |
| Transformer Depth | 6 |
| Attention Heads | 6 |
| MLP Ratio | 4 |
| Dropout | 0.1 |

---

## 5. 训练超参数

| 超参数 | 设置 |
|--------|------|
| Batch Size | 1024 (128 × 8 GPUs) |
| 每 GPU Batch Size | 128 |
| 优化器 | Adam |
| 学习率 | 1e-5 |
| 损失函数 | CrossEntropyLoss |
| 最大 Epochs | 100 |
| 早停策略 | Patience = 10 epochs |
| 权重初始化 | Xavier Uniform |
| 随机种子 | 42 |

---

## 6. 训练结果汇总

### 6.1 三模型对比

| 模型 | 参数量 | 最佳 Epoch | 最佳 Val Loss | 总训练 Epochs |
|------|--------|------------|---------------|---------------|
| **Baseline** | 0.71M | 18 | 0.687114 | 29 |
| **Baseline Large** | 10.23M | 26 | **0.686382** | 37 |
| **ViT** | 10.82M | 7 | 0.691686 | 18 |

### 6.2 关键发现

1. **Baseline Large 表现最佳**：虽然参数量增加了 14 倍，但 Val Loss 仅略微降低（0.687 → 0.686）。
2. **ViT 表现不佳**：Vision Transformer 在此任务上收敛困难，Val Loss 停滞在 0.692 左右。
3. **Baseline 性价比最高**：仅用 0.71M 参数就达到了接近最佳的效果。

---

## 7. 训练过程详情

### 7.1 Baseline 训练记录

| Epoch | Train Loss | Val Loss | 备注 |
|-------|-----------|----------|------|
| 0 | 1.0229 | 0.7226 | |
| 5 | 0.7600 | 0.6908 | |
| 10 | 0.7260 | 0.7013 | |
| 15 | 0.7098 | 0.6893 | |
| 18 | 0.7032 | **0.6871** | ⭐ Best |
| 28 | 0.6897 | 0.6917 | Early Stop |

### 7.2 Baseline Large 训练记录

| Epoch | Train Loss | Val Loss | 备注 |
|-------|-----------|----------|------|
| 0 | 0.9485 | 0.7182 | |
| 10 | 0.7109 | 0.6896 | |
| 20 | 0.6842 | 0.6882 | |
| 26 | 0.6761 | **0.6864** | ⭐ Best |
| 36 | 0.6727 | 0.6906 | Early Stop |

### 7.3 ViT 训练记录

| Epoch | Train Loss | Val Loss | 备注 |
|-------|-----------|----------|------|
| 0 | 0.7023 | 0.6979 | |
| 7 | 0.6935 | **0.6917** | ⭐ Best |
| 17 | 0.6927 | 0.6928 | Early Stop |

---

## 8. 训练效率统计

| 指标 | Baseline | Baseline Large | ViT |
|------|----------|----------------|-----|
| 每 Epoch 训练步数 | 543 | 543 | 543 |
| 每 Epoch 验证步数 | 233 | 233 | 233 |
| 训练速度 | ~30 it/s | ~25 it/s | ~15 it/s |
| 单 Epoch 时间 | ~25s | ~35s | ~55s |
| 总训练时间 | ~12 min | ~22 min | ~17 min |

---

## 9. 模型保存

| 模型 | 最佳模型路径 |
|------|-------------|
| Baseline | `pt/baseline/best.pt` |
| Baseline Large | `pt/baseline_large/best.pt` |
| ViT | `pt/vit/best.pt` |

---

## 10. 训练曲线

训练曲线对比图保存在：`pic/training_comparison.png`

![Training Comparison](pic/training_comparison.png)

---

## 11. 结论

1. **CNN 架构更适合此任务**：传统 CNN 在股票图像分类任务上表现优于 Vision Transformer。

2. **参数量与性能不成正比**：Baseline Large 参数量是 Baseline 的 14 倍，但 Val Loss 仅降低了 0.1%。

3. **早停策略有效**：所有模型都在合适的时机触发早停，避免了过拟合。

4. **推荐使用 Baseline**：考虑到训练效率和模型大小，Baseline 是最佳选择。

---

## 12. 文件结构

```
Stock_CNN/
├── notebooks/
│   └── train.ipynb              # 训练脚本
├── models/
│   ├── baseline.py              # Baseline CNN
│   ├── baseline_large.py        # Baseline Large CNN
│   └── vit.py                   # Vision Transformer
├── pt/
│   ├── baseline/best.pt         # Baseline 最佳模型
│   ├── baseline_large/best.pt   # Baseline Large 最佳模型
│   ├── vit/best.pt              # ViT 最佳模型
│   └── training_results.json    # 训练结果 JSON
├── runs/                        # TensorBoard 日志
└── pic/
    └── training_comparison.png  # 训练曲线对比图
```

---

*报告生成时间: 2025-12-02*
