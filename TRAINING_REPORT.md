# Stock CNN 训练报告

## 1. 实验概述

本实验基于论文 **(Re-)Imag(in)ing Price Trends** 的方法，使用卷积神经网络（CNN）对股票价格图像进行分类，预测未来 20 天的涨跌方向。

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

## 4. 模型架构

### Baseline CNN

| 层 | 输出通道 | 卷积核 | 步长 | 膨胀 | 填充 | 激活函数 |
|----|---------|--------|------|------|------|---------|
| Conv2d + BN + LeakyReLU + MaxPool | 64 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Conv2d + BN + LeakyReLU + MaxPool | 128 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Conv2d + BN + LeakyReLU + MaxPool | 256 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Dropout + Linear | 2 | - | - | - | - | Softmax |

| 模型统计 | 数值 |
|---------|------|
| 总参数量 | 708,866 (0.71M) |
| FLOPs (单样本) | 289.76 GFLOPs |

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
| 权重初始化 | Xavier Uniform |
| 随机种子 | 42 |

---

## 6. 早停策略对比实验

### 实验设置

| 实验名称 | Early Stopping Patience |
|---------|------------------------|
| baseline-earlystop-5 | 5 epochs |
| baseline-earlystop-10 | 10 epochs |

### 训练结果对比

| 实验 | 最佳 Epoch | 最佳 Train Loss | 最佳 Val Loss | 总训练 Epochs |
|------|-----------|-----------------|---------------|---------------|
| earlystop-5 | 19 | 0.7014 | **0.6867** | 25 |
| earlystop-10 | 24 | 0.6939 | 0.6873 | 30 |

---

## 7. 训练过程详情 (earlystop-10)

### 逐 Epoch 训练记录

| Epoch | Train Loss | Val Loss | 训练时间 | 验证时间 | 备注 |
|-------|-----------|----------|---------|---------|------|
| 0 | 1.0283 | 0.7169 | ~24s | ~7s | |
| 1 | 0.8808 | 0.7014 | ~21s | ~6s | |
| 2 | 0.8233 | 0.6958 | ~18s | ~6s | |
| 3 | 0.7951 | 0.6938 | ~18s | ~6s | |
| 4 | 0.7750 | 0.6955 | ~18s | ~6s | |
| 5 | 0.7629 | 0.6926 | ~18s | ~6s | |
| 6 | 0.7521 | 0.7139 | ~18s | ~6s | |
| 7 | 0.7446 | 0.6990 | ~17s | ~6s | |
| 8 | 0.7372 | 0.6919 | ~17s | ~6s | |
| 9 | 0.7309 | 0.6902 | ~18s | ~6s | |
| 10 | 0.7266 | 0.6900 | ~18s | ~6s | |
| 11 | 0.7227 | 0.6921 | ~17s | ~6s | |
| 12 | 0.7185 | 0.6976 | ~18s | ~6s | |
| 13 | 0.7152 | 0.6899 | ~17s | ~5s | |
| 14 | 0.7121 | 0.6947 | ~17s | ~6s | |
| 15 | 0.7098 | 0.6961 | ~39s | ~6s | |
| 16 | 0.7077 | 0.6883 | ~18s | ~6s | |
| 17 | 0.7051 | 0.6874 | ~17s | ~6s | |
| 18 | 0.7028 | 0.6898 | ~18s | ~6s | |
| 19 | 0.7014 | **0.6867** | ~17s | ~6s | ⭐ Best |
| 20 | 0.6994 | 0.6928 | ~17s | ~5s | |
| 21 | 0.6975 | 0.6949 | ~17s | ~6s | |
| 22 | 0.6961 | 0.6881 | ~18s | ~6s | |
| 23 | 0.6949 | 0.6903 | ~18s | ~6s | |
| 24 | 0.6939 | 0.6873 | ~18s | ~6s | |
| 25 | 0.6925 | 0.7009 | ~18s | ~6s | |
| 26 | 0.6915 | 0.6887 | ~18s | ~5s | |
| 27 | 0.6901 | 0.6931 | ~17s | ~6s | |
| 28 | 0.6893 | 0.6874 | ~17s | ~6s | |
| 29 | 0.6881 | 0.6882 | ~17s | ~6s | Early Stop |

---

## 8. 训练效率统计

| 指标 | 数值 |
|------|------|
| 每 Epoch 训练步数 | 543 steps |
| 每 Epoch 验证步数 | 233 steps |
| 训练速度 | ~29-30 it/s |
| 验证速度 | ~35-36 it/s |
| 单 Epoch 训练时间 | ~17-24 秒 |
| 单 Epoch 验证时间 | ~6-7 秒 |
| 总训练时间 (30 epochs) | ~12 分钟 |

---

## 9. 最佳模型

| 项目 | 值 |
|------|-----|
| 最佳模型路径 | `pt/baseline-earlystop-5/best_baseline_epoch_19_train_0.701418_val_0.686740.pt` |
| 最佳 Epoch | 19 |
| Train Loss | 0.7014 |
| Val Loss | 0.6867 |

---

## 10. 关键观察

1. **过拟合趋势**：Train Loss 持续下降，但 Val Loss 在 Epoch 19 后趋于平稳并有轻微波动，表明模型开始过拟合。

2. **早停效果**：Early Stopping Patience=5 和 10 的最终效果相近，但 Patience=5 可以更早结束训练，节省计算资源。

3. **收敛速度**：模型在前 10 个 epoch 快速收敛，之后进入缓慢优化阶段。

4. **GPU 利用**：使用 8 GPU DataParallel，有效 Batch Size 为 1024，显著加速了训练过程。

---

## 11. 文件结构

```
Stock_CNN/
├── notebooks/
│   ├── train.ipynb          # 训练脚本
│   └── test.ipynb           # 测试脚本
├── models/
│   ├── baseline.py          # Baseline CNN 模型
│   ├── baseline_large.py    # 大型 CNN 模型
│   └── vit.py               # Vision Transformer 模型
├── pt/
│   ├── baseline-earlystop-5/   # Patience=5 实验结果
│   └── baseline-earlystop-10/  # Patience=10 实验结果
├── data/
│   └── monthly_20d/         # 数据目录
└── runs/                    # TensorBoard 日志
```

---

## 12. 复现命令

```bash
# 1. 创建环境
conda create -n Math5470 python=3.10
conda activate Math5470
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. 启动 TensorBoard 监控
tensorboard --logdir=runs

# 3. 运行训练 notebook
jupyter notebook notebooks/train.ipynb
```

---

*报告生成时间: 2025-12-02*

