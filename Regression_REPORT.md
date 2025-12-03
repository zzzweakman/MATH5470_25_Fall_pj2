# Stock CNN 回归任务报告

## 1. 任务概述

本报告记录了基于 CNN 的股票收益率回归预测任务。与分类任务（预测涨跌方向）不同，回归任务直接预测股票未来的收益率数值。

### 1.1 任务对比

| 对比项 | 分类任务 | 回归任务 |
|--------|----------|----------|
| 目标 | 预测涨跌方向 (0/1) | 预测收益率数值 |
| 损失函数 | CrossEntropyLoss | MSELoss |
| 输出层 | 2 个神经元 + Softmax | 1 个神经元 |
| 评估指标 | Accuracy | MSE, MAE, Correlation |

---

## 2. 训练配置

### 2.1 数据集

| 项目 | 说明 |
|------|------|
| 训练数据 | 1993-2000 年 |
| 训练样本数 | 793,019 |
| 测试数据 | 2001-2019 年 |
| 测试样本数 | 1,403,975 |
| 图像尺寸 | 64 × 60 |
| 目标变量 | `Ret_5d` (未来 5 天收益率) |

### 2.2 数据划分

| 划分 | 比例 | 样本数 |
|------|------|--------|
| 训练集 | 70% | ~555,000 |
| 验证集 | 30% | ~238,000 |
| 划分方式 | 时序划分（非随机） | - |

### 2.3 模型架构

**Baseline Regression CNN**

```
输入: 20日K线图 (64 × 60)
     ↓
Conv2d(1→64) + BatchNorm + LeakyReLU + MaxPool
     ↓
Conv2d(64→128) + BatchNorm + LeakyReLU + MaxPool
     ↓
Conv2d(128→256) + BatchNorm + LeakyReLU + MaxPool
     ↓
Flatten (46080)
     ↓
Dropout(0.5) + Linear(46080→512) + LeakyReLU
     ↓
Dropout(0.25) + Linear(512→64) + LeakyReLU
     ↓
Linear(64→1)
     ↓
输出: 预测收益率 (标量)
```

**参数量**: ~0.73M

### 2.4 训练超参数

| 超参数 | 值 |
|--------|-----|
| 批大小 | 256 |
| 优化器 | Adam |
| 学习率 | 1e-4 |
| 损失函数 | MSELoss |
| 最大轮数 | 100 |
| 早停耐心值 | 10 epochs |

---

## 3. 训练结果

### 3.1 评估指标

| 指标 | 说明 |
|------|------|
| MSE | 均方误差，越小越好 |
| RMSE | 均方根误差 |
| MAE | 平均绝对误差 |
| Correlation | 预测值与真实值的相关系数 |
| Direction Accuracy | 预测涨跌方向的准确率 |

### 3.2 训练曲线

训练过程中记录以下指标：
- `Loss/train_mse`: 训练集 MSE
- `Loss/val_mse`: 验证集 MSE
- `Metrics/val_mae`: 验证集 MAE
- `Metrics/val_correlation`: 预测-真实值相关系数

### 3.3 模型保存

```
pt/regression_baseline_YYYYMMDD_HHMMSS/
├── best.pt                           # 最佳模型
├── epoch_X_mse_X_corr_X.pt          # 各 epoch 检查点
└── training_curves.png               # 训练曲线图
```

---

## 4. 回测策略

### 4.1 策略设计

采用与分类任务一致的简化回测逻辑：

**买入条件**: 预测收益率 > threshold（默认 1%）

**策略流程**:
1. 每个交易日，模型预测所有股票的 5 日收益率
2. 选出预测收益率 > 1% 的股票
3. 等权买入选中的股票
4. 持有 5 天后隐含卖出
5. 计算每日选中股票的平均实际收益

### 4.2 收益计算

```python
# 基准收益：每日所有股票的平均 5 日收益
ret_baseline = label_df.groupby('Date')['Ret_5d'].mean()

# 策略收益：每日选中股票的平均 5 日收益
mask = predictions > threshold  # threshold = 0.01 (1%)
ret_strategy = label_df[mask].groupby('Date')['Ret_5d'].mean()

# 累计收益
cum_return = (ret + 1).cumprod()
```

### 4.3 阈值敏感性

测试不同阈值对策略效果的影响：

| 阈值 | 选股比例 | 累计收益 | 超额收益 |
|------|----------|----------|----------|
| 0.5% | 较高 | - | - |
| 1.0% | 中等 | - | - |
| 2.0% | 较低 | - | - |
| 3.0% | 很低 | - | - |
| 5.0% | 极低 | - | - |

---

## 5. 输出文件

### 5.1 训练输出

| 文件 | 说明 |
|------|------|
| `pt/regression_baseline_*/best.pt` | 最佳回归模型 |
| `pt/regression_baseline_*/training_curves.png` | 训练曲线 |
| `runs/` | TensorBoard 日志 |

### 5.2 测试输出

| 文件 | 说明 |
|------|------|
| `pt/test_regression_results.json` | 测试结果 JSON |
| `pic/test_regression_comparison.png` | 回测对比图 |
| `pic/test_regression_thresholds.png` | 阈值敏感性分析图 |

---

## 6. 可视化说明

### 6.1 test_regression_comparison.png

包含 4 个子图：

1. **累计收益对比**: 策略 vs 基准的累计收益曲线
2. **预测 vs 真实散点图**: 评估预测准确性
3. **预测值分布**: 预测收益率的直方图
4. **每日选股数量**: 随时间变化的选股数量

### 6.2 test_regression_thresholds.png

包含 2 个子图：

1. **累计收益 vs 阈值**: 不同阈值下的策略收益
2. **选股比例 vs 阈值**: 不同阈值下的选股比例

---

## 7. 与分类任务的对比

| 对比项 | 分类任务 (test.ipynb) | 回归任务 (test_regression.ipynb) |
|--------|----------------------|----------------------------------|
| 模型输出 | 概率 (0~1) | 收益率预测值 |
| 选股条件 | 概率 > 0.58 | 预测收益 > 1% |
| 持有期 | 20 天 | 5 天 |
| 收益字段 | `Ret_20d` | `Ret_5d` |
| 回测逻辑 | 简化（隐含卖出） | 简化（隐含卖出） |

---

## 8. 复现指南

### 8.1 训练回归模型

```bash
cd notebooks
jupyter notebook train_regression.ipynb
```

配置说明：
- `model_type`: "baseline" 或 "large"
- `target_col`: "Ret_5d" 或 "Ret_20d"

### 8.2 测试与回测

```bash
cd notebooks
jupyter notebook test_regression.ipynb
```

配置说明：
- `model_path`: 训练好的模型路径
- `threshold`: 选股阈值（默认 0.01 即 1%）
- `target_col`: 需要与训练时一致

### 8.3 监控训练

```bash
tensorboard --logdir=runs
```

---

## 9. 注意事项

### 9.1 NaN 处理

`Ret_5d` 字段可能包含 NaN 值（某些股票停牌等原因），代码会自动过滤：

```python
if nan_count > 0:
    valid_mask = ~np.isnan(target_values)
    target_values = target_values[valid_mask]
    images = images[valid_mask]
```

### 9.2 回测局限性

- 未考虑交易成本和滑点
- 假设可以无限分割买入任意股票
- 持仓期重叠问题被简化处理
- 历史回测不代表未来表现

---

*报告生成时间: 2025年12月*

