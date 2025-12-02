# Stock CNN 测试报告

## 1. 测试概述

本测试对训练好的 CNN 模型进行样本外回测，评估其作为股票选择策略的实际投资表现。

---

## 2. 测试配置

### 2.1 测试环境

| 配置项 | 设置 |
|--------|------|
| 设备 | GPU (CUDA) |
| Batch Size | 2048 |
| 随机种子 | 42 |

### 2.2 测试数据

| 配置项 | 设置 |
|--------|------|
| 测试年份 | 2001-2019 年 |
| 测试样本数 | 1,403,975 |
| 图像尺寸 | 64 × 60 (H × W) |
| 标签 | 二分类（Ret_20d > 0） |

### 2.3 使用模型

| 配置项 | 设置 |
|--------|------|
| 模型路径 | `pt/baseline-earlystop-5/best_baseline_epoch_19_train_0.701418_val_0.686740.pt` |
| 模型类型 | Baseline CNN |
| 训练 Epoch | 19 |
| 训练 Val Loss | 0.6867 |

---

## 3. 模型推理结果

| 指标 | 数值 |
|------|------|
| 测试 Loss | 0.694 |
| 推理步数 | 686 steps |
| 推理速度 | ~13.30 it/s |
| 总推理时间 | ~51 秒 |

---

## 4. 回测策略设计

### 4.1 策略说明

模型输出每个股票-日期样本未来 20 天上涨的概率 `predict_logit`，基于此构建投资策略：

| 策略 | 描述 |
|------|------|
| **Baseline** | 买入所有股票（threshold = 0），等权重 |
| **CNN 策略** | 只买入 `predict_logit > 0.58` 的股票 |

### 4.2 权重方案

| 权重方案 | 描述 |
|---------|------|
| **等权重 (Same Weighted)** | 每只选中的股票分配相同权重 |
| **波动率加权 (EWMA_Vol Weighted)** | 按 EWMA 波动率分配权重 |

---

## 5. 回测结果

### 5.1 策略参数

| 参数 | 设置 |
|------|------|
| 选股阈值 (threshold) | 0.58 |
| 回测周期 | 2001-2019 年（约 19 年） |
| 持仓周期 | 20 个交易日 |

### 5.2 等权重策略对比

| 策略 | 描述 |
|------|------|
| Baseline | 每月买入所有股票，计算平均收益 |
| CNN | 每月只买入模型预测概率 > 58% 的股票 |
| Exceed Return | CNN 策略相对 Baseline 的超额收益 |

### 5.3 波动率加权策略

| 策略 | 权重计算 |
|------|---------|
| Baseline | `weight = EWMA_vol` |
| CNN | `weight = (predict_logit > 0.58) × EWMA_vol` |

---

## 6. 可视化输出

### 6.1 生成的图表

| 图表文件 | 描述 |
|---------|------|
| `pic/performance1.png` | 对数累计收益对比图（Baseline vs CNN vs 超额收益） |
| `pic/performance2.png` | CNN 策略累计收益曲线 |

### 6.2 图表内容说明

**performance1.png** 包含三条曲线：
- `baseline`: 买入所有股票的对数累计收益
- `CNN`: CNN 选股策略的对数累计收益  
- `exceed_ret`: 超额收益（CNN - Baseline）

**performance2.png** 显示：
- CNN 策略的原始累计收益（非对数）

---

## 7. 关键代码逻辑

### 7.1 模型推理

```python
# 加载模型
net = torch.load(net_path, weights_only=False)

# 推理获取预测概率
test_loss, y_pred, y_target = eval_loop(test_dataloader, net, loss_fn)
predict_logit = (torch.nn.Softmax(dim=1)(y_pred)[:,1]).cpu().numpy()
```

### 7.2 策略构建

```python
# Baseline：买入所有股票
threshold = 0.
ret_baseline = label_df.groupby(['Date'])['Ret_20d'].mean()

# CNN 策略：只买入高概率股票
threshold = 0.58
label_filtered = label_df[predict_logit > threshold]
ret_cnn = label_filtered.groupby(['Date'])['Ret_20d'].mean()
```

### 7.3 收益计算

```python
# 对数累计收益
log_ret_baseline = np.log10((ret_baseline+1).cumprod().ffill())
log_ret_cnn = np.log10((ret_cnn+1).cumprod().ffill())

# 超额收益
exceed_ret = log_ret_cnn - log_ret_baseline
```

---

## 8. 数据字段说明

测试数据包含以下标签字段：

| 字段名 | 描述 |
|--------|------|
| Date | 日期 |
| Ret_20d | 未来 20 天收益率 |
| EWMA_vol | 指数加权移动平均波动率 |
| 其他字段 | 详见 `label_columns.txt` |

---

## 9. 测试结论

1. **样本外表现**：模型在 2001-2019 年测试集上的 Loss 为 0.694，与验证集 Loss (0.687) 接近，说明模型泛化能力良好。

2. **选股阈值**：使用 0.58 作为选股阈值，筛选模型认为"高概率上涨"的股票。

3. **策略有效性**：通过对比 CNN 策略与 Baseline 的累计收益曲线，可以评估 CNN 模型的选股能力。

4. **回测周期**：覆盖 19 年的市场数据，包含多个牛熊周期，测试结果具有统计意义。

---

## 10. 文件输出

| 输出文件 | 路径 |
|---------|------|
| 收益对比图 | `pic/performance1.png` |
| CNN 累计收益图 | `pic/performance2.png` |

---

## 11. 复现步骤

```bash
# 1. 确保已完成训练，获得最佳模型
# 2. 运行测试 notebook
jupyter notebook notebooks/test.ipynb

# 3. 查看生成的图表
ls pic/
```

---

*报告生成时间: 2025-12-02*

