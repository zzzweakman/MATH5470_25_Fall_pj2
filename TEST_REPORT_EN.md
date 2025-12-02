# Stock CNN Test Report

## 1. Test Overview

This test performs out-of-sample backtesting on the trained CNN model to evaluate its performance as a stock selection strategy for actual investment.

---

## 2. Test Configuration

### 2.1 Test Environment

| Configuration | Setting |
|---------------|---------|
| Device | GPU (CUDA) |
| Batch Size | 2048 |
| Random Seed | 42 |

### 2.2 Test Data

| Configuration | Setting |
|---------------|---------|
| Test Years | 2001-2019 |
| Test Samples | 1,403,975 |
| Image Size | 64 × 60 (H × W) |
| Label | Binary Classification (Ret_20d > 0) |

### 2.3 Model Used

| Configuration | Setting |
|---------------|---------|
| Model Path | `pt/baseline-earlystop-5/best_baseline_epoch_19_train_0.701418_val_0.686740.pt` |
| Model Type | Baseline CNN |
| Training Epoch | 19 |
| Training Val Loss | 0.6867 |

---

## 3. Model Inference Results

| Metric | Value |
|--------|-------|
| Test Loss | 0.694 |
| Inference Steps | 686 steps |
| Inference Speed | ~13.30 it/s |
| Total Inference Time | ~51 seconds |

---

## 4. Backtesting Strategy Design

### 4.1 Strategy Description

The model outputs the probability of each stock-date sample rising in the next 20 days (`predict_logit`), based on which investment strategies are constructed:

| Strategy | Description |
|----------|-------------|
| **Baseline** | Buy all stocks (threshold = 0), equal weight |
| **CNN Strategy** | Only buy stocks with `predict_logit > 0.58` |

### 4.2 Weighting Schemes

| Weighting Scheme | Description |
|------------------|-------------|
| **Equal Weighted (Same Weighted)** | Allocate equal weight to each selected stock |
| **Volatility Weighted (EWMA_Vol Weighted)** | Allocate weight based on EWMA volatility |

---

## 5. Backtesting Results

### 5.1 Strategy Parameters

| Parameter | Setting |
|-----------|---------|
| Stock Selection Threshold | 0.58 |
| Backtesting Period | 2001-2019 (~19 years) |
| Holding Period | 20 trading days |

### 5.2 Equal Weight Strategy Comparison

| Strategy | Description |
|----------|-------------|
| Baseline | Buy all stocks monthly, calculate average return |
| CNN | Only buy stocks with model prediction probability > 58% |
| Exceed Return | Excess return of CNN strategy over Baseline |

### 5.3 Volatility Weighted Strategy

| Strategy | Weight Calculation |
|----------|-------------------|
| Baseline | `weight = EWMA_vol` |
| CNN | `weight = (predict_logit > 0.58) × EWMA_vol` |

---

## 6. Visualization Output

### 6.1 Generated Charts

| Chart File | Description |
|------------|-------------|
| `pic/performance1.png` | Log cumulative return comparison (Baseline vs CNN vs Excess Return) |
| `pic/performance2.png` | CNN strategy cumulative return curve |

### 6.2 Chart Content Description

**performance1.png** contains three curves:
- `baseline`: Log cumulative return of buying all stocks
- `CNN`: Log cumulative return of CNN stock selection strategy
- `exceed_ret`: Excess return (CNN - Baseline)

**performance2.png** shows:
- Raw cumulative return of CNN strategy (non-logarithmic)

---

## 7. Key Code Logic

### 7.1 Model Inference

```python
# Load model
net = torch.load(net_path, weights_only=False)

# Inference to get prediction probabilities
test_loss, y_pred, y_target = eval_loop(test_dataloader, net, loss_fn)
predict_logit = (torch.nn.Softmax(dim=1)(y_pred)[:,1]).cpu().numpy()
```

### 7.2 Strategy Construction

```python
# Baseline: Buy all stocks
threshold = 0.
ret_baseline = label_df.groupby(['Date'])['Ret_20d'].mean()

# CNN Strategy: Only buy high probability stocks
threshold = 0.58
label_filtered = label_df[predict_logit > threshold]
ret_cnn = label_filtered.groupby(['Date'])['Ret_20d'].mean()
```

### 7.3 Return Calculation

```python
# Log cumulative return
log_ret_baseline = np.log10((ret_baseline+1).cumprod().ffill())
log_ret_cnn = np.log10((ret_cnn+1).cumprod().ffill())

# Excess return
exceed_ret = log_ret_cnn - log_ret_baseline
```

---

## 8. Data Field Description

The test data contains the following label fields:

| Field Name | Description |
|------------|-------------|
| Date | Date |
| Ret_20d | Return over next 20 days |
| EWMA_vol | Exponentially Weighted Moving Average Volatility |
| Other fields | See `label_columns.txt` for details |

---

## 9. Test Conclusions

1. **Out-of-Sample Performance**: The model achieves a Loss of 0.694 on the 2001-2019 test set, close to the validation Loss (0.687), indicating good generalization ability.

2. **Stock Selection Threshold**: Using 0.58 as the selection threshold to filter stocks that the model considers "high probability of rising".

3. **Strategy Effectiveness**: By comparing the cumulative return curves of CNN strategy and Baseline, we can evaluate the stock selection capability of the CNN model.

4. **Backtesting Period**: Covering 19 years of market data, including multiple bull and bear cycles, the test results have statistical significance.

---

## 10. Output Files

| Output File | Path |
|-------------|------|
| Return Comparison Chart | `pic/performance1.png` |
| CNN Cumulative Return Chart | `pic/performance2.png` |

---

## 11. Reproduction Steps

```bash
# 1. Ensure training is completed and best model is obtained
# 2. Run test notebook
jupyter notebook notebooks/test.ipynb

# 3. View generated charts
ls pic/
```

---

*Report Generated: 2025-12-02*

