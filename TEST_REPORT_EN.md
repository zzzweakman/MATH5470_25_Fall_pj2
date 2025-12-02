# Stock CNN Test Report

## 1. Test Overview

This report documents the performance of three models (Baseline CNN, Baseline Large CNN, Vision Transformer) on the test set, as well as backtesting results of stock selection strategies based on model predictions.

---

## 2. Test Configuration

### 2.1 Test Data
| Item | Description |
|------|-------------|
| Time Range | 2001 - 2019 |
| Sample Count | 1,403,975 |
| Image Size | 64 × 60 |
| Test Batch Size | 2,048 |

### 2.2 Models Tested
| Model | Model Path | Selection Threshold |
|-------|------------|---------------------|
| Baseline | `pt/baseline/best.pt` | 0.58 |
| Baseline Large | `pt/baseline_large/best.pt` | 0.58 |
| ViT | `pt/vit/best.pt` | 0.50 |

> **Note**: ViT uses a lower threshold (0.50) because its prediction probability distribution differs from CNN models.

---

## 3. Model Inference Results

### 3.1 Test Metrics Comparison
| Model | Parameters | Test Loss | Accuracy | Selection Ratio |
|-------|------------|-----------|----------|-----------------|
| **Baseline** | 0.71M | 0.6942 | 51.58% | 1.85% |
| **Baseline Large** | 10.23M | **0.6926** | **52.10%** | 4.47% |
| **ViT** | 10.82M | 0.6935 | 49.84% | 23.27% |

### 3.2 Metrics Explanation
- **Test Loss**: CrossEntropyLoss, lower is better
- **Accuracy**: Proportion of correctly predicted samples
- **Selection Ratio**: Percentage of stocks with prediction probability above threshold

---

## 4. Backtesting Strategy Design

### 4.1 Strategy Description

#### Baseline Strategy (Buy All)
- Equal-weight purchase of all stocks each trading day
- Hold for 20 trading days then sell
- Represents overall market performance

#### CNN Stock Selection Strategy
- Only buy stocks with model prediction probability > threshold
- Equal-weight allocation among selected stocks
- Hold for 20 trading days then sell

### 4.2 Return Calculation
```python
# Baseline return: Average daily return of all stocks
ret_baseline = label_df.groupby(['Date'])['Ret_20d'].mean()

# Strategy return: Average daily return of selected stocks
ret_strategy = label_filtered.groupby(['Date'])['Ret_20d'].mean()

# Cumulative return
cum_return = (ret + 1).cumprod()
```

---

## 5. Backtesting Results

### 5.1 Cumulative Return Comparison
| Model | Cumulative Return | Excess Return | Annualized Return |
|-------|-------------------|---------------|-------------------|
| **Baseline** | **19.86x** | **+16.68** | ~17.5% |
| **Baseline Large** | 17.11x | +13.93 | ~16.8% |
| **ViT** | 2.84x | -0.34 | ~5.7% |
| Buy All (Benchmark) | 3.18x | 0 | ~6.3% |

### 5.2 Key Findings

1. 🏆 **Baseline CNN performs best**
   - Cumulative return of 19.86x, 6.2 times the benchmark
   - Selecting only 1.85% of stocks achieves superior returns

2. 📊 **Baseline Large is second best**
   - Cumulative return of 17.11x
   - Higher selection ratio (4.47%) but slightly lower returns

3. ⚠️ **ViT underperforms**
   - Cumulative return of 2.84x, below benchmark
   - Selection ratio too high (23.27%), weak stock picking ability

### 5.3 Stock Selection Statistics
| Model | Selected Stocks | Total Stocks | Selection Ratio |
|-------|-----------------|--------------|-----------------|
| Baseline | 25,922 | 1,403,975 | 1.85% |
| Baseline Large | 62,823 | 1,403,975 | 4.47% |
| ViT | 326,761 | 1,403,975 | 23.27% |

---

## 6. Visualization Results

### 6.1 Generated Charts

| Chart | File Path | Description |
|-------|-----------|-------------|
| Comprehensive Comparison | `pic/test_comparison.png` | Includes cumulative return, log return, excess return, test metrics |
| Stock Selection Count | `pic/stocks_selected.png` | Daily selected stock count changes for each model |

### 6.2 Chart Contents

**test_comparison.png** contains 4 subplots:
1. **Cumulative Return Comparison**: Strategy vs benchmark for each model
2. **Log Cumulative Return**: Clearer view of long-term trends
3. **Excess Return Curve**: Strategy performance relative to benchmark
4. **Test Metrics Bar Chart**: Test loss and accuracy comparison

**stocks_selected.png**:
- Scatter plot showing daily selected stock counts
- Observe temporal distribution of model selection behavior

---

## 7. Core Code Logic

### 7.1 Evaluation Loop
```python
def eval_loop(dataloader, net, loss_fn, model_name="model"):
    running_loss = 0.0
    current = 0
    net.eval()
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = net(X)
            loss = loss_fn(y_pred, y.long())
            
            # Weighted average loss calculation
            running_loss = (len(X) * loss.item() + running_loss * current) / (len(X) + current)
            current += len(X)
            
    return running_loss, torch.cat(predict), torch.cat(target)
```

### 7.2 Strategy Return Calculation
```python
def calculate_strategy_returns(label_df, predict_logit, threshold=0.58):
    # Baseline: Buy all stocks
    ret_baseline = label_df.groupby(['Date'])['Ret_20d'].mean()
    
    # Strategy: Only buy stocks with prediction probability > threshold
    mask = predict_logit > threshold
    label_filtered = label_df[mask]
    ret_strategy = label_filtered.groupby(['Date'])['Ret_20d'].mean()
    
    return ret_baseline, ret_strategy
```

---

## 8. Data Field Descriptions

### 8.1 label_df Fields
| Field | Description |
|-------|-------------|
| Date | Trading date |
| Ret_20d | 20-day future return |
| Others | Stock code, price, etc. |

### 8.2 Results Dictionary Fields
| Field | Description |
|-------|-------------|
| test_loss | Test set loss |
| accuracy | Prediction accuracy |
| threshold | Selection threshold |
| num_selected | Number of selected stocks |
| selection_ratio | Selection percentage |
| cum_ret_strategy | Strategy cumulative return |
| excess_return | Excess return |

---

## 9. Conclusions and Recommendations

### 9.1 Model Selection Recommendations
| Scenario | Recommended Model | Reason |
|----------|-------------------|--------|
| Production Deployment | Baseline | Few parameters, fast, highest returns |
| Research Exploration | Baseline Large | Large capacity, room for optimization |
| Not Recommended | ViT | Underperforms on this task |

### 9.2 Strategy Optimization Directions
1. **Threshold Tuning**: Try different thresholds to find optimal selection ratio
2. **Weighted Allocation**: Weight allocation by prediction probability instead of equal-weight
3. **Risk Control**: Add stop-loss mechanisms and position management
4. **Ensemble Strategy**: Combine predictions from multiple models

### 9.3 Limitations
- Transaction costs and slippage not considered
- Assumes unlimited divisibility for stock purchases
- Historical backtesting does not guarantee future performance

---

## 10. Reproduction Guide

### 10.1 Run Testing
```bash
cd notebooks
jupyter notebook test.ipynb
```

### 10.2 Output Files
| File | Description |
|------|-------------|
| `pt/test_results.json` | Test results JSON |
| `pic/test_comparison.png` | Comprehensive comparison chart |
| `pic/stocks_selected.png` | Stock selection count chart |

---

*Report generated: December 2025*
