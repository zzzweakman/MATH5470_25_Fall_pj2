# Stock CNN Regression Task Report

## 1. Task Overview

This report documents the CNN-based stock return regression prediction task. Unlike the classification task (predicting up/down direction), the regression task directly predicts the numerical value of future stock returns.

### 1.1 Task Comparison

| Comparison | Classification Task | Regression Task |
|------------|---------------------|-----------------|
| Target | Predict direction (0/1) | Predict return value |
| Loss Function | CrossEntropyLoss | MSELoss |
| Output Layer | 2 neurons + Softmax | 1 neuron |
| Evaluation Metrics | Accuracy | MSE, MAE, Correlation |

---

## 2. Training Configuration

### 2.1 Dataset

| Item | Description |
|------|-------------|
| Training Data | 1993-2000 |
| Training Samples | 793,019 |
| Test Data | 2001-2019 |
| Test Samples | 1,403,975 |
| Image Size | 64 × 60 |
| Target Variable | `Ret_5d` (5-day future return) |

### 2.2 Data Split

| Split | Ratio | Samples |
|-------|-------|---------|
| Training Set | 70% | ~555,000 |
| Validation Set | 30% | ~238,000 |
| Split Method | Temporal split (non-random) | - |

### 2.3 Model Architecture

**Baseline Regression CNN**

```
Input: 20-day candlestick chart (64 × 60)
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
Output: Predicted return (scalar)
```

**Parameters**: ~0.73M

### 2.4 Training Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Batch Size | 256 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | MSELoss |
| Max Epochs | 100 |
| Early Stopping Patience | 10 epochs |

---

## 3. Training Results

### 3.1 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error, lower is better |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| Correlation | Correlation between predictions and true values |
| Direction Accuracy | Accuracy of predicting up/down direction |

### 3.2 Training Curves

The following metrics are recorded during training:
- `Loss/train_mse`: Training set MSE
- `Loss/val_mse`: Validation set MSE
- `Metrics/val_mae`: Validation set MAE
- `Metrics/val_correlation`: Prediction-target correlation

### 3.3 Model Saving

```
pt/regression_baseline_YYYYMMDD_HHMMSS/
├── best.pt                           # Best model
├── epoch_X_mse_X_corr_X.pt          # Epoch checkpoints
└── training_curves.png               # Training curves plot
```

---

## 4. Backtesting Strategy

### 4.1 Strategy Design

Using simplified backtesting logic consistent with the classification task:

**Buy Condition**: Predicted return > threshold (default 1%)

**Strategy Flow**:
1. Each trading day, model predicts 5-day returns for all stocks
2. Select stocks with predicted return > 1%
3. Equal-weight purchase of selected stocks
4. Implicit sell after holding for 5 days
5. Calculate average actual return of selected stocks daily

### 4.2 Return Calculation

```python
# Baseline return: Average 5-day return of all stocks daily
ret_baseline = label_df.groupby('Date')['Ret_5d'].mean()

# Strategy return: Average 5-day return of selected stocks daily
mask = predictions > threshold  # threshold = 0.01 (1%)
ret_strategy = label_df[mask].groupby('Date')['Ret_5d'].mean()

# Cumulative return
cum_return = (ret + 1).cumprod()
```

### 4.3 Threshold Sensitivity

Testing the effect of different thresholds on strategy performance:

| Threshold | Selection Ratio | Cumulative Return | Excess Return |
|-----------|-----------------|-------------------|---------------|
| 0.5% | High | - | - |
| 1.0% | Medium | - | - |
| 2.0% | Low | - | - |
| 3.0% | Very Low | - | - |
| 5.0% | Extremely Low | - | - |

---

## 5. Output Files

### 5.1 Training Output

| File | Description |
|------|-------------|
| `pt/regression_baseline_*/best.pt` | Best regression model |
| `pt/regression_baseline_*/training_curves.png` | Training curves |
| `runs/` | TensorBoard logs |

### 5.2 Test Output

| File | Description |
|------|-------------|
| `pt/test_regression_results.json` | Test results JSON |
| `pic/test_regression_comparison.png` | Backtest comparison plot |
| `pic/test_regression_thresholds.png` | Threshold sensitivity analysis plot |

---

## 6. Visualization Description

### 6.1 test_regression_comparison.png

Contains 4 subplots:

1. **Cumulative Return Comparison**: Strategy vs benchmark cumulative return curves
2. **Prediction vs True Scatter Plot**: Evaluate prediction accuracy
3. **Prediction Distribution**: Histogram of predicted returns
4. **Daily Stock Selection**: Number of stocks selected over time

### 6.2 test_regression_thresholds.png

Contains 2 subplots:

1. **Cumulative Return vs Threshold**: Strategy returns at different thresholds
2. **Selection Ratio vs Threshold**: Selection ratio at different thresholds

---

## 7. Comparison with Classification Task

| Comparison | Classification (test.ipynb) | Regression (test_regression.ipynb) |
|------------|----------------------------|-----------------------------------|
| Model Output | Probability (0~1) | Predicted return value |
| Selection Condition | Probability > 0.58 | Predicted return > 1% |
| Holding Period | 20 days | 5 days |
| Return Field | `Ret_20d` | `Ret_5d` |
| Backtest Logic | Simplified (implicit sell) | Simplified (implicit sell) |

---

## 8. Reproduction Guide

### 8.1 Train Regression Model

```bash
cd notebooks
jupyter notebook train_regression.ipynb
```

Configuration:
- `model_type`: "baseline" or "large"
- `target_col`: "Ret_5d" or "Ret_20d"

### 8.2 Test and Backtest

```bash
cd notebooks
jupyter notebook test_regression.ipynb
```

Configuration:
- `model_path`: Path to trained model
- `threshold`: Selection threshold (default 0.01 i.e. 1%)
- `target_col`: Must match training configuration

### 8.3 Monitor Training

```bash
tensorboard --logdir=runs
```

---

## 9. Important Notes

### 9.1 NaN Handling

The `Ret_5d` field may contain NaN values (due to stock suspension, etc.), which are automatically filtered:

```python
if nan_count > 0:
    valid_mask = ~np.isnan(target_values)
    target_values = target_values[valid_mask]
    images = images[valid_mask]
```

### 9.2 Backtesting Limitations

- Transaction costs and slippage not considered
- Assumes unlimited divisibility for stock purchases
- Overlapping holding period issue is simplified
- Historical backtesting does not guarantee future performance

---

*Report generated: December 2025*

