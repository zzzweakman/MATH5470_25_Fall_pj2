# Stock Image CNN - PyTorch Implementation 

This is a PyTorch implementation of [**(Re-)Imag(in)ing Price Trends**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587)

## Announcement

This codebase is largely inherit from [Stock_CNN](https://github.com/lich99/Stock_CNN) by [@lich99](https://github.com/lich99).

Re-Implementation Authors:  Gongye Liu, Zixuan Ye, Yatian Wang, Haoqiang Guo, Zhizhou Zhong

---

## 📈 Experiment Results

### Training Results (Three Models Comparison)

| Model | Parameters | Best Epoch | Best Val Loss | Total Epochs |
|-------|------------|------------|---------------|--------------|
| **Baseline** | 0.71M | 18 | 0.6871 | 29 |
| **Baseline Large** | 10.23M | 26 | **0.6864** | 37 |
| **ViT** | 10.82M | 7 | 0.6917 | 18 |

### Test Results (Backtesting 2001-2019)

| Model | Threshold | Test Loss | Accuracy | Selection % | Cumulative Return | Excess Return |
|-------|-----------|-----------|----------|-------------|-------------------|---------------|
| **Baseline** | 0.58 | 0.6942 | 51.58% | 1.85% | **19.86x** | **+16.68** |
| **Baseline Large** | 0.58 | 0.6926 | 52.10% | 4.47% | 17.11x | +13.93 |
| **ViT** | 0.50 | 0.6935 | 49.84% | 23.27% | 2.84x | -0.34 |
| Buy All (Benchmark) | - | - | - | 100% | 3.18x | 0 |

### Key Findings

1. 🏆 **Baseline CNN achieves the best performance** with 19.86x cumulative return over 19 years
2. 📊 **Model complexity doesn't guarantee better results** - the simplest model outperforms larger ones
3. ⚠️ **ViT underperforms** on this task - traditional CNNs are more suitable for stock image classification
4. 💡 **Selective stock picking works** - selecting only 1.85% of stocks yields 6x better returns than buying all

### Performance Visualization

Training Loss Comparison | Test Results Comparison
:-------------------------:|:-------------------------:
![Training](pic/training_comparison.png) | ![Test](pic/test_comparison.png)

---

## Quick Start

### 🛠️ Installation

#### 1. Create a conda environment and install pytorch
```bash
conda create -n Math5470 python=3.10
conda activate Math5470
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

#### 2. Other dependencies
```bash
pip install -r requirements.txt
```

#### 3. Download data
Download the dataset from this [link](https://dachxiu.chicagobooth.edu/download/img_data.zip), then extract it to the `./data` directory:

```bash
cd Stock_CNN
mkdir -p data
wget https://dachxiu.chicagobooth.edu/download/img_data.zip -O data/img_data.zip
unzip data/img_data.zip -d data/
rm data/img_data.zip
```

---

### 🚀 Training

Run the training notebook to train all three models (Baseline, Baseline Large, ViT):

```bash
cd notebooks
jupyter notebook train.ipynb
```

Monitor the training process with TensorBoard:

```bash
tensorboard --logdir=runs
```

**Training Configuration:**
- Models: Baseline CNN, Baseline Large CNN, Vision Transformer
- Optimizer: Adam (lr=1e-5)
- Batch Size: 512 (256 × 2 GPUs)
- Early Stopping: Patience = 5 epochs

📄 For detailed training results, see:
- [TRAINING_REPORT.md](./TRAINING_REPORT.md) (中文)
- [TRAINING_REPORT_EN.md](./TRAINING_REPORT_EN.md) (English)

---

### 📊 Testing & Backtesting

Run the test notebook to evaluate all models and perform backtesting:

```bash
cd notebooks
jupyter notebook test.ipynb
```

The test notebook will:
1. Load the best trained models from `./pt/`
2. Run inference on the test set (2001-2019, 1.4M samples)
3. Construct stock selection strategies based on model predictions
4. Compare cumulative returns across all models
5. Generate performance comparison charts in `./pic/`

📄 For detailed test results, see:
- [TEST_REPORT.md](./TEST_REPORT.md) (中文)
- [TEST_REPORT_EN.md](./TEST_REPORT_EN.md) (English)

---

### 📈 Regression Task (New!)

In addition to the classification task, we also implemented a **regression task** that directly predicts stock returns.

#### Training Regression Model

```bash
cd notebooks
jupyter notebook train_regression.ipynb
```

**Regression Configuration:**
- Target: `Ret_5d` (5-day future return)
- Loss Function: MSELoss
- Output: Single scalar (predicted return)

#### Testing Regression Model

```bash
cd notebooks
jupyter notebook test_regression.ipynb
```

**Backtesting Strategy:**
- Buy stocks with predicted return > 1%
- Hold for 5 days (implicit sell)
- Compare with baseline (buy all stocks)

#### Classification vs Regression

| Comparison | Classification | Regression |
|------------|----------------|------------|
| Target | Direction (0/1) | Return value |
| Loss | CrossEntropyLoss | MSELoss |
| Output | 2 neurons + Softmax | 1 neuron |
| Selection | Probability > 0.58 | Predicted return > 1% |
| Holding Period | 20 days | 5 days |

📄 For detailed regression results, see:
- [Regression_REPORT.md](./Regression_REPORT.md) (中文)
- [Regression_REPORT_EN.md](./Regression_REPORT_EN.md) (English)

---

## 📁 Project Structure

```
Stock_CNN/
├── notebooks/
│   ├── train.ipynb              # Classification training (all 3 models)
│   ├── test.ipynb               # Classification testing & backtesting
│   ├── train_regression.ipynb   # Regression training
│   └── test_regression.ipynb    # Regression testing & backtesting
├── models/
│   ├── baseline.py              # Baseline CNN for classification (0.71M)
│   ├── baseline_large.py        # Large CNN for classification (10.23M)
│   ├── baseline_regression.py   # Baseline CNN for regression (0.73M)
│   └── vit.py                   # Vision Transformer (10.82M)
├── data/
│   └── monthly_20d/             # Stock image data
├── pt/
│   ├── baseline/best.pt         # Best classification model
│   ├── baseline_large/best.pt   # Best large classification model
│   ├── vit/best.pt              # Best ViT model
│   ├── regression_baseline_*/   # Regression model checkpoints
│   ├── training_results.json    # Classification training results
│   └── test_results.json        # Classification test results
├── pic/
│   ├── training_comparison.png  # Classification training curves
│   ├── test_comparison.png      # Classification test results
│   ├── stocks_selected.png      # Stock selection over time
│   ├── test_regression_comparison.png   # Regression backtest
│   └── test_regression_thresholds.png   # Threshold analysis
├── runs/                        # TensorBoard logs
├── TRAINING_REPORT.md           # Classification training report (Chinese)
├── TRAINING_REPORT_EN.md        # Classification training report (English)
├── TEST_REPORT.md               # Classification test report (Chinese)
├── TEST_REPORT_EN.md            # Classification test report (English)
├── Regression_REPORT.md         # Regression report (Chinese)
├── Regression_REPORT_EN.md      # Regression report (English)
└── requirements.txt             # Python dependencies
```

---

## 📊 Model Architectures

### Classification Models

#### Baseline CNN (Recommended)
- 3-layer ConvNet with BatchNorm and LeakyReLU
- Parameters: 708,866 (0.71M)
- Output: 2 classes (up/down)
- Best for: Production use, fast inference

#### Baseline Large CNN
- Same architecture with expanded channels (96→192→384)
- Parameters: 10,233,602 (10.23M)
- Output: 2 classes (up/down)
- Best for: When more model capacity is needed

#### Vision Transformer (ViT)
- Patch-based transformer with 6 layers
- Parameters: 10,821,314 (10.82M)
- Output: 2 classes (up/down)
- Note: Underperforms on this task

### Regression Models

#### Baseline Regression CNN
- Same CNN backbone as classification
- Additional MLP head: 46080 → 512 → 64 → 1
- Parameters: ~730,000 (0.73M)
- Output: Single scalar (predicted return)
- Loss: MSELoss

---

## 📚 References

- Paper: [(Re-)Imag(in)ing Price Trends](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587)
- Original Implementation: [Stock_CNN by @lich99](https://github.com/lich99/Stock_CNN)

---

## 📝 License

This project is for educational purposes only. Please refer to the original paper and implementation for licensing details.
