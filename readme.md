
# Stock Image CNN - PyTorch Implementation 

This is a PyTorch implementation of [**(Re-)Imag(in)ing Price Trends**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587)

## Announcement

This codebase is largely inherit from [Stock_CNN](https://github.com/lich99/Stock_CNN) by [@lich99](https://github.com/lich99).

Re-Implementation Authors:  Gongye Liu, Zixuan Ye, Yatian Wang, Haoqiang Guo, Zhizhou Zhong

## Quick Start

### 🛠️Installation

#### 1. Create a conda environment and install pytorch
```
conda create -n Math5470 python=3.10
conda activate Math5470
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```
#### 2. Other dependencies
```
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

### 🚀 Training

Run the training notebook to train the CNN model:

```bash
cd notebooks
jupyter notebook train.ipynb
```

Alternatively, you can monitor the training process with TensorBoard:

```bash
# In a separate terminal
tensorboard --logdir=runs
```

**Training Configuration:**
- Model: Baseline CNN (3-layer ConvNet)
- Optimizer: Adam (lr=1e-5)
- Batch Size: 1024 (128 × 8 GPUs)
- Early Stopping: Patience = 5 or 10 epochs

For detailed training results, see [TRAINING_REPORT.md](./TRAINING_REPORT.md) or [TRAINING_REPORT_EN.md](./TRAINING_REPORT_EN.md).

### 📊 Testing & Backtesting

Run the test notebook to evaluate the model and perform backtesting:

```bash
cd notebooks
jupyter notebook test.ipynb
```

The test notebook will:
1. Load the best trained model from `./pt/`
2. Run inference on the test set (2001-2019)
3. Construct stock selection strategies based on model predictions
4. Generate performance comparison charts in `./pic/`

For detailed test results, see [TEST_REPORT.md](./TEST_REPORT.md) or [TEST_REPORT_EN.md](./TEST_REPORT_EN.md).

## 📁 Project Structure

```
Stock_CNN/
├── notebooks/
│   ├── train.ipynb              # Training script
│   └── test.ipynb               # Testing & backtesting script
├── models/
│   ├── baseline.py              # Baseline CNN model
│   ├── baseline_large.py        # Large CNN model
│   └── vit.py                   # Vision Transformer model
├── data/
│   └── monthly_20d/             # Stock image data
├── pt/                          # Saved model checkpoints
├── pic/                         # Generated performance charts
├── runs/                        # TensorBoard logs
├── TRAINING_REPORT.md           # Training report (Chinese)
├── TRAINING_REPORT_EN.md        # Training report (English)
├── TEST_REPORT.md               # Test report (Chinese)
├── TEST_REPORT_EN.md            # Test report (English)
└── requirements.txt             # Python dependencies
```

## 📈 Results

| Metric | Value |
|--------|-------|
| Best Validation Loss | 0.6867 |
| Test Loss | 0.694 |
| Best Epoch | 19 |
| Total Parameters | 708,866 (0.71M) |

## 📚 References

- Paper: [(Re-)Imag(in)ing Price Trends](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587)
- Original Implementation: [Stock_CNN by @lich99](https://github.com/lich99/Stock_CNN)