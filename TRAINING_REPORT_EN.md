# Stock CNN Training Report

## 1. Experiment Overview

This experiment is based on the methodology from the paper **(Re-)Imag(in)ing Price Trends**, using Convolutional Neural Networks (CNN) to classify stock price images and predict the price movement direction over the next 20 days.

---

## 2. Hardware & Environment Configuration

| Configuration | Setting |
|---------------|---------|
| Number of GPUs | 8 |
| GPU Devices | CUDA 0-7 |
| Parallelization Strategy | `nn.DataParallel` |
| PyTorch Version | 2.6.0 |
| CUDA Version | 12.6 |
| Python Version | 3.10 |

---

## 3. Dataset Configuration

| Configuration | Setting |
|---------------|---------|
| Training + Validation Data | 1993-2000 |
| Test Data | 2001-2019 |
| Total Samples (Train + Val) | 793,019 |
| Image Size | 64 × 60 (H × W) |
| Label | Binary Classification (Ret_20d > 0) |
| Train/Val Split Ratio | 70% / 30% |
| Training Samples | 555,113 |
| Validation Samples | 237,906 |
| Split Method | Temporal Split (Non-random) |

---

## 4. Model Architecture

### Baseline CNN

| Layer | Output Channels | Kernel | Stride | Dilation | Padding | Activation |
|-------|-----------------|--------|--------|----------|---------|------------|
| Conv2d + BN + LeakyReLU + MaxPool | 64 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Conv2d + BN + LeakyReLU + MaxPool | 128 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Conv2d + BN + LeakyReLU + MaxPool | 256 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Dropout + Linear | 2 | - | - | - | - | Softmax |

| Model Statistics | Value |
|------------------|-------|
| Total Parameters | 708,866 (0.71M) |
| FLOPs (per sample) | 289.76 GFLOPs |

---

## 5. Training Hyperparameters

| Hyperparameter | Setting |
|----------------|---------|
| Batch Size | 1024 (128 × 8 GPUs) |
| Per-GPU Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate | 1e-5 |
| Loss Function | CrossEntropyLoss |
| Max Epochs | 100 |
| Weight Initialization | Xavier Uniform |
| Random Seed | 42 |

---

## 6. Early Stopping Comparison Experiments

### Experiment Settings

| Experiment Name | Early Stopping Patience |
|-----------------|-------------------------|
| baseline-earlystop-5 | 5 epochs |
| baseline-earlystop-10 | 10 epochs |

### Training Results Comparison

| Experiment | Best Epoch | Best Train Loss | Best Val Loss | Total Training Epochs |
|------------|------------|-----------------|---------------|----------------------|
| earlystop-5 | 19 | 0.7014 | **0.6867** | 25 |
| earlystop-10 | 24 | 0.6939 | 0.6873 | 30 |

---

## 7. Training Process Details (earlystop-10)

### Per-Epoch Training Log

| Epoch | Train Loss | Val Loss | Train Time | Val Time | Note |
|-------|------------|----------|------------|----------|------|
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

## 8. Training Efficiency Statistics

| Metric | Value |
|--------|-------|
| Training Steps per Epoch | 543 steps |
| Validation Steps per Epoch | 233 steps |
| Training Speed | ~29-30 it/s |
| Validation Speed | ~35-36 it/s |
| Single Epoch Training Time | ~17-24 seconds |
| Single Epoch Validation Time | ~6-7 seconds |
| Total Training Time (30 epochs) | ~12 minutes |

---

## 9. Best Model

| Item | Value |
|------|-------|
| Best Model Path | `pt/baseline-earlystop-5/best_baseline_epoch_19_train_0.701418_val_0.686740.pt` |
| Best Epoch | 19 |
| Train Loss | 0.7014 |
| Val Loss | 0.6867 |

---

## 10. Key Observations

1. **Overfitting Trend**: Train Loss continues to decrease, but Val Loss plateaus after Epoch 19 with slight fluctuations, indicating the model begins to overfit.

2. **Early Stopping Effect**: Early Stopping with Patience=5 and 10 achieve similar final results, but Patience=5 terminates training earlier, saving computational resources.

3. **Convergence Speed**: The model converges rapidly in the first 10 epochs, then enters a slow optimization phase.

4. **GPU Utilization**: Using 8-GPU DataParallel with effective Batch Size of 1024 significantly accelerates the training process.

---

## 11. File Structure

```
Stock_CNN/
├── notebooks/
│   ├── train.ipynb          # Training script
│   └── test.ipynb           # Testing script
├── models/
│   ├── baseline.py          # Baseline CNN model
│   ├── baseline_large.py    # Large CNN model
│   └── vit.py               # Vision Transformer model
├── pt/
│   ├── baseline-earlystop-5/   # Patience=5 experiment results
│   └── baseline-earlystop-10/  # Patience=10 experiment results
├── data/
│   └── monthly_20d/         # Data directory
└── runs/                    # TensorBoard logs
```

---

## 12. Reproduction Commands

```bash
# 1. Create environment
conda create -n Math5470 python=3.10
conda activate Math5470
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. Launch TensorBoard monitoring
tensorboard --logdir=runs

# 3. Run training notebook
jupyter notebook notebooks/train.ipynb
```

---

*Report Generated: 2025-12-02*

