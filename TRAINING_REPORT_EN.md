# Stock CNN Training Report

## 1. Experiment Overview

This experiment is based on the methodology from the paper **(Re-)Imag(in)ing Price Trends**, using deep learning models to classify stock price images and predict the price movement direction over the next 20 days. We implemented and compared three different architectures: **Baseline CNN**, **Baseline Large CNN**, and **Vision Transformer (ViT)**.

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

## 4. Model Architecture Comparison

### 4.1 Model Parameters

| Model | Parameters | Description |
|-------|------------|-------------|
| **Baseline** | 708,866 (0.71M) | 3-layer CNN + FC |
| **Baseline Large** | 10,233,602 (10.23M) | CNN with expanded channels |
| **ViT** | 10,821,314 (10.82M) | Vision Transformer |

### 4.2 Baseline CNN Architecture

| Layer | Output Channels | Kernel | Stride | Dilation | Padding | Activation |
|-------|-----------------|--------|--------|----------|---------|------------|
| Conv2d + BN + LeakyReLU + MaxPool | 64 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Conv2d + BN + LeakyReLU + MaxPool | 128 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Conv2d + BN + LeakyReLU + MaxPool | 256 | (5,3) | (3,1) | (2,1) | (12,1) | LeakyReLU(0.01) |
| Dropout(0.5) + Linear | 2 | - | - | - | - | Softmax |

### 4.3 Baseline Large CNN Architecture

Same structure as Baseline, but channels expanded from 64→128→256 to 96→192→384, with additional hidden layers.

### 4.4 Vision Transformer Architecture

| Configuration | Setting |
|---------------|---------|
| Patch Size | 4×4 |
| Embedding Dim | 384 |
| Transformer Depth | 6 |
| Attention Heads | 6 |
| MLP Ratio | 4 |
| Dropout | 0.1 |

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
| Early Stopping | Patience = 10 epochs |
| Weight Initialization | Xavier Uniform |
| Random Seed | 42 |

---

## 6. Training Results Summary

### 6.1 Three-Model Comparison

| Model | Parameters | Best Epoch | Best Val Loss | Total Epochs |
|-------|------------|------------|---------------|--------------|
| **Baseline** | 0.71M | 18 | 0.687114 | 29 |
| **Baseline Large** | 10.23M | 26 | **0.686382** | 37 |
| **ViT** | 10.82M | 7 | 0.691686 | 18 |

### 6.2 Key Findings

1. **Baseline Large performs best**: Despite 14x more parameters, Val Loss only slightly decreased (0.687 → 0.686).
2. **ViT underperforms**: Vision Transformer struggles to converge on this task, Val Loss stagnates around 0.692.
3. **Baseline is most cost-effective**: Achieves near-best performance with only 0.71M parameters.

---

## 7. Training Process Details

### 7.1 Baseline Training Log

| Epoch | Train Loss | Val Loss | Note |
|-------|-----------|----------|------|
| 0 | 1.0229 | 0.7226 | |
| 5 | 0.7600 | 0.6908 | |
| 10 | 0.7260 | 0.7013 | |
| 15 | 0.7098 | 0.6893 | |
| 18 | 0.7032 | **0.6871** | ⭐ Best |
| 28 | 0.6897 | 0.6917 | Early Stop |

### 7.2 Baseline Large Training Log

| Epoch | Train Loss | Val Loss | Note |
|-------|-----------|----------|------|
| 0 | 0.9485 | 0.7182 | |
| 10 | 0.7109 | 0.6896 | |
| 20 | 0.6842 | 0.6882 | |
| 26 | 0.6761 | **0.6864** | ⭐ Best |
| 36 | 0.6727 | 0.6906 | Early Stop |

### 7.3 ViT Training Log

| Epoch | Train Loss | Val Loss | Note |
|-------|-----------|----------|------|
| 0 | 0.7023 | 0.6979 | |
| 7 | 0.6935 | **0.6917** | ⭐ Best |
| 17 | 0.6927 | 0.6928 | Early Stop |

---

## 8. Training Efficiency Statistics

| Metric | Baseline | Baseline Large | ViT |
|--------|----------|----------------|-----|
| Training Steps per Epoch | 543 | 543 | 543 |
| Validation Steps per Epoch | 233 | 233 | 233 |
| Training Speed | ~30 it/s | ~25 it/s | ~15 it/s |
| Time per Epoch | ~25s | ~35s | ~55s |
| Total Training Time | ~12 min | ~22 min | ~17 min |

---

## 9. Model Checkpoints

| Model | Best Model Path |
|-------|-----------------|
| Baseline | `pt/baseline/best.pt` |
| Baseline Large | `pt/baseline_large/best.pt` |
| ViT | `pt/vit/best.pt` |

---

## 10. Training Curves

Training comparison plot saved at: `pic/training_comparison.png`

![Training Comparison](pic/training_comparison.png)

---

## 11. Conclusions

1. **CNN architecture is more suitable for this task**: Traditional CNNs outperform Vision Transformer on stock image classification.

2. **Parameters don't correlate with performance**: Baseline Large has 14x more parameters than Baseline, but Val Loss only decreased by 0.1%.

3. **Early stopping is effective**: All models triggered early stopping at appropriate times, avoiding overfitting.

4. **Recommend using Baseline**: Considering training efficiency and model size, Baseline is the best choice.

---

## 12. File Structure

```
Stock_CNN/
├── notebooks/
│   └── train.ipynb              # Training script
├── models/
│   ├── baseline.py              # Baseline CNN
│   ├── baseline_large.py        # Baseline Large CNN
│   └── vit.py                   # Vision Transformer
├── pt/
│   ├── baseline/best.pt         # Baseline best model
│   ├── baseline_large/best.pt   # Baseline Large best model
│   ├── vit/best.pt              # ViT best model
│   └── training_results.json    # Training results JSON
├── runs/                        # TensorBoard logs
└── pic/
    └── training_comparison.png  # Training comparison plot
```

---

*Report Generated: 2025-12-02*
