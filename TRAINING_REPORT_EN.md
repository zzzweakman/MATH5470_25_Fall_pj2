# Stock CNN Training Report

## 1. Experiment Overview

This experiment is based on the methodology from the paper **(Re-)Imag(in)ing Price Trends**, using convolutional neural networks to classify stock price images and predict the direction of returns over the next 20 days. We trained and compared three different model architectures: Baseline CNN, Baseline Large CNN, and Vision Transformer (ViT).

---

## 2. Experimental Environment

### 2.1 Hardware Configuration
| Item | Specification |
|------|---------------|
| GPU | NVIDIA GPU × 2 (using DataParallel) |
| GPU IDs | cuda:6, cuda:7 |
| Parallelization | nn.DataParallel |

### 2.2 Software Environment
| Software | Version |
|----------|---------|
| Python | 3.10 |
| PyTorch | 2.6.0 |
| CUDA | 12.6 |

---

## 3. Dataset

### 3.1 Training Data
| Item | Description |
|------|-------------|
| Time Range | 1993 - 2000 |
| Sample Count | 793,019 |
| Image Size | 64 × 60 (H × W) |
| Image Type | 20-day candlestick chart (with volume bars) |
| Label | Binary classification (20-day future return > 0) |

### 3.2 Data Split
| Split Method | Ratio | Sample Count |
|--------------|-------|--------------|
| Training Set | 70% | 555,113 |
| Validation Set | 30% | 237,906 |
| Split Method | Temporal split (non-random) | - |

---

## 4. Model Architectures

### 4.1 Baseline CNN
```
Structure: Conv2d → BatchNorm → LeakyReLU → MaxPool (×3) → FC → Softmax
Channels: 1 → 64 → 128 → 256 → 2
Kernel Size: 5×3
Parameters: 708,866 (0.71M)
FLOPs: 72.44G
```

### 4.2 Baseline Large CNN
```
Structure: Conv2d → BatchNorm → LeakyReLU → MaxPool (×3) → FC → Softmax
Channels: 1 → 96 → 192 → 384 → 2
Kernel Size: 5×3
Parameters: 10,233,602 (10.23M)
```

### 4.3 Vision Transformer (ViT)
```
Structure: PatchEmbed → Transformer Encoder (×6) → MLP Head
Patch Size: 8×8
Embedding Dimension: 256
Attention Heads: 8
Parameters: 10,821,314 (10.82M)
```

---

## 5. Training Configuration

### 5.1 Hyperparameters
| Hyperparameter | Value |
|----------------|-------|
| Batch Size (per GPU) | 256 |
| Effective Batch Size | 512 (256 × 2 GPUs) |
| Optimizer | Adam |
| Learning Rate | 1e-5 |
| Loss Function | CrossEntropyLoss |
| Max Epochs | 100 |
| Early Stopping Patience | 5 epochs |
| Weight Initialization | Xavier Uniform |

### 5.2 Training Steps
| Item | Value |
|------|-------|
| Training Steps per Epoch | 2,169 steps |
| Validation Steps per Epoch | 930 steps |
| Training Speed | ~50-60 it/s |

---

## 6. Training Results Comparison

### 6.1 Best Model Performance
| Model | Parameters | Best Epoch | Best Val Loss | Total Epochs |
|-------|------------|------------|---------------|--------------|
| **Baseline** | 0.71M | 18 | 0.6871 | 29 |
| **Baseline Large** | 10.23M | 26 | **0.6864** | 37 |
| **ViT** | 10.82M | 7 | 0.6917 | 18 |

### 6.2 Training Progress (Baseline Model Example)

| Epoch | Train Loss | Val Loss | Time |
|-------|------------|----------|------|
| 0 | 0.9300 | ~0.72 | ~36s |
| 1 | 0.7900 | ~0.71 | ~45s |
| 2 | 0.7540 | ~0.70 | ~34s |
| 3 | 0.7350 | ~0.70 | ~35s |
| 4 | 0.7220 | ~0.69 | ~38s |
| 5 | 0.7140 | ~0.69 | ~35s |
| ... | ... | ... | ... |
| 18 | ~0.689 | 0.6871 | ~35s |

### 6.3 Convergence Characteristics
| Model | Convergence Speed | Final Performance | Training Stability |
|-------|-------------------|-------------------|-------------------|
| Baseline | Medium | Good | ⭐⭐⭐⭐ |
| Baseline Large | Slow | Best | ⭐⭐⭐⭐ |
| ViT | Fast | Poor | ⭐⭐⭐ |

---

## 7. Training Efficiency Analysis

### 7.1 Time Cost
| Model | Time per Epoch | Total Training Time | Inference Speed |
|-------|----------------|---------------------|-----------------|
| Baseline | ~50s | ~25min | Fast |
| Baseline Large | ~80s | ~50min | Medium |
| ViT | ~120s | ~36min | Slow |

### 7.2 GPU Utilization
- Training parallelized across 2 GPUs using DataParallel
- Effective batch size = 256 × 2 = 512
- GPU memory utilization: ~70-80%

---

## 8. Model Saving

### 8.1 Save Paths
```
pt/
├── baseline/
│   ├── best.pt                    # Best model
│   └── baseline_epoch_X_train_X_val_X.pt  # Epoch checkpoints
├── baseline_large/
│   ├── best.pt
│   └── baseline_large_epoch_X_train_X_val_X.pt
└── vit/
    ├── best.pt
    └── vit_epoch_X_train_X_val_X.pt
```

### 8.2 TensorBoard Logs
```
runs/
└── YYYYMMDD_HH:MM:SS/
    ├── Loss/train_step    # Logged every 100 steps
    ├── Loss/train_epoch   # Logged every epoch
    └── Loss/val_epoch     # Logged every epoch
```

---

## 9. Key Findings

### 9.1 Model Complexity vs Performance
- 🏆 **Baseline Large achieves the lowest validation loss** (0.6864)
- 📊 However, Baseline model offers the best cost-performance ratio (only 0.71M parameters)
- ⚠️ ViT underperforms on this task, possibly unsuitable for small image classification

### 9.2 Training Recommendations
1. **Recommend using Baseline model**: Fewer parameters, faster training, stable performance
2. **Early stopping is effective**: Prevents overfitting, saves training time
3. **Learning rate 1e-5 is appropriate**: Stable convergence without oscillation

---

## 10. Reproduction Guide

### 10.1 Environment Setup
```bash
conda create -n Math5470 python=3.10
conda activate Math5470
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 10.2 Start Training
```bash
cd notebooks
jupyter notebook train.ipynb
```

### 10.3 Monitor Training
```bash
tensorboard --logdir=runs
```

---

## 11. Output Files

| File | Description |
|------|-------------|
| `pt/*/best.pt` | Best checkpoints for each model |
| `pic/training_comparison.png` | Training curves comparison |
| `runs/` | TensorBoard logs |
| `*.onnx` | ONNX format models (optional export) |

---

*Report generated: December 2025*
