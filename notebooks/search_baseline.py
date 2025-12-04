import os
import sys
sys.path.insert(0, '..')

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import TrialState
from tqdm import tqdm

# ==========================================
# 配置
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用单张卡
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

RESULT_DIR = "optuna_results"
os.makedirs(RESULT_DIR, exist_ok=True)

IMAGE_WIDTH = {20: 60}
IMAGE_HEIGHT = {20: 64}

# ==========================================
# 数据加载（直接从你的 train.py 复制）
# ==========================================
def load_data_1993_2000():
    """加载 1993-2000 年的训练数据"""
    year_list = np.arange(1993, 2001, 1)
    
    images = []
    label_df = []
    for year in year_list:
        images.append(np.memmap(
            os.path.join("../data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"), 
            dtype=np.uint8, mode='r'
        ).reshape((-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20])))
        
        label_df.append(pd.read_feather(
            os.path.join("../data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather")
        ))
    
    images = np.concatenate(images)
    label_df = pd.concat(label_df)
    
    print(f"Loaded data shape: images={images.shape}, labels={label_df.shape}")
    
    # 划分训练集和验证集（非随机，按时间顺序）
    split_idx = int(images.shape[0] * 0.7)
    train_dataset = MyDataset(images[:split_idx], (label_df.Ret_20d > 0).values[:split_idx])
    val_dataset = MyDataset(images[split_idx:], (label_df.Ret_20d > 0).values[split_idx:])
    
    return train_dataset, val_dataset

# ==========================================
# Dataset 定义（从你的 train.py）
# ==========================================
class MyDataset(Dataset):
    def __init__(self, img, label):
        self.img = torch.Tensor(img.copy())
        self.label = torch.Tensor(label)
        self.len = len(img)
  
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

# ==========================================
# 模型初始化（从你的 train.py）
# ==========================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

def create_model(model_type="baseline"):
    """创建模型"""
    from models import baseline, baseline_large, vit
    
    if model_type == "baseline":
        net = baseline.Net()
    elif model_type == "baseline_large":
        net = baseline_large.Net()
    elif model_type == "vit":
        net = vit.Net()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    net.apply(init_weights)
    return net.to(DEVICE)

# ==========================================
# Optuna 目标函数
# ==========================================
def objective(trial):
    # 采样超参数
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # 准备数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # 创建模型
    net = create_model("baseline")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练循环（减少 epoch 数以加快搜索）
    EPOCHS = 10
    
    for epoch in range(EPOCHS):
        # 训练阶段
        net.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = net(X)
            loss = criterion(y_pred, y.long())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(X)
        
        train_loss /= len(train_dataset)
        
        # 验证阶段
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                y_pred = net(X)
                loss = criterion(y_pred, y.long())
                val_loss += loss.item() * len(X)
        
        val_loss /= len(val_dataset)
        
        # 报告中间结果（用于剪枝）
        trial.report(val_loss, epoch)
        
        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_loss

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 加载数据（全局变量）
    print("Loading data...")
    train_dataset, val_dataset = load_data_1993_2000()
    
    # 创建 Study
    print("Starting hyperparameter search...")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # 执行搜索
    study.optimize(objective, n_trials=30)
    
    # 输出结果
    print("\n" + "="*50)
    print("Study statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"  Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Val Loss): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # 保存最佳参数
    best_params_path = os.path.join(RESULT_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(trial.params, f, indent=4)
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # 生成可视化
    print("\nGenerating visualizations...")
    try:
        import plotly
        
        fig1 = optuna.visualization.plot_param_importances(study)
        fig1.write_html(os.path.join(RESULT_DIR, "param_importance.html"))
        
        fig2 = optuna.visualization.plot_optimization_history(study)
        fig2.write_html(os.path.join(RESULT_DIR, "optimization_history.html"))
        
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(RESULT_DIR, "parallel_coordinate.html"))
        
        print(f"Visualizations saved to {RESULT_DIR}/")
    except ImportError:
        print("Plotly not installed, skipping visualization generation")
    
    print("\nDone!")