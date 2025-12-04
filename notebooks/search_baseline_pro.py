#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单卡（H800 优化版）Optuna 搜索脚本
- 继承 Claude 单卡方案
- 扩展精度模式 / 更大 batch / 多模型 / 余弦调度 / 梯度累积等
"""

import os
import sys
import json
import argparse
import datetime
from dataclasses import dataclass, asdict
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import optuna
from optuna.trial import TrialState

# -----------------------------------------------------------------------------
# 0. 基础配置
# -----------------------------------------------------------------------------
sys.path.insert(0, '..')  # 以便导入 models 包

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # 让 TF32 自动生效（适合 H800）

RESULT_DIR = "optuna_results_pro"
os.makedirs(RESULT_DIR, exist_ok=True)

IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

# -----------------------------------------------------------------------------
# 1. 数据集 & 模型
# -----------------------------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, img, label):
        # 维持与 train.py 一致的张量形状
        self.img = torch.tensor(img.copy(), dtype=torch.float32)
        self.label = torch.tensor(label.astype(np.int64))
        self.len = len(img)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


def load_images_and_labels(
    data_root: str,
    years: Sequence[int],
    window: int = 20,
) -> Tuple[np.ndarray, pd.DataFrame]:
    images, label_df = [], []
    for year in years:
        img_path = os.path.join(
            data_root, f"{window}d_month_has_vb_[{window}]_ma_{year}_images.dat"
        )
        lbl_path = os.path.join(
            data_root, f"{window}d_month_has_vb_[{window}]_ma_{year}_labels_w_delay.feather"
        )
        images.append(
            np.memmap(img_path, dtype=np.uint8, mode="r").reshape(
                (-1, IMAGE_HEIGHT[window], IMAGE_WIDTH[window])
            )
        )
        label_df.append(pd.read_feather(lbl_path))
    images = np.concatenate(images)
    label_df = pd.concat(label_df)
    return images, label_df


def build_datasets(
    images: np.ndarray,
    labels: pd.DataFrame,
    split_ratio: float = 0.7,
    random_split_flag: bool = False,
    seed: int = 42,
):
    targets = (labels.Ret_20d > 0).values.astype(np.int64)
    dataset = MyDataset(images, targets)

    if random_split_flag:
        train_len = int(len(dataset) * split_ratio)
        val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        split_idx = int(len(dataset) * split_ratio)
        train_ds = MyDataset(images[:split_idx], targets[:split_idx])
        val_ds = MyDataset(images[split_idx:], targets[split_idx:])

    return train_ds, val_ds


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def create_model(model_type: str):
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
    return net


# -----------------------------------------------------------------------------
# 2. 训练与验证逻辑
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_type: str = "baseline"
    epochs: int = 20
    patience: int = 5
    grad_accum: int = 1
    precision: str = "tf32"  # choices: tf32 / bf16 / fp16 / fp32
    optimizer: str = "adamw"
    lr: float = 3e-5
    weight_decay: float = 1e-4
    batch_size: int = 512
    scheduler: str = "cosine"  # cosine / none
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.0
    ema_decay: float = 0.0
    torch_compile: bool = False


class ModelEMA:
    """简单 EMA 用于在 eval 时提升稳定性"""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self, model):
        for shadow_p, model_p in zip(self.shadow, model.parameters()):
            if model_p.requires_grad:
                shadow_p.mul_(self.decay).add_(model_p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = []
        for shadow_p, model_p in zip(self.shadow, model.parameters()):
            if model_p.requires_grad:
                self.backup.append(model_p.detach().clone())
                model_p.data.copy_(shadow_p)

    @torch.no_grad()
    def restore(self, model):
        for backup_p, model_p in zip(self.backup, model.parameters()):
            if model_p.requires_grad:
                model_p.data.copy_(backup_p)


def get_precision_context(cfg: TrainConfig):
    if cfg.precision == "bf16":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    if cfg.precision == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    if cfg.precision == "tf32":
        # 已在全局开启 TF32，无需 autocast
        return torch.cuda.amp.autocast(enabled=False)
    return torch.cuda.amp.autocast(enabled=False)


def build_optimizer(params, cfg: TrainConfig):
    if cfg.optimizer == "adamw":
        return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: TrainConfig, total_steps: int):
    if cfg.scheduler == "cosine":
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        def lr_lambda(step):
            if step < warmup_steps and warmup_steps > 0:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return None


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    cfg: TrainConfig,
    device,
    ema: ModelEMA = None,
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)

    context = get_precision_context(cfg)

    for step, (inputs, targets) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with context:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss = loss / cfg.grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * inputs.size(0) * cfg.grad_accum
        total_samples += inputs.size(0)

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(model, val_loader, criterion, cfg: TrainConfig, device, ema: ModelEMA = None):
    if ema is not None:
        ema.apply_shadow(model)

    model.eval()
    total_loss = 0.0
    total_samples = 0

    context = get_precision_context(cfg)

    for inputs, targets in tqdm(val_loader, desc="Val", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with context:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    if ema is not None:
        ema.restore(model)

    return total_loss / max(1, total_samples)


def run_training_loop(model, train_loader, val_loader, cfg: TrainConfig, device):
    model = model.to(device)
    if cfg.torch_compile:
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model.parameters(), cfg)
    total_steps = cfg.epochs * len(train_loader) // max(1, cfg.grad_accum)
    scheduler = build_scheduler(optimizer, cfg, total_steps) if total_steps > 0 else None
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.precision in {"fp16"})
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    global_step = 0

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg, device, ema)
        val_loss = evaluate(model, val_loader, criterion, cfg, device, ema)

        if scheduler is not None:
            scheduler.step()

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": asdict(cfg),
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

        global_step += len(train_loader)

    return best_val, best_state


# -----------------------------------------------------------------------------
# 3. Optuna 目标函数
# -----------------------------------------------------------------------------
def objective_factory(train_dataset, val_dataset, device, args):
    def objective(trial):
        cfg = TrainConfig(
            model_type=trial.suggest_categorical("model_type", ["baseline"]),
            epochs=args.max_epochs,
            patience=args.patience,
            grad_accum=trial.suggest_categorical("grad_accum", [1, 2, 4]),
            precision=trial.suggest_categorical("precision", ["tf32", "bf16"]),
            optimizer=trial.suggest_categorical("optimizer", ["adamw", "adam"]),
            lr=trial.suggest_float("lr", 5e-6, 5e-4, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 384, 512, 768, 1024]),
            scheduler=trial.suggest_categorical("scheduler", ["cosine", "none"]),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
            label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.1),
            ema_decay=trial.suggest_float("ema_decay", 0.0, 0.999),
            torch_compile=trial.suggest_categorical("torch_compile", [False, True]),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(1024, cfg.batch_size),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = create_model(cfg.model_type)

        best_val, _ = run_training_loop(model, train_loader, val_loader, cfg, device)

        trial.report(best_val, cfg.epochs)
        return best_val

    return objective


# -----------------------------------------------------------------------------
# 4. 主入口
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="H800-friendly Optuna search")
    parser.add_argument("--data_root", type=str, default="../data/monthly_20d")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(1993, 2001)))
    parser.add_argument("--window", type=int, default=20, choices=[5, 20, 60])
    parser.add_argument("--split_ratio", type=float, default=0.7)
    parser.add_argument("--random_split", action="store_true")
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--study_name", type=str, default="cnn_finance_search_pro")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 单卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}, years={args.years}")

    print("[INFO] Loading data ...")
    images, labels = load_images_and_labels(args.data_root, args.years, window=args.window)
    train_ds, val_ds = build_datasets(
        images,
        labels,
        split_ratio=args.split_ratio,
        random_split_flag=args.random_split,
        seed=args.seed,
    )
    print(f"[INFO] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    objective = objective_factory(train_ds, val_ds, device, args)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=f"sqlite:///{os.path.join(RESULT_DIR, 'search_pro.db')}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print("\n" + "=" * 60)
    print(f"Finished Trials: {len(study.trials)}")
    print(f"Best value: {study.best_trial.value:.6f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_params_path = os.path.join(RESULT_DIR, f"best_params_{timestamp}.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print(f"\n[INFO] Saved best params to {best_params_path}")

    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(RESULT_DIR, f"opt_hist_{timestamp}.html"))
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(RESULT_DIR, f"param_imp_{timestamp}.html"))
        print("[INFO] Visualization exported.")
    except Exception as e:
        print(f"[WARN] Visualization skipped ({e}).")


if __name__ == "__main__":
    main()