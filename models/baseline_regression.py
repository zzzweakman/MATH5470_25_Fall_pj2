"""
Baseline CNN for Regression Task
预测股票未来收益率（回归任务）
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    基于 CNN 的回归模型
    输入: 20日K线图 (64 x 60)
    输出: 预测的收益率 (标量)
    """
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        # 特征提取层
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        
        # 回归头：输出单个值
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(46080, 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(64, 1),  # 输出单个预测值
        )
       
    def forward(self, x):
        x = x.reshape(-1, 1, 64, 60)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 46080)
        x = self.regressor(x)
        return x.squeeze(-1)  # 输出形状: (batch_size,)


class NetLarge(nn.Module):
    """
    更大的回归模型，增加通道数和全连接层容量
    """
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        # 特征提取层（更大的通道数）
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(69120, 1024),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(1024, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 1),
        )
       
    def forward(self, x):
        x = x.reshape(-1, 1, 64, 60)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 69120)
        x = self.regressor(x)
        return x.squeeze(-1)

