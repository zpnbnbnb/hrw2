import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class RecommendedLoss(nn.Module):
    """专门针对您的问题设计的损失函数：解决召回率高、精确率低的问题"""

    def __init__(self, num_classes=3, pos_weight=0.2, neg_weight=3.0, device=None):
        super().__init__()
        self.num_classes = num_classes

        # 手动设置类别权重，大幅降低正类权重，提高负类权重
        self.class_weights = torch.tensor([
            pos_weight,  # 正类权重（极低，减少过度预测）
            neg_weight,  # 负类权重（高，增加负类识别）
            1.0  # 第三类权重
        ])

        if device is not None:
            self.class_weights = self.class_weights.to(device)

        # 使用高gamma值的Focal Loss来关注困难样本
        self.focal_loss = FocalLoss(
            alpha=0.25,
            gamma=4,  # 高gamma值
            class_weights=self.class_weights
        )

    def to(self, device):
        """确保权重也移动到正确的设备"""
        super().to(device)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
            self.focal_loss.class_weights = self.class_weights
        return self

    def forward(self, inputs, targets):
        # 确保权重在正确的设备上
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
            self.focal_loss.class_weights = self.class_weights

        return self.focal_loss(inputs, targets)

    # 在 losses.py 文件末尾添加：
    OptimizedFocalLoss = FocalLoss  # 为了与新模型兼容