"""
MNIST 模型定义
ConvNet 架构：2 个卷积层 + 2 个全连接层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    MNIST 分类网络
    
    架构:
        Input: (batch, 1, 28, 28)
        Conv1: 32 filters, 3x3, ReLU, MaxPool -> (batch, 32, 13, 13)
        Conv2: 64 filters, 3x3, ReLU, MaxPool -> (batch, 64, 5, 5)
        Flatten -> (batch, 64*5*5)
        FC1: 128 units, ReLU, Dropout
        FC2: 10 units (output)
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MNISTNet, self).__init__()
        
        # 卷积层 1
        self.conv1 = nn.Conv2d(
            in_channels=1,      # 灰度图像
            out_channels=32,    # 32 个卷积核
            kernel_size=3,
            padding=1
        )
        
        # 卷积层 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """前向传播"""
        # Conv1 + ReLU + MaxPool
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 13, 13)
        
        # Conv2 + ReLU + MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 5, 5)
        
        # Flatten
        x = x.view(-1, 64 * 5 * 5)  # (batch, 1600)
        
        # FC1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # FC2 (output)
        x = self.fc2(x)  # (batch, 10)
        
        return x
    
    def predict(self, x):
        """预测（返回类别）"""
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output, 1)
        return predicted


def create_model(num_classes=10, dropout_rate=0.5):
    """创建模型实例"""
    model = MNISTNet(num_classes=num_classes, dropout_rate=dropout_rate)
    return model


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    model = create_model()
    print(f"✓ 模型创建成功")
    print(f"  参数量：{count_parameters(model):,}")
    
    # 测试前向传播
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"  输入形状：{dummy_input.shape}")
    print(f"  输出形状：{output.shape}")
