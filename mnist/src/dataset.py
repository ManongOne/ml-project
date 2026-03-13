"""
MNIST 数据集加载模块
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config


def get_transforms():
    """获取数据预处理和增强"""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return train_transform, test_transform


def load_mnist(batch_size=None):
    """
    加载 MNIST 数据集
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    train_transform, test_transform = get_transforms()
    
    # 下载并加载训练集
    train_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 下载并加载测试集
    test_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ 训练集大小：{len(train_dataset)}")
    print(f"✓ 测试集大小：{len(test_dataset)}")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载
    train_loader, test_loader = load_mnist()
    print(f"✓ 数据加载器测试成功")
    print(f"  训练批次数量：{len(train_loader)}")
    print(f"  测试批次数量：{len(test_loader)}")
