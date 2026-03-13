"""
MNIST 训练脚本
完整的训练流程，包含训练、验证、测试和模型保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime

import config
from src.dataset import load_mnist
from src.model import create_model, count_parameters


def set_seed(seed):
    """设置随机种子确保可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    return test_loss, test_acc


def plot_curves(train_losses, train_accs, test_losses, test_accs, save_path=None):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(test_losses, label='Test Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc', marker='o')
    ax2.plot(test_accs, label='Test Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 训练曲线已保存：{save_path}")
    else:
        plt.show()


def main():
    """主训练函数"""
    print("=" * 60)
    print("MNIST 手写数字分类训练")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(config.SEED)
    print(f"\n✓ 随机种子：{config.SEED}")
    
    # 检测设备
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"✓ 使用 CPU")
        config.DEVICE = "cpu"
    
    # 加载数据
    print(f"\n正在加载 MNIST 数据集...")
    train_loader, test_loader = load_mnist()
    
    # 创建模型
    print(f"\n正在创建模型...")
    model = create_model()
    model.to(device)
    
    num_params = count_parameters(model)
    print(f"✓ 模型参数量：{num_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # 训练记录
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_acc = 0.0
    
    # 创建模型保存目录
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # 开始训练
    print(f"\n开始训练 (EPOCHS={config.EPOCHS})...")
    print("-" * 60)
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 学习率调整
        scheduler.step()
        
        # 评估
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 打印结果
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(config.MODEL_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, best_model_path)
            print(f"  ⭐ 保存最佳模型：{best_model_path} (Test Acc: {test_acc:.2f}%)")
        
        # 定期保存检查点
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                config.MODEL_DIR, 
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, checkpoint_path)
            print(f"  ✓ 保存检查点：{checkpoint_path}")
    
    # 训练完成
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"  最佳测试准确率：{best_acc:.2f}%")
    print(f"  最终测试准确率：{test_accs[-1]:.2f}%")
    print(f"  模型保存位置：{config.MODEL_DIR}/")
    
    # 绘制训练曲线
    plot_path = os.path.join(config.MODEL_DIR, "training_curves.png")
    plot_curves(train_losses, train_accs, test_losses, test_accs, save_path=plot_path)
    
    # 打印 Colab 使用说明
    print("\n" + "=" * 60)
    print("📓 Colab 使用说明:")
    print("=" * 60)
    print("""
在 Colab 中运行以下命令:

# 1. 克隆仓库
!git clone https://github.com/ManongOne/ml-project.git
%cd ml-project/mnist

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 运行训练
!python src/train.py

# 4. (可选) 下载训练好的模型
from google.colab import files
files.download('models/best_model.pth')
    """)


if __name__ == "__main__":
    main()
