# MNIST 手写数字分类

使用 PyTorch 实现的 MNIST 手写数字识别项目。

## 📊 项目结构

```
mnist/
├── data/               # 数据目录（自动下载）
├── models/             # 保存的模型
├── src/
│   ├── __init__.py
│   ├── dataset.py      # 数据加载
│   ├── model.py        # 模型定义
│   └── train.py        # 训练脚本
├── requirements.txt    # 依赖
├── config.py          # 配置
└── README.md
```

## 🚀 快速开始

### Colab 执行
```python
# 1. 克隆仓库
!git clone https://github.com/ManongOne/ml-project.git
%cd ml-project/mnist
!pip install -r requirements.txt

# 2. 运行训练
!python src/train.py
```

### 本地执行
```bash
pip install -r requirements.txt
python src/train.py
```

## 📈 模型架构

- **输入**: 28x28 灰度图像
- **网络**: ConvNet (2 个卷积层 + 2 个全连接层)
- **输出**: 10 个数字类别 (0-9)
- **优化器**: Adam
- **损失函数**: CrossEntropyLoss

## 🎯 预期结果

- **训练准确率**: >98%
- **测试准确率**: >97%
- **训练时间**: ~5 分钟 (Colab GPU)

## 📝 配置

编辑 `config.py` 调整参数：
- `EPOCHS`: 训练轮数
- `BATCH_SIZE`: 批次大小
- `LEARNING_RATE`: 学习率
- `DEVICE`: 训练设备 (cuda/cpu)

## 📦 依赖

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib

## 📄 License

MIT License
