"""
MNIST 训练配置
"""

# 数据配置
DATA_DIR = "./data"
NUM_CLASSES = 10
IMG_SIZE = 28

# 训练配置
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 设备配置
DEVICE = "cuda"  # 自动检测 GPU，如果没有则用 CPU

# 模型保存
MODEL_DIR = "./models"
SAVE_INTERVAL = 1  # 每几个 epoch 保存一次

# 随机种子（确保可复现）
SEED = 42
