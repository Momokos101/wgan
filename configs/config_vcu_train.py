"""
VCU 训练相关配置
负责人：GAN应用开发（成员B）
包含：训练参数、学习率、批次大小等
"""
# 导入基础配置
from .config_vcu_base import *

# 训练参数
MAX_EPOCH = 100
STAGEI_G_LR = 1e-4 
STAGEI_D_LR = 1e-4  # 降低判别器学习率以避免数值不稳定
OPT_TYPE = 'adam'
CHECKPOINT_DIR = "checkpoints"

# 批次大小
SHUFFLE = True
BATCH_SIZE = 16 # VCU 数据可能较少，使用较小的批次

# 训练比例
TRAIN_RATIO_I = 2
TRAIN_RATIO_II = 2

# 第二阶段学习率
STAGEII_GI_LR = 1e-4
STAGEII_GII_LR = 1e-4
STAGEII_DII_LR = 7e-5

# 学习率衰减
DECAY = 0.50
DECAY_EPOCHS = 10

