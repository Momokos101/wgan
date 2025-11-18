"""
VCU 模型相关配置
负责人：GAN架构开发（成员A）
包含：模型架构参数、网络维度、优化器类型等
"""
# 导入基础配置
from .config_vcu_base import *

# 噪声和嵌入维度
Z_DIM = 128
EMBEDDING_DIM = 128

# 生成器和判别器维度
G_DIM = 64
D_DIM = 32

# 优化器类型
OPT_TYPE = 'adam'

