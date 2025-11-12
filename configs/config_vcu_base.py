"""
VCU 项目基础配置（共享配置）
所有团队成员共享的基础配置，修改前需要讨论
"""
import os

# 项目模式：VCU 唤醒-休眠测试
FLAG = 'LWCO'  # 使用条件生成模式
PROGRAM = 'vcu'

# 数据输出目录
OUTPUT_DIR = 'data/' + PROGRAM
LOG_DIR = 'logs/' + PROGRAM
MODEL_PATH = 'model_weights/' + PROGRAM

# 数据精度（保留用于兼容性）
PRECISION = '2'

# 最大序列长度（动态调整）
MAX_SEQUENCE_LENGTH = 100  # 初始值，会根据实际数据调整

# 条件维度：基础条件(3) + 异常类型编码(1) + 电压特征(5) = 9
# 基础条件：异常标志(1) + 整车状态归一化(1) + READY标志位(1)
# 异常类型编码：normal(0), state_follow_mismatch(1), error(2), stuck(3), ready_flag_mismatch(4)
# 电压特征：震荡顶峰比例(1) + 连续极大值比例(1) + 边界值比例(1) + 震荡强度(1) + 电压范围(1)
C_DIM = 9

# 模型参数
MODEL_TYPE = 'conv1d'
INDEX = 'vcu'

# CC2 电压范围（基础范围，数据相关）
CC2_MIN_VOLTAGE = 4.8
CC2_MAX_VOLTAGE = 7.8
SLEEP_VOLTAGE = 12.0  # 休眠时的固定电压

