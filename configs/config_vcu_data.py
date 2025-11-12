"""
VCU 数据相关配置
负责人：后端开发
包含：数据库路径、异常检测阈值、上下文提取参数等
"""
# 导入基础配置
from .config_vcu_base import *

# 数据库配置
DB_PATH = 'database/db.db'  # 数据库文件路径（相对于项目根目录）

# 异常检测阈值
VEHICLE_STATUS_MIN = 30  # 整车状态极小值，此时 READY 标志位应该为 0
VEHICLE_STATUS_MAX = 170  # 整车状态极大值，此时 READY 标志位应该为 1
VEHICLE_STATUS_TOLERANCE = 5  # 容差范围，用于判断是否接近极值

# 异常点上下文提取参数
CONTEXT_BEFORE = 4  # 向前提取的唤醒电压数量（N）
CONTEXT_AFTER = 4   # 向后提取的唤醒电压数量（M）

# 数据分割
SPLIT_RATIO = 0.8  # 训练集比例（80%训练，20%测试）

