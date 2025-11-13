"""
VCU 项目主配置文件
自动导入所有子配置，保持向后兼容
所有现有代码仍然可以使用：from configs.config_vcu import *
"""
# 导入基础配置（共享配置）
from .config_vcu_base import (
    FLAG, PROGRAM, OUTPUT_DIR, LOG_DIR, MODEL_PATH,
    PRECISION, MAX_SEQUENCE_LENGTH, C_DIM,
    MODEL_TYPE, INDEX,
    CC2_MIN_VOLTAGE, CC2_MAX_VOLTAGE, SLEEP_VOLTAGE,
)

# 导入数据配置（后端负责人负责）
from .config_vcu_data import (
    DB_PATH,
    DB_PATHS,
    VEHICLE_STATUS_MIN,
    VEHICLE_STATUS_MAX,
    VEHICLE_STATUS_TOLERANCE,
    CONTEXT_BEFORE,
    CONTEXT_AFTER,
    SPLIT_RATIO,
)

# 导入模型配置（GAN架构负责人负责）
from .config_vcu_model import (
    Z_DIM,
    EMBEDDING_DIM,
    G_DIM,
    D_DIM,
    OPT_TYPE,
)

# 导入训练配置（GAN应用负责人负责）
from .config_vcu_train import (
    MAX_EPOCH,
    STAGEI_G_LR,
    STAGEI_D_LR,
    SHUFFLE,
    BATCH_SIZE,
    TRAIN_RATIO_I,
    TRAIN_RATIO_II,
    STAGEII_GI_LR,
    STAGEII_GII_LR,
    STAGEII_DII_LR,
    DECAY,
    DECAY_EPOCHS,
)

