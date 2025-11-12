import numpy as np
import tensorflow as tf
from nn.conv1d import Conv1D_Model
try:
    from configs.config_vcu import *
except ImportError:
    from config_vcu import *


class ScaleModel(tf.keras.Model):
    def __init__(self, seed_length):
        super(ScaleModel, self).__init__()
        self.model_type = MODEL_TYPE
        g_dim = G_DIM
        z_dim = Z_DIM
        c_dim = C_DIM
        ef_dim = EMBEDDING_DIM
        batch_size = BATCH_SIZE
        d_dim = D_DIM
        precision = PRECISION
        
        # 对于 VCU 场景，precision 只是用于兼容性，实际序列长度就是 seed_length
        # 但 NetModel 会根据 precision 计算 self.s，所以需要调整
        # 如果 precision='2'，NetModel 会计算 self.s = seed_length * 4
        # 但实际数据长度就是 seed_length，所以传入 seed_length/4 给 NetModel
        if self.model_type == "conv1d":
            # 根据 precision 调整传入的 seed_length，使 NetModel 计算的 self.s 等于实际序列长度
            if precision == '2':
                adjusted_seed_length = seed_length // 4  # 这样 self.s = (seed_length//4) * 4 = seed_length
            elif precision == '4':
                adjusted_seed_length = seed_length // 2
            elif precision == '8':
                adjusted_seed_length = seed_length
            elif precision == '16':
                adjusted_seed_length = seed_length * 2
            elif precision == '1':
                adjusted_seed_length = seed_length // 8
            else:
                adjusted_seed_length = seed_length
            
            self.model = Conv1D_Model(batch_size, adjusted_seed_length, c_dim, z_dim, ef_dim, g_dim, d_dim, precision)
        else:
            pass
        self.flag = FLAG

    def build(self):
        if self.flag == 'LWOC':
            self.model.build_generator_woc()
            self.model.build_discriminator_woc()
            self.generator = self.model.generator_wo_condition
            self.discriminator = self.model.discriminator_wo_condition

        elif self.flag == 'LWCO':
            self.model.build_generator_wc()
            self.model.build_discriminator_wc()
            self.generator = self.model.generator_wi_condition
            self.discriminator = self.model.discriminator_wi_condition

        elif self.flag == 'LWCA':
            self.model.build_generator_wca()
            self.model.build_discriminator_wc()
            self.generator = self.model.generator_wi_conaugment
            self.discriminator = self.model.discriminator_wi_condition

        else:
            pass

