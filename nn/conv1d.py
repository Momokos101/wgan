"""
Conv1D GAN 模型实现

功能：
1. Conv1D 生成器：输入条件向量+噪声，输出电压序列
2. Conv1D 判别器：输入电压序列+条件向量，输出真实/虚假评分
3. 支持条件生成机制（WGAN-GP）
"""

import tensorflow as tf
from nn.model import NetModel


class Conv1D_Model(NetModel):
    """
    Conv1D 模型实现
    
    输入：
        - 生成器：condition (batch_size, C_DIM) + noise (batch_size, Z_DIM)
        - 判别器：voltage_sequence (batch_size, seq_length) + condition (batch_size, C_DIM)
    
    输出：
        - 生成器：voltage_sequence (batch_size, seq_length)
        - 判别器：logit (batch_size, 1) 真实/虚假评分
    """
    
    def __init__(self, batch_size, seed_length, c_dim, z_dim, ef_dim, g_dim, d_dim, precision, is_onedim=True):
        super(Conv1D_Model, self).__init__(
            batch_size, seed_length, c_dim, z_dim, ef_dim, g_dim, d_dim, precision, is_onedim
        )
        
    # ==================== 生成器 Generator ====================
    
    def generator_simple(self, z_var):
        """
        生成器核心网络
        
        输入：
            z_var: (batch_size, ef_dim + z_dim) 条件嵌入 + 噪声
        
        输出：
            generated_sequence: (batch_size, seq_length) 生成的电压序列
        
        网络结构：
            1. 全连接层将输入映射到高维空间
            2. 多层 Conv1DTranspose 上采样
            3. 最后输出序列长度为 self.s
        """
        # self.n 是根据序列长度计算的初始特征数
        # self.s 是最终序列长度
        # self.g_dim 是生成器基础维度
        
        # 第一层：全连接层，将输入扩展到 (batch_size, n * g_dim * 8)
        h0 = tf.keras.layers.Dense(self.n * self.g_dim * 8)(z_var)
        h0 = tf.keras.layers.BatchNormalization()(h0)
        h0 = tf.keras.layers.LeakyReLU(alpha=0.2)(h0)
        
        # 重塑为 3D 张量：(batch_size, n, g_dim * 8)
        h0 = tf.keras.layers.Reshape((self.n, self.g_dim * 8))(h0)
        
        # 第二层：Conv1DTranspose 上采样
        # 输出形状：(batch_size, n * 2, g_dim * 4)
        h1 = tf.keras.layers.Conv1DTranspose(
            filters=self.g_dim * 4,
            kernel_size=5,
            strides=2,
            padding='same'
        )(h0)
        h1 = tf.keras.layers.BatchNormalization()(h1)
        h1 = tf.keras.layers.LeakyReLU(alpha=0.2)(h1)
        
        # 第三层：Conv1DTranspose 上采样
        # 输出形状：(batch_size, n * 4, g_dim * 2)
        h2 = tf.keras.layers.Conv1DTranspose(
            filters=self.g_dim * 2,
            kernel_size=5,
            strides=2,
            padding='same'
        )(h1)
        h2 = tf.keras.layers.BatchNormalization()(h2)
        h2 = tf.keras.layers.LeakyReLU(alpha=0.2)(h2)
        
        # 第四层：Conv1DTranspose 上采样
        # 输出形状：(batch_size, n * 8, g_dim)
        h3 = tf.keras.layers.Conv1DTranspose(
            filters=self.g_dim,
            kernel_size=5,
            strides=2,
            padding='same'
        )(h2)
        h3 = tf.keras.layers.BatchNormalization()(h3)
        h3 = tf.keras.layers.LeakyReLU(alpha=0.2)(h3)
        
        # 第五层：Conv1D 生成最终序列
        # 输出形状：(batch_size, n * 8, 1)
        h4 = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=5,
            strides=1,
            padding='same',
            activation='tanh'  # tanh 输出范围 [-1, 1]
        )(h3)
        
        # 重塑为 2D 张量：(batch_size, n * 8)
        output = tf.keras.layers.Reshape((self.n * 8,))(h4)
        
        # 裁剪或填充到目标长度 self.s
        if self.n * 8 > self.s:
            # 如果生成的序列太长，裁剪
            output = output[:, :self.s]
        elif self.n * 8 < self.s:
            # 如果生成的序列太短，填充
            padding = self.s - self.n * 8
            output = tf.pad(output, [[0, 0], [0, padding]], mode='CONSTANT', constant_values=0)
        
        return output
    
    # ==================== 判别器 Discriminator ====================
    
    def d_encode_sample(self, x_var):
        """
        判别器编码电压序列
        
        输入：
            x_var: (batch_size, seq_length) 电压序列
        
        输出：
            encoded: (batch_size, features) 编码后的特征
        
        网络结构：
            1. 多层 Conv1D 下采样提取特征
            2. 输出特征向量
        """
        # 重塑为 3D 张量：(batch_size, seq_length, 1)
        h0 = tf.keras.layers.Reshape((self.s, 1))(x_var)
        
        # 第一层：Conv1D
        # 输出形状：(batch_size, seq_length/2, d_dim)
        h1 = tf.keras.layers.Conv1D(
            filters=self.d_dim,
            kernel_size=5,
            strides=2,
            padding='same'
        )(h0)
        h1 = tf.keras.layers.LeakyReLU(alpha=0.2)(h1)
        
        # 第二层：Conv1D
        # 输出形状：(batch_size, seq_length/4, d_dim * 2)
        h2 = tf.keras.layers.Conv1D(
            filters=self.d_dim * 2,
            kernel_size=5,
            strides=2,
            padding='same'
        )(h1)
        h2 = tf.keras.layers.BatchNormalization()(h2)
        h2 = tf.keras.layers.LeakyReLU(alpha=0.2)(h2)
        
        # 第三层：Conv1D
        # 输出形状：(batch_size, seq_length/8, d_dim * 4)
        h3 = tf.keras.layers.Conv1D(
            filters=self.d_dim * 4,
            kernel_size=5,
            strides=2,
            padding='same'
        )(h2)
        h3 = tf.keras.layers.BatchNormalization()(h3)
        h3 = tf.keras.layers.LeakyReLU(alpha=0.2)(h3)
        
        # 第四层：Conv1D
        # 输出形状：(batch_size, seq_length/16, d_dim * 8)
        h4 = tf.keras.layers.Conv1D(
            filters=self.d_dim * 8,
            kernel_size=5,
            strides=2,
            padding='same'
        )(h3)
        h4 = tf.keras.layers.BatchNormalization()(h4)
        h4 = tf.keras.layers.LeakyReLU(alpha=0.2)(h4)
        
        # 展平为 2D 张量
        encoded = tf.keras.layers.Flatten()(h4)
        
        return encoded
    
    def discriminator_simple(self, var, out_shape):
        """
        判别器核心网络
        
        输入：
            var: (batch_size, features + ef_dim) 编码特征 + 条件嵌入
            out_shape: 输出维度（通常为 1）
        
        输出：
            logit: (batch_size, out_shape) 真实/虚假评分
        """
        # 全连接层
        h0 = tf.keras.layers.Dense(512)(var)
        h0 = tf.keras.layers.LeakyReLU(alpha=0.2)(h0)
        h0 = tf.keras.layers.Dropout(0.3)(h0)
        
        h1 = tf.keras.layers.Dense(256)(h0)
        h1 = tf.keras.layers.LeakyReLU(alpha=0.2)(h1)
        h1 = tf.keras.layers.Dropout(0.3)(h1)
        
        # 输出层（不使用激活函数，输出原始 logit）
        output = tf.keras.layers.Dense(out_shape)(h1)
        
        return output
    
    # ==================== 判别器条件处理 ====================
    
    def build_discriminator_wc(self):
        """
        构建带条件的判别器
        
        输入：
            c_var: (batch_size, C_DIM) 条件向量
            x_var: (batch_size, seq_length) 电压序列
        
        输出：
            logit: (batch_size, 1) 真实/虚假评分
        """
        c_var = tf.keras.Input(shape=(self.c_dim,), name='condition')
        x_var = tf.keras.Input(shape=(self.s,), name='voltage_sequence')
        
        # 编码条件向量
        c_code = self.condition(c_var)
        
        # 编码电压序列
        x_code = self.d_encode_sample(x_var)
        
        # 拼接条件和序列特征
        combined = tf.keras.layers.Concatenate(axis=-1)([c_code, x_code])
        
        # 判别
        logit = self.discriminator_simple(combined, 1)
        
        self.discriminator_wi_condition = tf.keras.models.Model(
            inputs=[c_var, x_var],
            outputs=logit,
            name='discriminator_with_condition'
        )
