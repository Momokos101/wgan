import os
try:
    from configs.config_vcu import *
except ImportError:
    from config_vcu import *
import tensorflow as tf
from datetime import datetime


class Trainer(object):
    def __init__(self, train_data, scale_model, optimizer, lambda_kl=1, lambda_gp=10):
        self.scale_model = scale_model
        self.train_data = train_data
        self.flag = FLAG
        self.max_epoch = MAX_EPOCH

        self.model_path = MODEL_PATH
        self.log_dir = LOG_DIR
        self.train_ratio_I = TRAIN_RATIO_I
        self.train_ratio_II = TRAIN_RATIO_II
        self.lambda_kl = lambda_kl
        self.lambda_gp = lambda_gp

        self.model_type = MODEL_TYPE
        self.decay_epochs = DECAY_EPOCHS

        # 优化器：根据配置初始化一阶/二阶优化器
        self.optimizer = optimizer
        self.optimizer.init_opt()

        now = datetime.now()
        self.date_time = now.strftime("%m%d%H%M%S")

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.log_name = 'train_' + self.date_time
        self.save_log = os.path.join(self.log_dir, self.log_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log = open(self.save_log, "a")

        self.log.write('flag:{}\n'.format(self.flag))
        self.log.write('max_epoch:{}\n'.format(self.max_epoch))
        self.log.write('train_ratio_I:{}\ntrain_ratio_II:{}\n'.format(self.train_ratio_I, self.train_ratio_II))
        self.log.write('G_LR:{}\nD_LR:{}\n'.format(STAGEI_G_LR, STAGEI_D_LR))
        self.log.write('precision:{}\n'.format(PRECISION))
        self.log.flush()

    def count_klloss(self, mu, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss

    def cross_entropy(self, label, pred, i):
        """
        简单的交叉熵实现（目前仅在 LWCA 场景中可能会用到）
        label = [1.0, 0, 0]
        pred  = [0.7, 0.1, 0.2]
        i     = 0 / 1 / 2
        期望 loss --> 0
        """
        loss = label * tf.math.log(pred + 1e-8)
        loss = tf.slice(loss, [0, i], [loss.shape[0], 1])
        loss = - tf.math.reduce_mean(loss)
        return loss

    # ---------------- WGAN-GP 相关工具函数 ----------------
    def gradient_penalty(self, real_seed, fake_seed, condition):
        """计算 WGAN-GP 的梯度惩罚项
        real_seed/fake_seed : (batch, seq_len)
        condition           : (batch, C_DIM)
        """
        batch_size = tf.shape(real_seed)[0]
        # 在真实样本和生成样本之间插值
        alpha = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
        diff = fake_seed - real_seed
        interpolated = real_seed + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            d_interpolated = self.scale_model.discriminator([condition, interpolated], training=True)
        grads = gp_tape.gradient(d_interpolated, interpolated)
        grads = tf.reshape(grads, [batch_size, -1])
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        gp = tf.reduce_mean((slopes - 1.0) ** 2)
        return gp

    # ---------------- 生成器训练步骤 ----------------
    def train_step_g_woc(self):
        """无条件生成（目前项目 FLAG 默认为 LWCO，这里保留占位实现）"""
        log_vars = []
        return log_vars

    def train_step_g_wc(self, c):
        """有条件生成器训练步骤（FLAG = 'LWCO' 使用）
        return:
            [('g_loss_fake', g_loss_fake)]
        """
        batch_size = tf.shape(c)[0]
        z = tf.random.normal(shape=(batch_size, Z_DIM))

        with tf.GradientTape() as tape:
            fake_seed = self.scale_model.generator([c, z], training=True)
            d_fake = self.scale_model.discriminator([c, fake_seed], training=True)
            # WGAN 生成器损失：希望判别器给生成样本高分
            g_loss_fake = -tf.reduce_mean(d_fake)

        grads = tape.gradient(g_loss_fake, self.scale_model.generator.trainable_variables)
        self.optimizer.stageI_g_opt.apply_gradients(zip(grads, self.scale_model.generator.trainable_variables))

        log_vars = [('g_loss_fake', g_loss_fake)]
        return log_vars

    def train_step_g_wca(self, c):
        """条件增强生成（LWCA 模式，当前项目未启用，保留占位）"""
        log_vars = []
        return log_vars

    # ---------------- 判别器训练步骤 ----------------
    def train_step_d_woc(self, lr_seeds):
        """无条件判别器训练（占位，实现留空）"""
        log_vars = []
        return log_vars

    def train_step_d_wc(self, lr_seeds, c, w_c):
        """有条件判别器训练步骤（FLAG = 'LWCO' 使用）
        args:
            lr_seeds: 真实电压序列 (batch, seq_len)
            c       : 条件向量       (batch, C_DIM)
            w_c     : 错误条件/标签，目前未在损失中使用，占位
        return:
            [
                ('d_loss', discriminator_loss),
                ('d_loss_real', d_loss_real),
                ('d_loss_fake', d_loss_fake),
                ('gp', gp),
            ]
        """
        batch_size = tf.shape(lr_seeds)[0]
        z = tf.random.normal(shape=(batch_size, Z_DIM))

        with tf.GradientTape() as tape:
            # 生成假样本
            fake_seed = self.scale_model.generator([c, z], training=True)

            # 判别器在真实/伪造样本上的输出
            d_real = self.scale_model.discriminator([c, lr_seeds], training=True)
            d_fake = self.scale_model.discriminator([c, fake_seed], training=True)

            # WGAN 损失：E[D(fake)] - E[D(real)]
            d_loss_real = -tf.reduce_mean(d_real)
            d_loss_fake = tf.reduce_mean(d_fake)

            # 梯度惩罚
            gp = self.gradient_penalty(lr_seeds, fake_seed, c)

            discriminator_loss = d_loss_real + d_loss_fake + self.lambda_gp * gp

        grads = tape.gradient(discriminator_loss, self.scale_model.discriminator.trainable_variables)
        self.optimizer.stageI_d_opt.apply_gradients(zip(grads, self.scale_model.discriminator.trainable_variables))

        log_vars = [
            ('d_loss', discriminator_loss),
            ('d_loss_real', d_loss_real),
            ('d_loss_fake', d_loss_fake),
            ('gp', gp),
        ]
        return log_vars

    def train_step_d_wca(self, lr_seeds, c, w_c):
        """条件增强判别器训练（LWCA 模式，占位）"""
        log_vars = []
        return log_vars

    # ---------------- 统一入口 ----------------
    def train_step_g(self, lr_seeds, c):
        if self.flag == 'LWOC':
            return self.train_step_g_woc()
        elif self.flag == 'LWCO':
            return self.train_step_g_wc(c)
        else:  # self.flag == 'LWCA':
            return self.train_step_g_wca(c)

    def train_step_d(self, lr_seeds, c, w_c):
        if self.flag == 'LWOC':
            return self.train_step_d_woc(lr_seeds)
        elif self.flag == 'LWCO':
            return self.train_step_d_wc(lr_seeds, c, w_c)
        else:  # self.flag == 'LWCA':
            return self.train_step_d_wca(lr_seeds, c, w_c)

    # ---------------- 训练主循环 ----------------
    def train(self):
        cnt = 0
        g_log_vars = []
        d_log_vars = []

        for epoch in range(self.max_epoch):
            for real_seed, condition, labels in self.train_data:
                wrong_condition = tf.roll(condition, shift=1, axis=0)
                cnt += 1
                if cnt % self.train_ratio_I != 0:
                    g_log_vars = self.train_step_g(real_seed, condition)
                else:
                    d_log_vars = self.train_step_d(real_seed, condition, wrong_condition)

            # 记录日志（只记录最后一个 batch 的指标）
            self.log.write('Epoch: {}\n'.format(epoch + 1))
            for k, v in g_log_vars:
                # v 是 tensor，转成 float 方便阅读
                try:
                    v_val = float(v.numpy())
                except Exception:
                    v_val = float(v)
                self.log.write('{}: {} '.format(k, v_val))
            self.log.write('\n')
            for k, v in d_log_vars:
                try:
                    v_val = float(v.numpy())
                except Exception:
                    v_val = float(v)
                self.log.write('{}: {} '.format(k, v_val))
            self.log.write('\n')
            self.log.flush()

            # 学习率更新
            if (epoch + 1) % self.decay_epochs == 0:
                self.optimizer.lr_decay('I')

        # 训练结束后保存模型
        self.save_model()

    def save_model(self):
        submodel_names = ['generator', 'discriminator']
        submodels = [self.scale_model.generator, self.scale_model.discriminator]
        for i in range(2):
            submodel_name = submodel_names[i]
            submodel = submodels[i]
            model_name = '{}_{}_{}_{}'.format(self.date_time, self.model_type, self.flag, submodel_name)
            print('model_name:', model_name)
            model_path = os.path.join(self.model_path, '{}.weights.h5'.format(model_name))
            print('model_path:', model_path)
            submodel.save_weights(model_path)
