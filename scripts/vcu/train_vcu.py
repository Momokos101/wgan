"""
VCU 唤醒-休眠场景 GAN 训练脚本
"""
import os
import sys
import numpy as np
import tensorflow as tf
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.vcu_data_process import load_vcu_data
from nn.wd_trainer import WassersteinTrainer
from nn.scale_model import ScaleModel
from configs.config_vcu import *
from nn.optimizer import Optimizer


class VcuWassersteinTrainer(WassersteinTrainer):
    """适配 VCU 数据格式的训练器"""
    
    def train(self):
        """重写训练循环以适配 VCU 数据格式"""
        cnt = 0
        for epoch in range(self.max_epoch):
            for voltage_seq, condition, abnormal_label in self.train_data:
                cnt += 1
                
                # 生成错误的条件（用于判别器训练）
                # 可以随机生成或使用异常条件
                wrong_condition = tf.random.uniform(
                    shape=condition.shape,
                    minval=0.0,
                    maxval=1.0
                )
                
                if cnt % self.train_ratio_I != 0:
                    g_log_vars = self.train_step_g(voltage_seq, condition)
                else:
                    d_log_vars = self.train_step_d(voltage_seq, condition, wrong_condition)
            
            self.log.write('Epoch: {}\n'.format(epoch + 1))
            for k, v in g_log_vars:
                g_loss_fake = v
                self.log.write('{}: {} '.format(k, v))
            self.log.write('\n')
            for k, v in d_log_vars:
                self.log.write('{}: {} '.format(k, v))
            self.log.write('\n')
            self.log.flush()

            # 学习率更新
            if (epoch + 1) % self.decay_epochs == 0:
                self.optimizer.lr_decay('I')
                
        self.save_model()


def main():
    print("=" * 60)
    print("VCU 唤醒-休眠场景 GAN 训练")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    train_data, test_data, max_seq_len = load_vcu_data(
        precision=PRECISION,
        db_path=DB_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # 更新全局配置中的最大序列长度
    import config_vcu
    config_vcu.MAX_SEQUENCE_LENGTH = max_seq_len
    
    print(f"数据加载完毕！")
    print(f"  训练集批次: {len(list(train_data))}")
    print(f"  测试集批次: {len(list(test_data))}")
    print(f"  最大序列长度: {max_seq_len}")
    
    # 显示配置信息
    print("\n[2/4] 配置信息:")
    print(f"  FLAG: {FLAG}")
    print(f"  PROGRAM: {PROGRAM}")
    print(f"  MAX_EPOCH: {MAX_EPOCH}")
    print(f"  G_LR: {STAGEI_G_LR}")
    print(f"  D_LR: {STAGEI_D_LR}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  C_DIM: {C_DIM}")
    print(f"  Z_DIM: {Z_DIM}")
    
    # 确认开始训练
    print("\n请确认以上信息是否正确，输入 Y 继续，其他键退出:")
    try:
        s = input()
    except EOFError:
        # 非交互式环境，自动确认
        print("非交互式环境，自动确认继续...")
        s = 'Y'
    if s != 'Y':
        print("训练已取消")
        exit()
    
    # 构建模型
    print("\n[3/4] 构建模型...")
    model = ScaleModel(max_seq_len)
    model.build()
    print("模型构建完成！")
    
    # 初始化优化器
    print("\n[4/4] 初始化优化器...")
    optimizer = Optimizer()
    optimizer.init_opt()
    print("优化器初始化完成！")
    
    # 创建训练器
    print("\n开始训练...")
    trainer = VcuWassersteinTrainer(train_data, model, optimizer)
    trainer.train()
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()

