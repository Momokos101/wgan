"""
用于快速验证：数据加载 + ScaleModel 构建 + Trainer 单步训练 是否能跑通
"""

import os
import sys

# 把项目根目录加入 sys.path（和 test_vcu_data.py 用法保持一致）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.vcu_data_process import load_vcu_data
from nn.scale_model import ScaleModel
from nn.optimizer import Optimizer
from nn.trainer import Trainer
from configs.config_vcu import *


def main():
    print("=" * 60)
    print("测试训练单步：数据管道 + ScaleModel + Trainer")
    print("=" * 60)

    # 1. 加载数据：与你的 vcu_data_process.py 对齐，返回 3 个值
    train_data, test_data, max_seq_len = load_vcu_data()

    print(f"max_seq_len = {max_seq_len}")
    # 取一个 batch 看看形状
    example_batch = next(iter(train_data))
    real_seq, condition, labels = example_batch
    print(f"  real_seq shape: {real_seq.shape}")
    print(f"  condition shape: {condition.shape}")
    print(f"  labels shape: {labels.shape}")

    # 2. 构造 ScaleModel（seed_length 用 max_seq_len）
    scale_model = ScaleModel(seed_length=max_seq_len)
    scale_model.build()
    print("ScaleModel 构建完成")

    # 3. 构造 Optimizer（注意：你的 Optimizer 不接受任何参数）
    optimizer = Optimizer()
    print("Optimizer 构建完成")

    # 4. 构造 Trainer
    trainer = Trainer(train_data, scale_model, optimizer,
                      lambda_kl=1.0, lambda_gp=10.0)
    print("Trainer 构建完成")

    # 5. 调用一次 D/G 的训练步
    #    这里第三个参数本来设计是 wrong_condition，
    #    但你当前数据管道返回的是 labels，这里先用 labels 占位，
    #    目前 Trainer 的 train_step_* 内部只是 stub，不会真正用到 w_c。
    d_log_vars = trainer.train_step_d(real_seq, condition, labels)
    g_log_vars = trainer.train_step_g(real_seq, condition)

    print("train_step_g 返回:", g_log_vars)
    print("train_step_d 返回:", d_log_vars)
    print("训练单步测试成功！")


if __name__ == "__main__":
    main()
