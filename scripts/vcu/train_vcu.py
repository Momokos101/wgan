# scripts/vcu/train_vcu.py
# VCU 唤醒-休眠场景 WGAN-GP 训练脚本

import os
import sys
import argparse
import time

# ========== 1. 添加项目根目录到 sys.path ==========
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ========== 2. 导入全部配置 ==========
from configs.config_vcu import (
    # base
    FLAG, PROGRAM, OUTPUT_DIR, LOG_DIR, MODEL_PATH,
    PRECISION, C_DIM, MODEL_TYPE,

    # data
    DB_PATH, CONTEXT_BEFORE, CONTEXT_AFTER, SPLIT_RATIO,

    # model
    Z_DIM, EMBEDDING_DIM, G_DIM, D_DIM, OPT_TYPE,

    # train
    MAX_EPOCH, BATCH_SIZE, STAGEI_G_LR, STAGEI_D_LR,
    TRAIN_RATIO_I, DECAY, DECAY_EPOCHS
)

# ========== 3. 导入数据加载与 GAN 组件 ==========
from sequence.vcu_data_process import load_vcu_data
from nn.scale_model import ScaleModel
from nn.wd_trainer import WassersteinTrainer
from nn.optimizer import Optimizer


# ========== 工具 ==========
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def count_batches(dataset):
    return sum(1 for _ in dataset)


# ========== 4. 主训练流程 ==========
def main(args):
    print("=" * 60)
    print("VCU 唤醒-休眠场景 WGAN-GP 训练脚本")
    print("=" * 60)

    # ----- 4.1 加载数据 -----
    print("\n[1/4] 加载数据（含数据库 → 序列抽取 → npy 缓存）...")
    train_data, test_data, max_seq_len = load_vcu_data(
        precision=PRECISION,
        db_path=DB_PATH,
        output_dir=OUTPUT_DIR,
        limit=args.limit
    )

    print(f"训练集批次数: {count_batches(train_data)}")
    print(f"测试集批次数: {count_batches(test_data)}")
    print(f"最大序列长度: {max_seq_len}")

    # ----- 4.2 打印配置 -----
    print("\n[2/4] GAN 配置:")
    print(f"FLAG: {FLAG}")
    print(f"MODEL_TYPE: {MODEL_TYPE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"C_DIM: {C_DIM}")
    print(f"Z_DIM: {Z_DIM}")
    print(f"G_DIM / D_DIM: {G_DIM} / {D_DIM}")
    print(f"学习率: G={STAGEI_G_LR}, D={STAGEI_D_LR}")
    print(f"训练轮数: {MAX_EPOCH}")
    print(f"模型保存目录: {MODEL_PATH}")

    if not args.no_confirm:
        print("\n确认开始训练？Y开始，其他键退出：")
        try:
            s = input().strip()
        except EOFError:
            s = "Y"
        if s != "Y":
            print("训练已取消")
            return

    # ----- 4.3 构建模型 -----
    print("\n[3/4] 构建 ScaleModel...")
    scale_model = ScaleModel(seed_length=max_seq_len)
    scale_model.build()
    print("生成器:", scale_model.generator)
    print("判别器:", scale_model.discriminator)

    # ----- 4.4 初始化优化器 -----
    print("\n[4/4] 初始化优化器...")
    optimizer = Optimizer()
    optimizer.init_opt()
    print("优化器加载完成。")

    # ----- 创建训练器并开始训练 -----
    print("\n开始训练...")
    trainer = WassersteinTrainer(
        train_data=train_data,
        scale_model=scale_model,
        optimizer=optimizer,
        lambda_gp=10.0,       # 或从配置导入
    )

    t0 = time.time()
    trainer.train()
    t1 = time.time()

    print("\n训练完成！")
    print(f"总耗时: {t1 - t0:.1f} 秒")
    print(f"模型已保存至: {MODEL_PATH}")


# ========== 脚本入口 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-confirm", action="store_true")
    main(parser.parse_args())
