"""
GAN 架构测试脚本
测试 Conv1D 生成器和判别器是否正确构建
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from configs.config_vcu import *
from nn.scale_model import ScaleModel
from nn.optimizer import Optimizer


def test_model_build():
    """测试模型构建"""
    print("=" * 80)
    print("测试 1: 模型构建")
    print("=" * 80)
    
    # 假设序列长度为 100
    seed_length = 100
    
    # 构建模型
    print(f"\n创建 ScaleModel，序列长度: {seed_length}")
    scale_model = ScaleModel(seed_length=seed_length)
    scale_model.build()
    
    print(f"✓ 模型构建成功！")
    print(f"  - 生成器名称: {scale_model.generator.name}")
    print(f"  - 判别器名称: {scale_model.discriminator.name}")
    print(f"  - FLAG 模式: {scale_model.flag}")
    
    return scale_model


def test_generator(scale_model):
    """测试生成器"""
    print("\n" + "=" * 80)
    print("测试 2: 生成器 (Generator)")
    print("=" * 80)
    
    # 创建测试条件向量
    batch_size = 4
    condition = np.random.randn(batch_size, C_DIM).astype(np.float32)
    
    print(f"\n输入条件向量形状: {condition.shape}")
    print(f"条件向量维度说明: C_DIM={C_DIM}")
    print(f"  - 基础条件 (3): 异常标志 + 整车状态归一化 + READY标志位")
    print(f"  - 异常类型编码 (1): 0=normal, 1=state_mismatch, 2=error, 3=stuck, 4=ready_mismatch")
    print(f"  - 电压特征 (5): 震荡顶峰 + 连续极大值 + 边界值 + 震荡强度 + 电压范围")
    
    # 生成序列
    generated = scale_model.model.generate_wc(condition)
    
    print(f"\n生成器输出形状: {generated.shape}")
    print(f"✓ 生成器测试成功！")
    print(f"\n生成的电压序列示例（第一个样本的前10个值）:")
    print(generated[0, :10].numpy())
    
    return generated


def test_discriminator(scale_model, generated_sequences):
    """测试判别器"""
    print("\n" + "=" * 80)
    print("测试 3: 判别器 (Discriminator)")
    print("=" * 80)
    
    batch_size = generated_sequences.shape[0]
    condition = np.random.randn(batch_size, C_DIM).astype(np.float32)
    
    print(f"\n输入条件向量形状: {condition.shape}")
    print(f"输入序列形状: {generated_sequences.shape}")
    
    # 判别序列
    logits = scale_model.model.discriminate_wc([condition, generated_sequences])
    
    print(f"\n判别器输出形状: {logits.shape}")
    print(f"✓ 判别器测试成功！")
    print(f"\n判别器输出 logits（真实/虚假评分）:")
    print(logits.numpy().flatten())
    
    return logits


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "=" * 80)
    print("测试 4: 梯度流 (Gradient Flow)")
    print("=" * 80)
    
    print("\n⚠️  注意: 由于 TensorFlow Metal 后端的已知问题,梯度测试将在 CPU 上运行")
    print("这不影响实际训练,训练时可以正常使用 GPU 加速\n")
    
    # 强制使用CPU进行梯度测试(避免MPS bug)
    with tf.device('/CPU:0'):
        seed_length = 100
        scale_model = ScaleModel(seed_length=seed_length)
        scale_model.build()
        
        # 创建优化器
        optimizer = Optimizer()
        optimizer.init_opt()
        
        # 创建测试数据
        batch_size = 4
        condition = tf.random.normal((batch_size, C_DIM))
        real_sequences = tf.random.normal((batch_size, seed_length))
        
        # 测试生成器梯度
        print("测试生成器梯度...")
        with tf.GradientTape() as tape:
            fake_sequences = scale_model.model.generate_wc(condition)
            fake_logits = scale_model.model.discriminate_wc([condition, fake_sequences])
            g_loss = -tf.reduce_mean(fake_logits)
        
        g_gradients = tape.gradient(g_loss, scale_model.generator.trainable_variables)
        print(f"✓ 生成器梯度计算成功！")
        print(f"  - 损失值: {g_loss.numpy():.4f}")
        print(f"  - 可训练变量数量: {len(scale_model.generator.trainable_variables)}")
        print(f"  - 梯度数量: {len([g for g in g_gradients if g is not None])}")
        
        # 测试判别器梯度
        print("\n测试判别器梯度...")
        with tf.GradientTape() as tape:
            real_logits = scale_model.model.discriminate_wc([condition, real_sequences])
            fake_sequences = scale_model.model.generate_wc(condition)
            fake_logits = scale_model.model.discriminate_wc([condition, fake_sequences])
            d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
        
        d_gradients = tape.gradient(d_loss, scale_model.discriminator.trainable_variables)
        print(f"✓ 判别器梯度计算成功！")
        print(f"  - 损失值: {d_loss.numpy():.4f}")
        print(f"  - 可训练变量数量: {len(scale_model.discriminator.trainable_variables)}")
        print(f"  - 梯度数量: {len([g for g in d_gradients if g is not None])}")


def test_wgan_gp():
    """测试 WGAN-GP 梯度惩罚"""
    print("\n" + "=" * 80)
    print("测试 5: WGAN-GP 梯度惩罚")
    print("=" * 80)
    
    print("\n⚠️  注意: 梯度惩罚测试也在 CPU 上运行\n")
    
    with tf.device('/CPU:0'):
        seed_length = 100
        scale_model = ScaleModel(seed_length=seed_length)
        scale_model.build()
        
        batch_size = 4
        condition = tf.random.normal((batch_size, C_DIM))
        real_sequences = tf.random.normal((batch_size, seed_length))
        
        print("计算梯度惩罚...")
        with tf.GradientTape() as tape:
            # 生成假序列
            fake_sequences = scale_model.model.generate_wc(condition)
            
            # 插值
            alpha = tf.random.uniform(real_sequences.shape, 0.0, 1.0)
            inter_sequences = fake_sequences * alpha + real_sequences * (1 - alpha)
            
            # 计算插值序列的梯度
            with tf.GradientTape() as tape_gp:
                tape_gp.watch(inter_sequences)
                inter_logits = scale_model.model.discriminate_wc([condition, inter_sequences])
            
            gp_gradients = tape_gp.gradient(inter_logits, inter_sequences)
            gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis=[1]))
            gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)
        
        print(f"✓ 梯度惩罚计算成功！")
        print(f"  - 梯度惩罚值: {gp.numpy():.4f}")
        print(f"  - 梯度范数: {gp_gradients_norm.numpy()}")


def test_model_summary():
    """打印模型摘要"""
    print("\n" + "=" * 80)
    print("测试 6: 模型摘要")
    print("=" * 80)
    
    seed_length = 100
    scale_model = ScaleModel(seed_length=seed_length)
    scale_model.build()
    
    print("\n生成器结构:")
    print("-" * 80)
    scale_model.generator.summary()
    
    print("\n判别器结构:")
    print("-" * 80)
    scale_model.discriminator.summary()


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("GAN 架构测试开始")
    print("=" * 80)
    print(f"\n配置信息:")
    print(f"  - FLAG: {FLAG}")
    print(f"  - MODEL_TYPE: {MODEL_TYPE}")
    print(f"  - BATCH_SIZE: {BATCH_SIZE}")
    print(f"  - C_DIM: {C_DIM}")
    print(f"  - Z_DIM: {Z_DIM}")
    print(f"  - EMBEDDING_DIM: {EMBEDDING_DIM}")
    print(f"  - G_DIM: {G_DIM}")
    print(f"  - D_DIM: {D_DIM}")
    
    try:
        # 测试 1: 模型构建
        scale_model = test_model_build()
        
        # 测试 2: 生成器
        generated = test_generator(scale_model)
        
        # 测试 3: 判别器
        test_discriminator(scale_model, generated)
        
        # 测试 4: 梯度流
        test_gradient_flow()
        
        # 测试 5: WGAN-GP
        test_wgan_gp()
        
        # 测试 6: 模型摘要
        test_model_summary()
        
        print("\n" + "=" * 80)
        print("✓ 所有测试通过！GAN 架构正常工作")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ 测试失败！")
        print("=" * 80)
        print(f"\n错误信息: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
