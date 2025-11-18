# scripts/vcu/generate_vcu.py
import os
import sys
import numpy as np
import tensorflow as tf

# ===== 把项目根目录和当前目录加入 sys.path =====
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# ===== 导入配置与模型 =====
from configs.config_vcu import (
    C_DIM, Z_DIM,
    MODEL_PATH,
    PRECISION,
)

from nn.scale_model import ScaleModel
from scoring import score_voltage_sequence


# =======================================
# 工具：扫描最新权重文件
# =======================================
def get_latest_generator_weight():
    all_files = os.listdir(MODEL_PATH)
    gen_files = [f for f in all_files if "generator" in f and f.endswith(".weights.h5")]
    if not gen_files:
        raise FileNotFoundError("未找到任何 generator 权重，请先训练模型")

    gen_files.sort(reverse=True)
    return os.path.join(MODEL_PATH, gen_files[0])


# =======================================
# 工具：构建 condition（9维）
# =======================================
def build_condition(
    abnormal_flag=1.0,       # 异常标志
    vehicle_state=0.5,       # 车辆运行状态归一化
    ready_flag=0.0,          # READY 标志
    anomaly_type=2,          # 异常类型编码（0~8）
    peak_ratio=0.4,
    max_ratio=0.3,
    boundary_ratio=0.2,
    oscillation_strength=0.5,
    voltage_range=0.4
):
    anomaly_type_norm = anomaly_type / 8.0

    condition = np.array([
        abnormal_flag,
        vehicle_state,
        ready_flag,
        anomaly_type_norm,
        peak_ratio,
        max_ratio,
        boundary_ratio,
        oscillation_strength,
        voltage_range
    ], dtype=np.float32)

    return condition.reshape(1, -1)   # shape (1, 9)


# =======================================
# 工具：反归一化
# =======================================
def denormalize_voltage(x):
    # 数据预处理阶段使用电压范围 [0, 4095]
    return np.clip(x * 4095, 0, 4095).astype(np.int32)


# =======================================
# 工具：转 HEX
# =======================================
def to_hex(seq):
    return [hex(int(v))[2:].zfill(3) for v in seq]


# =======================================
# 主生成 + 评分 + 排序函数
# =======================================
def generate_sequence(
    abnormal_type=2,
    sample_times=1,
    score_mode="default"
):
    """
    生成多条序列，并根据评分结果进行排序。

    参数:
        abnormal_type: 异常类型编码（0~8），会写入 condition 的第 4 个维度。
        sample_times: 生成样本数量。
        score_mode: 评分模式，目前支持 'default'。

    返回:
        list[dict]，按 total_score 从大到小排序，每个元素结构：
            {
                'float': [...],      # 归一化后的生成结果
                'voltage': [...],    # 0~4095 的整型电压
                'hex': [...],        # 对应的十六进制字符串
                'score': {           # 评分明细
                    'peak_ratio': ...,
                    'high_peak_ratio': ...,
                    'boundary_ratio': ...,
                    'std_norm': ...,
                    'range_norm': ...,
                    'total_score': ...
                }
            }
    """

    # ---- 1. 加载最新生成器 ----
    weight_path = get_latest_generator_weight()
    print("加载模型权重：", weight_path)

    model = ScaleModel(seed_length=8)
    model.build()
    model.generator.load_weights(weight_path)

    # ---- 2. 构造 condition ----
    c = build_condition(anomaly_type=abnormal_type)   # shape (1, 9)

    results = []

    # ---- 3. 多次采样 ----
    for i in range(sample_times):
        z = np.random.normal(size=(1, Z_DIM)).astype(np.float32)

        fake = model.generator([c, z], training=False).numpy()
        fake = fake.reshape(-1)           # shape (8,)
        volt = denormalize_voltage(fake)  # int 电压序列
        hex_seq = to_hex(volt)

        # 4. 评分
        score = score_voltage_sequence(fake)

        results.append({
            "float": fake.tolist(),
            "voltage": volt.tolist(),
            "hex": hex_seq,
            "score": score
        })

    # ---- 5. 按总评分从高到低排序 ----
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# =======================================
# CLI 调用
# =======================================
if __name__ == "__main__":
    out = generate_sequence(abnormal_type=2, sample_times=5, score_mode="default")
    print("\n按评分排序后的生成结果（HEX + score）：")
    for i, r in enumerate(out):
        print(f"[样本 {i}] score = {r['score']:.4f}, hex = {r['hex']}")
