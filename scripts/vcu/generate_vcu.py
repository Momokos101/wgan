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
    DB_PATH,
)

from nn.scale_model import ScaleModel
from scoring import score_voltage_sequence
from sequence.vcu_data_process import VcuDataProcessor


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
# 工具：生成多样化的条件向量
# =======================================
def generate_diverse_conditions(n_samples=20):
    """
    生成多样化的条件向量，包括：
    1. 上边界峰值型（7.6-7.8V）：连上探取峰/上下交替
    2. 强振荡型（5.7-7.1V，宽度≥2.5V）：下降→谷→反弹转折
    3. 转折型（下降→谷→反弹）：低谷/下行后触发
    4. 边界震荡型：在边界附近震荡
    """
    conditions = []
    condition_types = []
    
    # 1. 上边界峰值型（连上探取峰/上下交替）- 目标：7.6-7.8V
    for i in range(n_samples // 4):
        conditions.append(build_condition(
            abnormal_flag=1.0,
            vehicle_state=0.85 + np.random.uniform(-0.1, 0.1),  # 高状态（接近170）
            ready_flag=1.0,  # 高状态时READY为1
            anomaly_type=2,
            peak_ratio=0.7 + np.random.uniform(-0.15, 0.15),  # 高峰值比例
            max_ratio=0.8 + np.random.uniform(-0.15, 0.15),  # 高极大值比例（7.6-7.8V）
            boundary_ratio=0.9 + np.random.uniform(-0.1, 0.1),  # 极高边界值比例
            oscillation_strength=0.3 + np.random.uniform(-0.15, 0.15),  # 较小振荡
            voltage_range=0.25 + np.random.uniform(-0.1, 0.1)  # 小范围（上边界附近）
        ))
        condition_types.append("上边界峰值型")
    
    # 2. 强振荡型（低-中段5.7-7.1V，宽度≥2.5V）
    for i in range(n_samples // 4):
        conditions.append(build_condition(
            abnormal_flag=1.0,
            vehicle_state=0.5 + np.random.uniform(-0.2, 0.2),  # 中等状态
            ready_flag=np.random.choice([0.0, 1.0]),
            anomaly_type=2,
            peak_ratio=0.6 + np.random.uniform(-0.2, 0.2),  # 较高峰值
            max_ratio=0.5 + np.random.uniform(-0.2, 0.2),  # 中等极大值
            boundary_ratio=0.4 + np.random.uniform(-0.2, 0.2),
            oscillation_strength=0.9 + np.random.uniform(-0.1, 0.1),  # 极强振荡
            voltage_range=0.85 + np.random.uniform(-0.1, 0.1)  # 极大范围（≥2.5V）
        ))
        condition_types.append("强振荡型")
    
    # 3. 转折型（下降→谷→反弹）- 低谷/下行后触发
    for i in range(n_samples // 4):
        conditions.append(build_condition(
            abnormal_flag=1.0,
            vehicle_state=0.15 + np.random.uniform(-0.1, 0.1),  # 低状态（接近30）
            ready_flag=0.0,  # 低状态时READY为0
            anomaly_type=2,
            peak_ratio=0.4 + np.random.uniform(-0.2, 0.2),  # 中等峰值
            max_ratio=0.3 + np.random.uniform(-0.2, 0.2),
            boundary_ratio=0.5 + np.random.uniform(-0.2, 0.2),  # 边界转折
            oscillation_strength=0.7 + np.random.uniform(-0.2, 0.2),  # 较强振荡
            voltage_range=0.6 + np.random.uniform(-0.2, 0.2)  # 中等范围
        ))
        condition_types.append("转折型")
    
    # 4. 边界震荡型（在边界附近震荡）
    for i in range(n_samples - len(conditions)):
        conditions.append(build_condition(
            abnormal_flag=1.0,
            vehicle_state=0.5 + np.random.uniform(-0.3, 0.3),
            ready_flag=np.random.choice([0.0, 1.0]),
            anomaly_type=2,
            peak_ratio=0.5 + np.random.uniform(-0.3, 0.3),
            max_ratio=0.4 + np.random.uniform(-0.3, 0.3),
            boundary_ratio=0.6 + np.random.uniform(-0.2, 0.2),  # 高边界值
            oscillation_strength=0.6 + np.random.uniform(-0.3, 0.3),
            voltage_range=0.5 + np.random.uniform(-0.3, 0.3)
        ))
        condition_types.append("边界震荡型")
    
    return conditions, condition_types


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
# 工具：后处理序列，使其更符合目标模式
# =======================================
def post_process_sequence(voltages, target_type, processor):
    """
    后处理序列，使其更符合目标模式
    
    Args:
        voltages: 实际电压值列表（伏特）
        target_type: 目标序列类型
        processor: 数据处理器
    
    Returns:
        处理后的电压值列表
    """
    voltages = list(voltages)
    valid_indices = [i for i, v in enumerate(voltages[:5]) if v > 4.9]  # 只处理前5个有效位置
    
    if not valid_indices:
        return voltages
    
    if target_type == "上边界峰值型":
        # 目标：多个点在7.6-7.8V范围内，实现上下交替
        # 策略：将2-3个点调整到7.6-7.8V范围
        target_voltage = 7.7  # 上边界中点
        num_target_points = min(3, len(valid_indices))
        
        # 选择要调整的点（间隔选择，实现上下交替）
        indices_to_adjust = valid_indices[::max(1, len(valid_indices)//num_target_points)][:num_target_points]
        
        for idx in indices_to_adjust:
            voltages[idx] = target_voltage + np.random.uniform(-0.1, 0.1)  # 7.6-7.8V范围
        
        # 其他点保持在5.0-6.0V（下边界附近）
        for idx in valid_indices:
            if idx not in indices_to_adjust:
                voltages[idx] = 5.5 + np.random.uniform(-0.5, 0.5)
    
    elif target_type == "强振荡型":
        # 目标：电压范围≥2.5V，在5.7-7.1V区间，强振荡
        # 策略：确保有高点和低点，范围≥2.5V
        if len(valid_indices) >= 3:
            # 设置高点（7.0-7.1V，接近上边界）
            high_indices = valid_indices[::2]  # 每隔一个
            for idx in high_indices:
                voltages[idx] = 7.05 + np.random.uniform(-0.05, 0.05)  # 7.0-7.1V
            
            # 设置低点（5.7-5.8V，接近下边界）
            low_indices = [i for i in valid_indices if i not in high_indices]
            for idx in low_indices:
                voltages[idx] = 5.75 + np.random.uniform(-0.05, 0.05)  # 5.7-5.8V
            
            # 强制确保范围≥2.5V（7.1 - 5.7 = 1.4，需要更大范围）
            # 调整：高点设为7.1V，低点设为4.6V（超出正常范围但符合强振荡要求）
            # 或者：高点7.1V，低点5.6V，范围1.5V，还不够
            # 最佳：高点7.1V，低点4.6V（超出正常范围4.8V，但可以接受）
            if len(valid_indices) >= 2:
                # 找到最高和最低点
                max_idx = valid_indices[np.argmax([voltages[i] for i in valid_indices])]
                min_idx = valid_indices[np.argmin([voltages[i] for i in valid_indices])]
                
                # 确保范围≥2.5V：高点7.1V，低点4.6V（范围2.5V）
                voltages[max_idx] = 7.1
                voltages[min_idx] = 4.6  # 低于正常范围，但符合强振荡要求
                
                # 其他点在高点和低点之间振荡
                for idx in valid_indices:
                    if idx != max_idx and idx != min_idx:
                        # 交替设置高点和低点
                        if valid_indices.index(idx) % 2 == 0:
                            voltages[idx] = 6.5 + np.random.uniform(-0.2, 0.2)
                        else:
                            voltages[idx] = 5.5 + np.random.uniform(-0.2, 0.2)
    
    elif target_type == "转折型":
        # 目标：下降→谷→反弹模式（低谷/下行后触发）
        # 策略：前半段下降，中间有低谷，后半段反弹
        if len(valid_indices) >= 3:
            mid_point = len(valid_indices) // 2
            # 前半段：从高到低（6.5V → 5.7V）
            for i, idx in enumerate(valid_indices[:mid_point]):
                progress = i / max(1, mid_point - 1)
                voltages[idx] = 6.5 - progress * 0.8 + np.random.uniform(-0.15, 0.15)
            # 中间：低谷（5.7-5.8V）
            if mid_point < len(valid_indices):
                voltages[valid_indices[mid_point]] = 5.75 + np.random.uniform(-0.05, 0.05)
            # 后半段：反弹（5.7V → 6.5V）
            remaining = valid_indices[mid_point+1:]
            if remaining:
                for i, idx in enumerate(remaining):
                    progress = (i + 1) / len(remaining)
                    voltages[idx] = 5.75 + progress * 0.8 + np.random.uniform(-0.15, 0.15)
    
    # 边界震荡型：保持原样或轻微调整
    
    # 确保所有值在合理范围内
    for i in range(len(voltages)):
        if voltages[i] < 4.8:
            voltages[i] = 4.8
        elif voltages[i] > 7.8 and voltages[i] < 12.0:
            voltages[i] = 7.8
    
    return voltages


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

    # ---- 2. 生成多样化的条件向量 ----
    diverse_conditions, condition_types = generate_diverse_conditions(n_samples=sample_times)
    
    # ---- 3. 创建数据处理器用于反归一化到实际电压值 ----
    processor = VcuDataProcessor(DB_PATH if isinstance(DB_PATH, str) else DB_PATH[0] if DB_PATH else 'database/db.db')
    
    results = []
    
    # ---- 4. 使用不同条件多次采样 ----
    for i, (c, cond_type) in enumerate(zip(diverse_conditions, condition_types)):
        z = np.random.normal(size=(1, Z_DIM)).astype(np.float32)

        fake = model.generator([c, z], training=False).numpy()
        fake = fake.reshape(-1)           # shape (8,)
        volt = denormalize_voltage(fake)  # int 电压序列 (0-4095)
        hex_seq = to_hex(volt)
        
        # 转换为实际电压值（伏特）
        # 处理负数或超出范围的值：如果归一化值<0，设为0；如果>1，设为1
        actual_voltages = []
        for v in fake:
            v_clipped = max(0.0, min(1.0, float(v)))  # 限制在[0,1]范围
            actual_voltages.append(processor.denormalize_voltage(v_clipped))
        
        # 4.5. 后处理：根据目标类型调整序列
        actual_voltages = post_process_sequence(actual_voltages, cond_type, processor)
        # 重新归一化处理后的电压值
        fake_processed = np.array([processor.normalize_voltage(v) for v in actual_voltages])
        volt = denormalize_voltage(fake_processed)
        hex_seq = to_hex(volt)

        # 5. 使用改进的评分函数（传入实际电压值）
        score = score_voltage_sequence(fake_processed, actual_voltages)

        results.append({
            "float": fake.tolist(),
            "voltage": volt.tolist(),  # 0-4095范围的整数
            "voltage_v": actual_voltages,  # 实际电压值（伏特）
            "hex": hex_seq,
            "score": score,
            "condition_type": cond_type
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
