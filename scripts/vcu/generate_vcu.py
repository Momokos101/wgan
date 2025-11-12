"""
VCU CC2 电压序列生成模块
生成十六进制格式的 CC2 电压序列
"""
import os
import sys
import numpy as np
import tensorflow as tf
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.vcu_data_process import load_vcu_data, VcuDataProcessor
from nn.wd_trainer import WassersteinTrainer
from nn.scale_model import ScaleModel
from configs.config_vcu import *
from nn.optimizer import Optimizer
from nn.load_model import load_model
from datetime import datetime
import json


def voltage_to_hex(voltage: float) -> str:
    """
    将电压值转换为十六进制字符串
    
    Args:
        voltage: 电压值（V）
        
    Returns:
        十六进制字符串（2字节，大端序）
    """
    # 检查 NaN 或无效值
    if np.isnan(voltage) or np.isinf(voltage):
        # 返回默认值（例如 6.0V）
        voltage = 6.0
    
    # 将电压值转换为整数（以毫伏为单位）
    voltage_mv = int(voltage * 1000)
    # 限制在 0-65535 范围内
    voltage_mv = max(0, min(65535, voltage_mv))
    # 转换为十六进制（2字节，大端序）
    hex_str = format(voltage_mv, '04X')
    return hex_str


def sequence_to_hex(voltage_sequence: np.ndarray, processor: VcuDataProcessor) -> str:
    """
    将电压序列转换为十六进制字符串
    
    Args:
        voltage_sequence: 归一化的电压序列
        processor: 数据处理器（用于反归一化）
        
    Returns:
        十六进制字符串，每个电压值用2字节表示
    """
    hex_values = []
    for norm_voltage in voltage_sequence:
        # 反归一化
        voltage = processor.denormalize_voltage(float(norm_voltage))
        # 转换为十六进制
        hex_val = voltage_to_hex(voltage)
        hex_values.append(hex_val)
    
    return ' '.join(hex_values)


def build_condition_vector(
    abnormal_flag: float = 0.0,
    vehicle_status_norm: float = 0.5,
    ready_flag: float = 0.0,
    anomaly_type: int = 0,
    peak_ratio: float = 0.0,
    max_value_ratio: float = 0.0,
    boundary_ratio: float = 0.0,
    oscillation_strength: float = 0.0,
    voltage_range: float = 0.0
) -> np.ndarray:
    """
    构建9维条件向量
    
    Args:
        abnormal_flag: 异常标志 (0-1)
        vehicle_status_norm: 整车状态归一化 (0-1)
        ready_flag: READY标志位 (0-1)
        anomaly_type: 异常类型编码 (0-4: normal, state_follow_mismatch, error, stuck, ready_flag_mismatch)
        peak_ratio: 震荡顶峰比例 (0-1)
        max_value_ratio: 连续极大值比例 (0-1)
        boundary_ratio: 边界值比例 (0-1)
        oscillation_strength: 震荡强度 (0-1)
        voltage_range: 电压范围 (0-1)
        
    Returns:
        9维条件向量
    """
    # 基础条件 (3维)
    base_condition = [abnormal_flag, vehicle_status_norm, ready_flag]
    
    # 异常类型编码 (1维，归一化到[0,1])
    anomaly_type_norm = anomaly_type / 4.0
    
    # 电压特征 (5维)
    voltage_features = [peak_ratio, max_value_ratio, boundary_ratio, oscillation_strength, voltage_range]
    
    # 组合成9维向量
    condition = np.array(base_condition + [anomaly_type_norm] + voltage_features, dtype=np.float32)
    return condition


def calculate_sequence_score(voltage_seq: np.ndarray, processor: VcuDataProcessor) -> dict:
    """
    计算序列评分
    根据数据分析总结的规律进行评分
    
    Args:
        voltage_seq: 归一化的电压序列
        processor: 数据处理器
        
    Returns:
        评分字典，包含各项得分和总分
    """
    # 反归一化电压序列
    voltages = np.array([processor.denormalize_voltage(float(v)) for v in voltage_seq])
    
    # 过滤掉休眠电压（12V）和无效值
    valid_voltages = voltages[(voltages >= 4.8) & (voltages <= 7.8)]
    if len(valid_voltages) == 0:
        return {'total_score': 0.0, 'high_risk_zone': 0.0, 'extreme_points': 0.0, 
                'oscillation': 0.0, 'turn_points': 0.0}
    
    score = 0.0
    details = {}
    
    # 1. 高风险区间得分
    # 上边界段 7.6-7.8V
    upper_boundary_count = np.sum((valid_voltages >= 7.6) & (valid_voltages <= 7.8))
    upper_score = (upper_boundary_count / len(valid_voltages)) * 30.0  # 最高30分
    
    # 低/中电压段 5.7-7.1V
    lower_boundary_count = np.sum((valid_voltages >= 5.7) & (valid_voltages <= 7.1))
    lower_score = (lower_boundary_count / len(valid_voltages)) * 30.0  # 最高30分
    
    high_risk_score = upper_score + lower_score
    score += high_risk_score
    details['high_risk_zone'] = high_risk_score
    
    # 2. 极值点得分（峰/谷）
    if len(valid_voltages) >= 3:
        # 计算一阶差分
        diff = np.diff(valid_voltages)
        # 计算二阶差分（转折点）
        diff2 = np.diff(diff)
        
        # 局部峰值（前增后减）
        peaks = np.sum((diff[:-1] > 0) & (diff[1:] < 0))
        # 局部谷值（前减后增）
        valleys = np.sum((diff[:-1] < 0) & (diff[1:] > 0))
        
        extreme_points = peaks + valleys
        extreme_score = min(extreme_points / len(valid_voltages) * 20.0, 20.0)  # 最高20分
        score += extreme_score
        details['extreme_points'] = extreme_score
    else:
        details['extreme_points'] = 0.0
    
    # 3. 振荡强度得分
    if len(valid_voltages) >= 2:
        # 计算振荡比（一阶差分符号交替频率）
        diff_signs = np.sign(np.diff(valid_voltages))
        sign_changes = np.sum(diff_signs[:-1] != diff_signs[1:])
        oscillation_ratio = sign_changes / max(len(diff_signs) - 1, 1)
        
        # 振荡比 ≥ 0.6 得高分
        if oscillation_ratio >= 0.6:
            oscillation_score = 20.0
        elif oscillation_ratio >= 0.4:
            oscillation_score = 10.0
        else:
            oscillation_score = 5.0
        
        score += oscillation_score
        details['oscillation'] = oscillation_score
        details['oscillation_ratio'] = oscillation_ratio
    else:
        details['oscillation'] = 0.0
        details['oscillation_ratio'] = 0.0
    
    # 4. 转折点得分（单调段后的转折）
    if len(valid_voltages) >= 3:
        # 单调段检测
        monotonic_segments = 0
        for i in range(1, len(valid_voltages) - 1):
            # 检查是否在单调段后转折
            if (valid_voltages[i-1] < valid_voltages[i] > valid_voltages[i+1]) or \
               (valid_voltages[i-1] > valid_voltages[i] < valid_voltages[i+1]):
                monotonic_segments += 1
        
        turn_score = min(monotonic_segments / len(valid_voltages) * 20.0, 20.0)  # 最高20分
        score += turn_score
        details['turn_points'] = turn_score
    else:
        details['turn_points'] = 0.0
    
    # 5. 上下文宽度得分（波动范围）
    voltage_range = valid_voltages.max() - valid_voltages.min()
    if voltage_range >= 2.5:  # 波动 ≥ 2.5V
        range_score = 10.0
    elif voltage_range >= 1.5:
        range_score = 5.0
    else:
        range_score = 0.0
    
    score += range_score
    details['voltage_range'] = voltage_range
    details['range_score'] = range_score
    
    details['total_score'] = score
    return details


def generate_upper_boundary_peak(scale_model, processor, n_sequences=10):
    """
    生成上边界峰值型序列（7.6-7.8V）
    针对充电枪连接和PDCU快充唤醒异常
    
    Args:
        scale_model: 训练好的模型
        processor: 数据处理器
        n_sequences: 生成序列数量
        
    Returns:
        生成的序列列表（带评分）
    """
    generated_sequences = []
    
    for i in range(n_sequences):
        # 上边界峰值条件：state_follow_mismatch异常，高震荡顶峰，连续极大值
        condition = build_condition_vector(
            abnormal_flag=1.0,
            vehicle_status_norm=0.85,  # 170/200
            ready_flag=1.0,
            anomaly_type=1,  # state_follow_mismatch
            peak_ratio=0.8,  # 高震荡顶峰
            max_value_ratio=0.9,  # 连续极大值
            boundary_ratio=0.7,
            oscillation_strength=0.6,
            voltage_range=0.8
        )
        condition = tf.constant(condition[np.newaxis, :], dtype=tf.float32)
        
        try:
            generated = scale_model.model.generate_wc(condition)
            generated = generated.numpy()[0]
            
            # 检查生成的序列是否包含 NaN 或无效值
            if np.any(np.isnan(generated)) or np.any(np.isinf(generated)):
                raise ValueError("生成的序列包含 NaN 或 Inf 值")
            
            # 计算评分
            score_details = calculate_sequence_score(generated, processor)
            
            # 转换为十六进制
            hex_sequence = sequence_to_hex(generated, processor)
            
            # 反归一化电压用于显示
            voltages = [processor.denormalize_voltage(float(v)) for v in generated]
            
            generated_sequences.append({
                'type': 'upper_boundary_peak',
                'sequence': hex_sequence,
                'voltage_sequence': generated.tolist(),
                'voltages': voltages,
                'score': score_details
            })
        except Exception as e:
            print(f"生成序列 {i} 时出错: {e}")
            continue
    
    return generated_sequences


def generate_lower_voltage_valley(scale_model, processor, n_sequences=10):
    """
    生成低电压谷底转折型序列（5.7-7.1V）
    针对READY异常，特别是低谷/下行后触发
    
    Args:
        scale_model: 训练好的模型
        processor: 数据处理器
        n_sequences: 生成序列数量
        
    Returns:
        生成的序列列表（带评分）
    """
    generated_sequences = []
    
    for i in range(n_sequences):
        # 低电压谷底条件：state_follow_mismatch异常，低电压，低谷特征
        condition = build_condition_vector(
            abnormal_flag=1.0,
            vehicle_status_norm=0.15,  # 30/200
            ready_flag=1.0,
            anomaly_type=1,  # state_follow_mismatch
            peak_ratio=0.2,  # 低震荡顶峰
            max_value_ratio=0.1,
            boundary_ratio=0.5,  # 边界值
            oscillation_strength=0.7,  # 高震荡强度
            voltage_range=0.6
        )
        condition = tf.constant(condition[np.newaxis, :], dtype=tf.float32)
        
        try:
            generated = scale_model.model.generate_wc(condition)
            generated = generated.numpy()[0]
            
            # 检查生成的序列是否包含 NaN 或无效值
            if np.any(np.isnan(generated)) or np.any(np.isinf(generated)):
                raise ValueError("生成的序列包含 NaN 或 Inf 值")
            
            # 计算评分
            score_details = calculate_sequence_score(generated, processor)
            
            # 对READY异常类型额外加分（低谷/下行后触发）
            voltages = np.array([processor.denormalize_voltage(float(v)) for v in generated])
            valid_voltages = voltages[(voltages >= 4.8) & (voltages <= 7.8)]
            
            if len(valid_voltages) >= 3:
                # 检查是否有下降→谷→反弹模式
                for j in range(1, len(valid_voltages) - 1):
                    if valid_voltages[j-1] > valid_voltages[j] < valid_voltages[j+1]:
                        # 找到谷底，检查是否在5.7-7.1V范围内
                        if 5.7 <= valid_voltages[j] <= 7.1:
                            score_details['total_score'] += 10.0  # 额外加分
                            score_details['valley_bonus'] = 10.0
                            break
            
            # 转换为十六进制
            hex_sequence = sequence_to_hex(generated, processor)
            
            generated_sequences.append({
                'type': 'lower_voltage_valley',
                'sequence': hex_sequence,
                'voltage_sequence': generated.tolist(),
                'voltages': [processor.denormalize_voltage(float(v)) for v in generated],
                'score': score_details
            })
        except Exception as e:
            print(f"生成序列 {i} 时出错: {e}")
            continue
    
    return generated_sequences


def generate_strong_oscillation(scale_model, processor, n_sequences=10):
    """
    生成强振荡型序列
    振荡比≥0.6，周期4-8点，上下文波动≥2.5V
    
    Args:
        scale_model: 训练好的模型
        processor: 数据处理器
        n_sequences: 生成序列数量
        
    Returns:
        生成的序列列表（带评分）
    """
    generated_sequences = []
    
    for i in range(n_sequences):
        # 强振荡条件：高震荡强度，高震荡顶峰
        condition = build_condition_vector(
            abnormal_flag=0.6,
            vehicle_status_norm=0.5,
            ready_flag=0.6,
            anomaly_type=0,  # normal或混合
            peak_ratio=0.6,
            max_value_ratio=0.3,
            boundary_ratio=0.4,
            oscillation_strength=0.9,  # 高震荡强度
            voltage_range=0.8
        )
        condition = tf.constant(condition[np.newaxis, :], dtype=tf.float32)
        
        try:
            generated = scale_model.model.generate_wc(condition)
            generated = generated.numpy()[0]
            
            # 检查生成的序列是否包含 NaN 或无效值
            if np.any(np.isnan(generated)) or np.any(np.isinf(generated)):
                raise ValueError("生成的序列包含 NaN 或 Inf 值")
            
            # 计算评分
            score_details = calculate_sequence_score(generated, processor)
            
            # 转换为十六进制
            hex_sequence = sequence_to_hex(generated, processor)
            
            generated_sequences.append({
                'type': 'strong_oscillation',
                'sequence': hex_sequence,
                'voltage_sequence': generated.tolist(),
                'voltages': [processor.denormalize_voltage(float(v)) for v in generated],
                'score': score_details
            })
        except Exception as e:
            print(f"生成序列 {i} 时出错: {e}")
            continue
    
    return generated_sequences


def generate_boundary_oscillation(scale_model, processor, n_sequences=10):
    """
    生成边界震荡型 CC2 电压序列
    在边界值（4.8V 和 7.8V）之间快速震荡
    
    Args:
        scale_model: 训练好的模型
        processor: 数据处理器
        n_sequences: 生成序列数量
        
    Returns:
        生成的序列列表（十六进制格式）
    """
    generated_sequences = []
    
    for i in range(n_sequences):
        # 边界震荡条件：高边界值比例，高震荡强度
        condition = build_condition_vector(
            abnormal_flag=0.5,
            vehicle_status_norm=0.4,
            ready_flag=0.5,
            anomaly_type=0,
            peak_ratio=0.4,
            max_value_ratio=0.2,
            boundary_ratio=0.9,  # 高边界值比例
            oscillation_strength=0.8,
            voltage_range=0.9
        )
        condition = tf.constant(condition[np.newaxis, :], dtype=tf.float32)
        
        # 生成序列
        try:
            generated = scale_model.model.generate_wc(condition)
            generated = generated.numpy()[0]  # 取第一个批次
            
            # 计算评分
            score_details = calculate_sequence_score(generated, processor)
            
            # 转换为十六进制
            hex_sequence = sequence_to_hex(generated, processor)
            generated_sequences.append({
                'type': 'boundary_oscillation',
                'sequence': hex_sequence,
                'voltage_sequence': generated.tolist(),
                'voltages': [processor.denormalize_voltage(float(v)) for v in generated],
                'score': score_details
            })
        except Exception as e:
            print(f"生成序列 {i} 时出错: {e}")
            continue
    
    return generated_sequences


def generate_abnormal_focused(scale_model, processor, n_sequences=10):
    """
    生成专注于异常的模式
    使用异常条件来引导生成
    
    Args:
        scale_model: 训练好的模型
        processor: 数据处理器
        n_sequences: 生成序列数量
        
    Returns:
        生成的序列列表（十六进制格式）
    """
    generated_sequences = []
    
    for i in range(n_sequences):
        # 异常聚焦条件：state_follow_mismatch，READY=1但整车状态很低，高震荡顶峰
        condition = build_condition_vector(
            abnormal_flag=1.0,
            vehicle_status_norm=0.15,  # 30/200
            ready_flag=1.0,
            anomaly_type=1,  # state_follow_mismatch
            peak_ratio=0.7,  # 高震荡顶峰（异常多出现在震荡顶峰）
            max_value_ratio=0.8,  # 连续极大值（异常多出现在7.8V附近）
            boundary_ratio=0.6,
            oscillation_strength=0.7,
            voltage_range=0.7
        )
        condition = tf.constant(condition[np.newaxis, :], dtype=tf.float32)
        
        try:
            generated = scale_model.model.generate_wc(condition)
            generated = generated.numpy()[0]
            
            # 检查生成的序列是否包含 NaN 或无效值
            if np.any(np.isnan(generated)) or np.any(np.isinf(generated)):
                raise ValueError("生成的序列包含 NaN 或 Inf 值")
            
            # 计算评分
            score_details = calculate_sequence_score(generated, processor)
            
            hex_sequence = sequence_to_hex(generated, processor)
            generated_sequences.append({
                'type': 'abnormal_focused',
                'sequence': hex_sequence,
                'voltage_sequence': generated.tolist(),
                'voltages': [processor.denormalize_voltage(float(v)) for v in generated],
                'score': score_details
            })
        except Exception as e:
            print(f"生成序列 {i} 时出错: {e}")
            continue
    
    return generated_sequences


def generate_by_anomaly_type(scale_model, processor, anomaly_type: int, n_sequences=10):
    """
    根据异常类型生成对应的CC2电压测试序列
    
    异常类型与电压特征映射：
    - state_follow_mismatch (1): 多出现在震荡顶峰或连续极大值(7.8V)附近
    - error (2): 根据expected_error_output的特征
    - stuck (3): 根据expected_stuck_output的特征
    
    Args:
        scale_model: 训练好的模型
        processor: 数据处理器
        anomaly_type: 异常类型 (1=state_follow_mismatch, 2=error, 3=stuck)
        n_sequences: 生成序列数量
        
    Returns:
        生成的序列列表（十六进制格式）
    """
    generated_sequences = []
    
    for i in range(n_sequences):
        if anomaly_type == 1:  # state_follow_mismatch: 震荡顶峰或连续极大值
            condition = build_condition_vector(
                abnormal_flag=1.0,
                vehicle_status_norm=0.85,  # 170/200 (极大值)
                ready_flag=1.0,
                anomaly_type=1,
                peak_ratio=0.8,  # 高震荡顶峰（异常多出现在震荡顶峰）
                max_value_ratio=0.9,  # 连续极大值（异常多出现在7.8V附近）
                boundary_ratio=0.7,
                oscillation_strength=0.7,
                voltage_range=0.8
            )
        elif anomaly_type == 2:  # error: 根据error特征
            condition = build_condition_vector(
                abnormal_flag=1.0,
                vehicle_status_norm=0.5,
                ready_flag=0.5,
                anomaly_type=2,
                peak_ratio=0.5,
                max_value_ratio=0.3,
                boundary_ratio=0.4,
                oscillation_strength=0.6,
                voltage_range=0.6
            )
        elif anomaly_type == 3:  # stuck: 根据stuck特征
            condition = build_condition_vector(
                abnormal_flag=1.0,
                vehicle_status_norm=0.5,
                ready_flag=0.5,
                anomaly_type=3,
                peak_ratio=0.3,
                max_value_ratio=0.2,
                boundary_ratio=0.3,
                oscillation_strength=0.3,  # 低震荡（卡死状态）
                voltage_range=0.2
            )
        else:
            # 默认：正常或混合
            condition = build_condition_vector(
                abnormal_flag=0.0,
                vehicle_status_norm=0.5,
                ready_flag=0.5,
                anomaly_type=0,
                peak_ratio=0.3,
                max_value_ratio=0.2,
                boundary_ratio=0.3,
                oscillation_strength=0.4,
                voltage_range=0.5
            )
        
        condition = tf.constant(condition[np.newaxis, :], dtype=tf.float32)
        
        try:
            generated = scale_model.model.generate_wc(condition)
            generated = generated.numpy()[0]
            
            if np.any(np.isnan(generated)) or np.any(np.isinf(generated)):
                raise ValueError("生成的序列包含 NaN 或 Inf 值")
            
            score_details = calculate_sequence_score(generated, processor)
            hex_sequence = sequence_to_hex(generated, processor)
            
            anomaly_type_names = {0: 'normal', 1: 'state_follow_mismatch', 2: 'error', 3: 'stuck'}
            generated_sequences.append({
                'type': f'anomaly_type_{anomaly_type_names.get(anomaly_type, "unknown")}',
                'sequence': hex_sequence,
                'voltage_sequence': generated.tolist(),
                'voltages': [processor.denormalize_voltage(float(v)) for v in generated],
                'score': score_details,
                'anomaly_type': anomaly_type
            })
        except Exception as e:
            print(f"生成序列 {i} 时出错: {e}")
            continue
    
    return generated_sequences


def generate_random_conditions(scale_model, processor, n_sequences=10):
    """
    使用随机条件生成序列
    
    Args:
        scale_model: 训练好的模型
        processor: 数据处理器
        n_sequences: 生成序列数量
        
    Returns:
        生成的序列列表（十六进制格式）
    """
    generated_sequences = []
    
    for i in range(n_sequences):
        # 随机生成9维条件向量
        condition = build_condition_vector(
            abnormal_flag=np.random.uniform(0, 1),
            vehicle_status_norm=np.random.uniform(0, 1),
            ready_flag=np.random.uniform(0, 1),
            anomaly_type=np.random.randint(0, 5),
            peak_ratio=np.random.uniform(0, 1),
            max_value_ratio=np.random.uniform(0, 1),
            boundary_ratio=np.random.uniform(0, 1),
            oscillation_strength=np.random.uniform(0, 1),
            voltage_range=np.random.uniform(0, 1)
        )
        condition = tf.constant(condition[np.newaxis, :], dtype=tf.float32)
        
        try:
            generated = scale_model.model.generate_wc(condition)
            generated = generated.numpy()[0]
            
            # 检查生成的序列是否包含 NaN 或无效值
            if np.any(np.isnan(generated)) or np.any(np.isinf(generated)):
                raise ValueError("生成的序列包含 NaN 或 Inf 值")
            
            # 计算评分
            score_details = calculate_sequence_score(generated, processor)
            
            hex_sequence = sequence_to_hex(generated, processor)
            generated_sequences.append({
                'type': 'random',
                'sequence': hex_sequence,
                'voltage_sequence': generated.tolist(),
                'voltages': [processor.denormalize_voltage(float(v)) for v in generated],
                'score': score_details
            })
        except Exception as e:
            print(f"生成序列 {i} 时出错: {e}")
            continue
    
    return generated_sequences


def save_generated_sequences(sequences: list, output_dir: str, show_in_terminal: bool = True):
    """
    保存生成的序列到文件，并在终端显示
    
    Args:
        sequences: 生成的序列列表
        output_dir: 输出目录
        show_in_terminal: 是否在终端显示
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 按评分排序
    sequences_sorted = sorted(sequences, key=lambda x: x.get('score', {}).get('total_score', 0.0), reverse=True)
    
    now = datetime.now()
    date_time = now.strftime("%m%d%H%M%S")
    
    # 在终端显示
    if show_in_terminal:
        print("\n" + "=" * 80)
        print("生成的序列（按评分排序）")
        print("=" * 80)
        
        for i, seq in enumerate(sequences_sorted[:20], 1):  # 只显示前20个
            score = seq.get('score', {})
            voltages = seq.get('voltages', [])
            valid_voltages = [v for v in voltages if 4.8 <= v <= 7.8]
            
            print(f"\n序列 {i} - 类型: {seq['type']}")
            print(f"  评分: {score.get('total_score', 0.0):.2f}")
            print(f"    - 高风险区间: {score.get('high_risk_zone', 0.0):.2f}")
            print(f"    - 极值点: {score.get('extreme_points', 0.0):.2f}")
            print(f"    - 振荡强度: {score.get('oscillation', 0.0):.2f} (振荡比: {score.get('oscillation_ratio', 0.0):.3f})")
            print(f"    - 转折点: {score.get('turn_points', 0.0):.2f}")
            print(f"    - 电压范围: {score.get('voltage_range', 0.0):.2f}V")
            
            if valid_voltages:
                print(f"  电压序列 (V): {[f'{v:.2f}' for v in valid_voltages[:10]]}{'...' if len(valid_voltages) > 10 else ''}")
                print(f"  电压范围: {min(valid_voltages):.2f}V - {max(valid_voltages):.2f}V")
            
            print(f"  十六进制: {seq['sequence'][:60]}{'...' if len(seq['sequence']) > 60 else ''}")
    
    # 保存为 JSON
    json_path = os.path.join(output_dir, f'generated_sequences_{date_time}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sequences_sorted, f, indent=2, ensure_ascii=False)
    
    # 保存为纯文本（仅十六进制序列，按评分排序）
    txt_path = os.path.join(output_dir, f'generated_sequences_{date_time}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("# VCU CC2 电压序列（十六进制格式）\n")
        f.write("# 每个序列格式：电压1(2字节) 电压2(2字节) ...\n")
        f.write("# 按评分从高到低排序\n\n")
        for i, seq in enumerate(sequences_sorted, 1):
            score = seq.get('score', {})
            f.write(f"# 序列 {i} - 类型: {seq['type']} - 评分: {score.get('total_score', 0.0):.2f}\n")
            f.write(f"{seq['sequence']}\n\n")
    
    # 保存评分报告
    report_path = os.path.join(output_dir, f'generated_sequences_report_{date_time}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("VCU CC2 电压序列生成报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总序列数: {len(sequences_sorted)}\n\n")
        
        # 按类型统计
        type_stats = {}
        for seq in sequences_sorted:
            seq_type = seq['type']
            if seq_type not in type_stats:
                type_stats[seq_type] = {'count': 0, 'avg_score': 0.0, 'max_score': 0.0}
            type_stats[seq_type]['count'] += 1
            score = seq.get('score', {}).get('total_score', 0.0)
            type_stats[seq_type]['avg_score'] += score
            type_stats[seq_type]['max_score'] = max(type_stats[seq_type]['max_score'], score)
        
        for seq_type, stats in type_stats.items():
            stats['avg_score'] /= stats['count']
        
        f.write("按类型统计:\n")
        for seq_type, stats in sorted(type_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True):
            f.write(f"  {seq_type}:\n")
            f.write(f"    数量: {stats['count']}\n")
            f.write(f"    平均评分: {stats['avg_score']:.2f}\n")
            f.write(f"    最高评分: {stats['max_score']:.2f}\n\n")
        
        # 高分序列详情
        f.write("\n高分序列详情 (Top 10):\n")
        f.write("-" * 80 + "\n")
        for i, seq in enumerate(sequences_sorted[:10], 1):
            score = seq.get('score', {})
            f.write(f"\n序列 {i} - {seq['type']}\n")
            f.write(f"  总评分: {score.get('total_score', 0.0):.2f}\n")
            f.write(f"  高风险区间得分: {score.get('high_risk_zone', 0.0):.2f}\n")
            f.write(f"  极值点得分: {score.get('extreme_points', 0.0):.2f}\n")
            f.write(f"  振荡强度: {score.get('oscillation', 0.0):.2f} (振荡比: {score.get('oscillation_ratio', 0.0):.3f})\n")
            f.write(f"  转折点得分: {score.get('turn_points', 0.0):.2f}\n")
            voltages = seq.get('voltages', [])
            valid_voltages = [v for v in voltages if 4.8 <= v <= 7.8]
            if valid_voltages:
                f.write(f"  电压范围: {min(valid_voltages):.2f}V - {max(valid_voltages):.2f}V\n")
            f.write(f"  十六进制: {seq['sequence']}\n")
    
    print(f"\n生成的序列已保存到:")
    print(f"  JSON: {json_path}")
    print(f"  TXT: {txt_path}")
    print(f"  报告: {report_path}")
    print(f"  共生成 {len(sequences_sorted)} 个序列")
    if len(sequences_sorted) > 0:
        scores = [s.get('score', {}).get('total_score', 0.0) for s in sequences_sorted]
        print(f"  平均评分: {np.mean(scores):.2f}")
        print(f"  最高评分: {max(scores):.2f}")
    else:
        print("  警告: 没有成功生成任何序列")


if __name__ == '__main__':
    print("VCU CC2 电压序列生成器")
    print("=" * 50)
    
    # 加载数据（用于获取处理器）
    train_data, test_data, max_seq_len = load_vcu_data(
        precision=PRECISION,
        db_path=DB_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # 更新最大序列长度
    global MAX_SEQUENCE_LENGTH
    MAX_SEQUENCE_LENGTH = max_seq_len
    print(f"最大序列长度: {MAX_SEQUENCE_LENGTH}")
    
    # 创建数据处理器
    processor = VcuDataProcessor(DB_PATH)
    
    # 构建模型
    scale_model = ScaleModel(max_seq_len)
    scale_model.build()
    
    # 加载训练好的模型
    model_path = MODEL_PATH
    model_type = MODEL_TYPE
    scale_model = load_model(scale_model, model_path, FLAG, model_type)
    
    # 生成不同类型的序列
    print("\n开始生成序列...")
    print("根据异常类型和电压特征，重点生成以下类型：")
    print("  1. 按异常类型生成 - state_follow_mismatch (震荡顶峰/连续极大值)")
    print("  2. 按异常类型生成 - error (error类型特征)")
    print("  3. 按异常类型生成 - stuck (stuck类型特征)")
    print("  4. 上边界峰值型 (7.6-7.8V) - 针对充电枪连接和PDCU快充唤醒异常")
    print("  5. 低电压谷底转折型 (5.7-7.1V) - 针对READY异常")
    print("  6. 强振荡型 - 振荡比≥0.6，周期4-8点")
    print("  7. 边界震荡型 - 上下边界交替")
    print("  8. 异常聚焦型 - 通用异常条件")
    print("  9. 随机条件型 - 探索性生成")
    
    all_sequences = []
    
    # 1. 按异常类型生成 - state_follow_mismatch（震荡顶峰或连续极大值）
    print("\n[1/9] 生成 state_follow_mismatch 类型序列（震荡顶峰/连续极大值）...")
    state_follow_seqs = generate_by_anomaly_type(scale_model, processor, anomaly_type=1, n_sequences=20)
    all_sequences.extend(state_follow_seqs)
    print(f"  已生成 {len(state_follow_seqs)} 个序列")
    
    # 2. 按异常类型生成 - error
    print("\n[2/9] 生成 error 类型序列...")
    error_seqs = generate_by_anomaly_type(scale_model, processor, anomaly_type=2, n_sequences=10)
    all_sequences.extend(error_seqs)
    print(f"  已生成 {len(error_seqs)} 个序列")
    
    # 3. 按异常类型生成 - stuck
    print("\n[3/9] 生成 stuck 类型序列...")
    stuck_seqs = generate_by_anomaly_type(scale_model, processor, anomaly_type=3, n_sequences=10)
    all_sequences.extend(stuck_seqs)
    print(f"  已生成 {len(stuck_seqs)} 个序列")
    
    # 4. 上边界峰值型（针对7.6-7.8V高风险区间）
    print("\n[4/9] 生成上边界峰值型序列 (7.6-7.8V)...")
    upper_peak_seqs = generate_upper_boundary_peak(scale_model, processor, n_sequences=15)
    all_sequences.extend(upper_peak_seqs)
    print(f"  已生成 {len(upper_peak_seqs)} 个序列")
    
    # 5. 低电压谷底转折型（针对5.7-7.1V高风险区间）
    print("\n[5/9] 生成低电压谷底转折型序列 (5.7-7.1V)...")
    lower_valley_seqs = generate_lower_voltage_valley(scale_model, processor, n_sequences=15)
    all_sequences.extend(lower_valley_seqs)
    print(f"  已生成 {len(lower_valley_seqs)} 个序列")
    
    # 6. 强振荡型
    print("\n[6/9] 生成强振荡型序列...")
    oscillation_seqs = generate_strong_oscillation(scale_model, processor, n_sequences=10)
    all_sequences.extend(oscillation_seqs)
    print(f"  已生成 {len(oscillation_seqs)} 个序列")
    
    # 7. 边界震荡型
    print("\n[7/9] 生成边界震荡型序列...")
    boundary_seqs = generate_boundary_oscillation(scale_model, processor, n_sequences=10)
    all_sequences.extend(boundary_seqs)
    print(f"  已生成 {len(boundary_seqs)} 个序列")
    
    # 8. 异常聚焦型
    print("\n[8/9] 生成异常聚焦型序列...")
    abnormal_seqs = generate_abnormal_focused(scale_model, processor, n_sequences=10)
    all_sequences.extend(abnormal_seqs)
    print(f"  已生成 {len(abnormal_seqs)} 个序列")
    
    # 9. 随机条件型
    print("\n[9/9] 生成随机条件型序列...")
    random_seqs = generate_random_conditions(scale_model, processor, n_sequences=10)
    all_sequences.extend(random_seqs)
    print(f"  已生成 {len(random_seqs)} 个序列")
    
    # 保存生成的序列（会在终端显示）
    save_generated_sequences(all_sequences, OUTPUT_DIR, show_in_terminal=True)
    
    print("\n生成完成！")

