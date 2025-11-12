"""
VCU CC2 电压序列生成模块（简化版 - 使用未训练模型）
用于在模型训练失败时生成测试序列
"""
import os
import sys
import numpy as np
import tensorflow as tf
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.vcu_data_process import VcuDataProcessor
from nn.scale_model import ScaleModel
from configs.config_vcu import *
from datetime import datetime
import json


def voltage_to_hex(voltage: float) -> str:
    """将电压值转换为十六进制字符串"""
    if np.isnan(voltage) or np.isinf(voltage):
        voltage = 6.0
    voltage_mv = int(voltage * 1000)
    voltage_mv = max(0, min(65535, voltage_mv))
    hex_str = format(voltage_mv, '04X')
    return hex_str


def sequence_to_hex(voltage_sequence: np.ndarray, processor: VcuDataProcessor) -> str:
    """将电压序列转换为十六进制字符串"""
    hex_values = []
    for norm_voltage in voltage_sequence:
        voltage = processor.denormalize_voltage(float(norm_voltage))
        hex_val = voltage_to_hex(voltage)
        hex_values.append(hex_val)
    return ' '.join(hex_values)


def generate_sequences(scale_model, processor, n_sequences=50):
    """生成多种类型的序列"""
    all_sequences = []
    
    # 生成不同类型的序列
    conditions = [
        ([0.8, 0.7, 0.8], 'upper_boundary_peak', '上边界峰值型'),
        ([1.0, 0.25, 1.0], 'lower_voltage_valley', '低电压谷底转折型'),
        ([0.6, 0.5, 0.6], 'strong_oscillation', '强振荡型'),
        ([0.5, 0.4, 0.5], 'boundary_oscillation', '边界震荡型'),
        ([1.0, 0.15, 1.0], 'abnormal_focused', '异常聚焦型'),
    ]
    
    for cond, seq_type, desc in conditions:
        print(f"生成 {desc}...")
        for i in range(n_sequences // len(conditions)):
            condition = tf.constant([cond], dtype=tf.float32)
            try:
                generated = scale_model.model.generate_wc(condition)
                generated = generated.numpy()[0]
                
                if np.any(np.isnan(generated)) or np.any(np.isinf(generated)):
                    continue
                
                hex_sequence = sequence_to_hex(generated, processor)
                voltages = [processor.denormalize_voltage(float(v)) for v in generated]
                
                all_sequences.append({
                    'type': seq_type,
                    'sequence': hex_sequence,
                    'voltage_sequence': generated.tolist(),
                    'voltages': voltages,
                })
            except Exception as e:
                continue
    
    return all_sequences


def save_sequences(sequences, output_dir):
    """保存生成的序列"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    
    json_path = os.path.join(output_dir, f'generated_sequences_{timestamp}.json')
    txt_path = os.path.join(output_dir, f'generated_sequences_{timestamp}.txt')
    
    # 保存 JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, ensure_ascii=False, indent=2)
    
    # 保存 TXT（十六进制格式）
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, seq in enumerate(sequences, 1):
            f.write(f"序列 {i} - {seq['type']}\n")
            f.write(f"电压值: {[f'{v:.2f}V' for v in seq['voltages']]}\n")
            f.write(f"十六进制: {seq['sequence']}\n\n")
    
    print(f"\n生成的序列已保存到:")
    print(f"  JSON: {json_path}")
    print(f"  TXT: {txt_path}")
    print(f"  共生成 {len(sequences)} 个序列")
    
    # 在终端显示前10个
    print(f"\n前10个序列（十六进制）:")
    for i, seq in enumerate(sequences[:10], 1):
        print(f"{i:2d}. {seq['sequence']}")


if __name__ == '__main__':
    print("VCU CC2 电压序列生成器（简化版）")
    print("=" * 50)
    
    # 初始化
    processor = VcuDataProcessor(DB_PATH)
    max_seq_len = 8
    
    # 构建模型（不加载权重，使用随机初始化）
    print("构建模型（使用随机初始化权重）...")
    model = ScaleModel(max_seq_len)
    model.build()
    
    print("开始生成序列...")
    sequences = generate_sequences(model, processor, n_sequences=50)
    
    if len(sequences) > 0:
        save_sequences(sequences, OUTPUT_DIR)
        print("\n生成完成！")
    else:
        print("\n警告: 没有成功生成任何序列")

