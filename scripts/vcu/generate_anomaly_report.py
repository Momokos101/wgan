"""
生成异常点分析的Markdown报告
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.db_loader import VcuDataLoader
from sequence.vcu_data_process import VcuDataProcessor
from configs.config_vcu import DB_PATHS, CONTEXT_BEFORE, CONTEXT_AFTER
from collections import defaultdict, Counter
import numpy as np
import json

def generate_markdown_report():
    """生成Markdown格式的详细报告"""
    
    processor = VcuDataProcessor(DB_PATHS[0])
    anomalies_by_type = defaultdict(list)
    all_anomalies = []
    
    # 收集所有异常点
    for db_idx, db_path in enumerate(DB_PATHS, 1):
        with VcuDataLoader(db_path) as loader:
            sequences = loader.load_sequences_by_round()
            
            for seq_idx, seq in enumerate(sequences):
                anomaly_indices = [i for i, d in enumerate(seq) if d['is_abnormal']]
                
                for anomaly_idx in anomaly_indices:
                    anomaly_data = seq[anomaly_idx]
                    anomaly_info = anomaly_data['anomaly_info']
                    anomaly_type = anomaly_info.get('anomaly_type', 'unknown')
                    
                    context_seq = processor.extract_context_around_anomaly(
                        seq, anomaly_idx, CONTEXT_BEFORE, CONTEXT_AFTER
                    )
                    
                    if context_seq is None:
                        continue
                    
                    voltages = [d['cc2_voltage'] for d in context_seq]
                    voltages_array = np.array(voltages)
                    
                    anomaly_detail = {
                        'db_name': os.path.basename(db_path),
                        'run_id': anomaly_data['run_id'],
                        'round_id': anomaly_data['round_id'],
                        'anomaly_type': anomaly_type,
                        'anomaly_voltage': float(anomaly_data['cc2_voltage']),
                        'output_fields': anomaly_data['output_fields'],
                        'anomaly_info': anomaly_info,
                        'context_voltage_sequence': voltages,
                        'context_stats': {
                            'mean': float(np.mean(voltages_array)),
                            'std': float(np.std(voltages_array)),
                            'min': float(np.min(voltages_array)),
                            'max': float(np.max(voltages_array)),
                            'wake_count': sum(1 for v in voltages if 4.8 <= v <= 7.8),
                            'sleep_count': sum(1 for v in voltages if abs(v - 12.0) < 0.1),
                            'high_count': sum(1 for v in voltages if v >= 7.6),
                            'low_count': sum(1 for v in voltages if v <= 5.5)
                        }
                    }
                    
                    anomalies_by_type[anomaly_type].append(anomaly_detail)
                    all_anomalies.append(anomaly_detail)
    
    # 生成Markdown报告
    report_lines = []
    report_lines.append("# 详细异常点分析报告\n")
    report_lines.append(f"**生成时间**: {os.popen('date').read().strip()}\n")
    report_lines.append(f"**配置参数**: CONTEXT_BEFORE={CONTEXT_BEFORE}, CONTEXT_AFTER={CONTEXT_AFTER}\n")
    report_lines.append(f"**总异常点数**: {len(all_anomalies)}\n\n")
    
    # 第一部分：所有异常点列表
    report_lines.append("## 第一部分：所有异常点详细列表\n\n")
    
    for idx, anomaly in enumerate(all_anomalies, 1):
        report_lines.append(f"### 异常点 #{idx}\n\n")
        report_lines.append(f"- **数据库**: {anomaly['db_name']}\n")
        report_lines.append(f"- **Run ID**: {anomaly['run_id']}, **Round ID**: {anomaly['round_id']}\n")
        report_lines.append(f"- **异常类型**: `{anomaly['anomaly_type']}`\n")
        report_lines.append(f"- **异常点电压**: {anomaly['anomaly_voltage']:.2f}V\n\n")
        
        report_lines.append("**输出字段**:\n")
        for key, value in anomaly['output_fields'].items():
            report_lines.append(f"- {key}: {value}\n")
        report_lines.append("\n")
        
        if anomaly['anomaly_info'].get('ready_flag_mismatch'):
            report_lines.append("**READY标志位不匹配详情**:\n")
            for msg in anomaly['anomaly_info']['ready_flag_mismatch']:
                report_lines.append(f"- {msg}\n")
            report_lines.append("\n")
        
        if anomaly['anomaly_info'].get('state_follow_mismatch'):
            report_lines.append(f"**状态跟随不匹配**: {len(anomaly['anomaly_info']['state_follow_mismatch'])} 个问题\n\n")
        
        report_lines.append("**上下文电压序列**:\n")
        voltage_str = ' → '.join([f'{v:.2f}V' for v in anomaly['context_voltage_sequence']])
        report_lines.append(f"`{voltage_str}`\n\n")
        
        stats = anomaly['context_stats']
        report_lines.append("**上下文电压统计**:\n")
        report_lines.append(f"- 均值: {stats['mean']:.2f}V\n")
        report_lines.append(f"- 标准差: {stats['std']:.2f}V\n")
        report_lines.append(f"- 范围: {stats['min']:.2f}V - {stats['max']:.2f}V\n")
        report_lines.append(f"- 唤醒电压数: {stats['wake_count']}/{len(anomaly['context_voltage_sequence'])}\n")
        report_lines.append(f"- 休眠电压数: {stats['sleep_count']}/{len(anomaly['context_voltage_sequence'])}\n")
        report_lines.append(f"- 高电压数(>=7.6V): {stats['high_count']}\n")
        report_lines.append(f"- 低电压数(<=5.5V): {stats['low_count']}\n\n")
        report_lines.append("---\n\n")
    
    # 第二部分：按类型分析
    report_lines.append("## 第二部分：按异常类型分析共同规律\n\n")
    
    for anomaly_type, anomalies in sorted(anomalies_by_type.items()):
        report_lines.append(f"### {anomaly_type} (共 {len(anomalies)} 个异常点)\n\n")
        
        # 异常点电压统计
        anomaly_voltages = [a['anomaly_voltage'] for a in anomalies]
        report_lines.append("#### 异常点电压统计\n\n")
        report_lines.append(f"- 均值: {np.mean(anomaly_voltages):.2f}V\n")
        report_lines.append(f"- 标准差: {np.std(anomaly_voltages):.2f}V\n")
        report_lines.append(f"- 范围: {np.min(anomaly_voltages):.2f}V - {np.max(anomaly_voltages):.2f}V\n")
        
        wake_count = sum(1 for v in anomaly_voltages if 4.8 <= v <= 7.8)
        sleep_count = sum(1 for v in anomaly_voltages if abs(v - 12.0) < 0.1)
        report_lines.append(f"- 唤醒电压异常: {wake_count} ({wake_count/len(anomaly_voltages)*100:.1f}%)\n")
        report_lines.append(f"- 休眠电压异常: {sleep_count} ({sleep_count/len(anomaly_voltages)*100:.1f}%)\n\n")
        
        # 上下文电压特征
        all_context_voltages = []
        for a in anomalies:
            all_context_voltages.extend(a['context_voltage_sequence'])
        
        report_lines.append("#### 上下文电压序列共同特征\n\n")
        if all_context_voltages:
            report_lines.append(f"- 所有上下文电压均值: {np.mean(all_context_voltages):.2f}V\n")
            report_lines.append(f"- 所有上下文电压标准差: {np.std(all_context_voltages):.2f}V\n")
            report_lines.append(f"- 所有上下文电压范围: {np.min(all_context_voltages):.2f}V - {np.max(all_context_voltages):.2f}V\n\n")
        
        # 平均序列特征
        avg_mean = np.mean([a['context_stats']['mean'] for a in anomalies])
        avg_std = np.mean([a['context_stats']['std'] for a in anomalies])
        avg_wake_ratio = np.mean([a['context_stats']['wake_count']/len(a['context_voltage_sequence']) for a in anomalies])
        avg_high_ratio = np.mean([a['context_stats']['high_count']/len(a['context_voltage_sequence']) for a in anomalies])
        avg_low_ratio = np.mean([a['context_stats']['low_count']/len(a['context_voltage_sequence']) for a in anomalies])
        
        report_lines.append("#### 平均序列特征\n\n")
        report_lines.append(f"- 平均电压均值: {avg_mean:.2f}V\n")
        report_lines.append(f"- 平均电压标准差: {avg_std:.2f}V\n")
        report_lines.append(f"- 平均唤醒电压比例: {avg_wake_ratio*100:.1f}%\n")
        report_lines.append(f"- 平均高电压比例(>=7.6V): {avg_high_ratio*100:.1f}%\n")
        report_lines.append(f"- 平均低电压比例(<=5.5V): {avg_low_ratio*100:.1f}%\n\n")
        
        # 共同规律
        report_lines.append("#### 共同规律分析\n\n")
        
        if sleep_count > wake_count:
            report_lines.append(f"✅ **此类异常多发生在休眠电压(12V)时** ({sleep_count}/{len(anomaly_voltages)})\n\n")
        
        if avg_high_ratio > 0.5:
            report_lines.append(f"✅ **上下文序列中高电压(>=7.6V)占比较高** ({avg_high_ratio*100:.1f}%)\n\n")
        elif avg_low_ratio > 0.5:
            report_lines.append(f"✅ **上下文序列中低电压(<=5.5V)占比较高** ({avg_low_ratio*100:.1f}%)\n\n")
        
        if avg_std > 2.0:
            report_lines.append(f"✅ **上下文电压波动较大** (平均标准差: {avg_std:.2f}V)\n\n")
        elif avg_std < 1.0:
            report_lines.append(f"✅ **上下文电压相对稳定** (平均标准差: {avg_std:.2f}V)\n\n")
        
        # 特定异常类型的特殊规律
        if 'ready_flag_mismatch' in anomaly_type:
            report_lines.append("#### 动力防盗READY标志位与整车状态不匹配的特殊规律\n\n")
            ready_anomalies = [a for a in anomalies if 'ready_flag_mismatch' in a['anomaly_type']]
            if ready_anomalies:
                vehicle_statuses = [a['output_fields'].get('整车状态') for a in ready_anomalies if a['output_fields'].get('整车状态') is not None]
                ready_flags = [a['output_fields'].get('动力防盗允许READY标志位') for a in ready_anomalies if a['output_fields'].get('动力防盗允许READY标志位') is not None]
                
                if vehicle_statuses:
                    report_lines.append(f"- 整车状态范围: {min(vehicle_statuses)} - {max(vehicle_statuses)}\n")
                    report_lines.append(f"- 整车状态均值: {np.mean(vehicle_statuses):.1f}\n")
                    low_count = sum(1 for s in vehicle_statuses if s <= 35)
                    high_count = sum(1 for s in vehicle_statuses if s >= 165)
                    report_lines.append(f"- 接近极小值(<=35): {low_count} 个\n")
                    report_lines.append(f"- 接近极大值(>=165): {high_count} 个\n\n")
                
                if ready_flags:
                    flag_counter = Counter(ready_flags)
                    report_lines.append(f"- READY标志位分布: {dict(flag_counter)}\n\n")
                
                # 分析这类异常的上下文电压特征
                ready_context_voltages = []
                for a in ready_anomalies:
                    ready_context_voltages.extend(a['context_voltage_sequence'])
                
                if ready_context_voltages:
                    report_lines.append("**上下文电压序列特征**:\n")
                    report_lines.append(f"- 上下文电压均值: {np.mean(ready_context_voltages):.2f}V\n")
                    report_lines.append(f"- 上下文电压标准差: {np.std(ready_context_voltages):.2f}V\n")
                    report_lines.append(f"- 上下文电压范围: {np.min(ready_context_voltages):.2f}V - {np.max(ready_context_voltages):.2f}V\n\n")
                    
                    # 检查是否有规律
                    wake_ratio = sum(1 for v in ready_context_voltages if 4.8 <= v <= 7.8) / len(ready_context_voltages)
                    sleep_ratio = sum(1 for v in ready_context_voltages if abs(v - 12.0) < 0.1) / len(ready_context_voltages)
                    report_lines.append(f"- 唤醒电压比例: {wake_ratio*100:.1f}%\n")
                    report_lines.append(f"- 休眠电压比例: {sleep_ratio*100:.1f}%\n\n")
        
        if 'state_follow_mismatch' in anomaly_type:
            report_lines.append("#### 状态跟随不匹配的特殊规律\n\n")
            state_anomalies = [a for a in anomalies if 'state_follow_mismatch' in a['anomaly_type']]
            if state_anomalies:
                vehicle_statuses = [a['output_fields'].get('整车状态') for a in state_anomalies if a['output_fields'].get('整车状态') is not None]
                if vehicle_statuses:
                    report_lines.append(f"- 整车状态范围: {min(vehicle_statuses)} - {max(vehicle_statuses)}\n")
                    report_lines.append(f"- 整车状态均值: {np.mean(vehicle_statuses):.1f}\n")
                    extreme_count = sum(1 for s in vehicle_statuses if s <= 35 or s >= 165)
                    report_lines.append(f"- 极端状态(<=35或>=165): {extreme_count} 个 ({extreme_count/len(vehicle_statuses)*100:.1f}%)\n\n")
        
        # 典型示例
        report_lines.append("#### 典型示例\n\n")
        for i, anomaly in enumerate(anomalies[:3], 1):
            report_lines.append(f"**示例 {i}**:\n")
            report_lines.append(f"- Run ID: {anomaly['run_id']}, 异常点电压: {anomaly['anomaly_voltage']:.2f}V\n")
            report_lines.append(f"- 整车状态: {anomaly['output_fields'].get('整车状态', 'N/A')}\n")
            report_lines.append(f"- READY标志位: {anomaly['output_fields'].get('动力防盗允许READY标志位', 'N/A')}\n")
            voltage_str = ' → '.join([f'{v:.2f}V' for v in anomaly['context_voltage_sequence']])
            report_lines.append(f"- 上下文序列: `{voltage_str}`\n\n")
        
        report_lines.append("---\n\n")
    
    # 保存报告
    report_path = 'docs/详细异常点分析报告.md'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"✅ Markdown报告已生成: {report_path}")
    print(f"   共分析了 {len(all_anomalies)} 个异常点")
    print(f"   异常类型: {list(anomalies_by_type.keys())}")

if __name__ == '__main__':
    generate_markdown_report()

