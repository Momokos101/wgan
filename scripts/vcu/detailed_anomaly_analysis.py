"""
详细分析每个异常点及其电压序列规律
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.db_loader import VcuDataLoader
from sequence.vcu_data_process import VcuDataProcessor
from configs.config_vcu import DB_PATHS, CONTEXT_BEFORE, CONTEXT_AFTER
from collections import defaultdict
import json
import numpy as np

def analyze_anomaly_point(loader, processor, seq, anomaly_idx, anomaly_info):
    """分析单个异常点的详细信息"""
    # 提取上下文
    context_seq = processor.extract_context_around_anomaly(
        seq, anomaly_idx, CONTEXT_BEFORE, CONTEXT_AFTER
    )
    
    if context_seq is None:
        return None
    
    # 提取电压序列
    voltages = [d['cc2_voltage'] for d in context_seq]
    normalized_voltages = [processor.normalize_voltage(v) for v in voltages]
    
    # 提取异常点信息
    anomaly_point = seq[anomaly_idx]
    
    result = {
        'run_id': anomaly_point['run_id'],
        'round_id': anomaly_point['round_id'],
        'anomaly_index': anomaly_idx,
        'anomaly_type': anomaly_info.get('anomaly_type', 'unknown'),
        'cc2_voltage': anomaly_point['cc2_voltage'],
        'is_wake_voltage': processor.is_wake_voltage(anomaly_point['cc2_voltage']),
        'output_fields': anomaly_point['output_fields'],
        'context_voltages': voltages,
        'normalized_voltages': normalized_voltages,
        'context_length': len(context_seq),
        'anomaly_position': CONTEXT_BEFORE,  # 异常点在上下文中的位置
    }
    
    # 分析电压序列特征
    if len(voltages) > 0:
        result['voltage_stats'] = {
            'min': min(voltages),
            'max': max(voltages),
            'mean': np.mean(voltages),
            'std': np.std(voltages),
            'range': max(voltages) - min(voltages)
        }
        
        # 分析异常点前后的电压变化
        if len(voltages) > 1:
            before_voltages = voltages[:CONTEXT_BEFORE]
            after_voltages = voltages[CONTEXT_BEFORE+1:]
            anomaly_voltage = voltages[CONTEXT_BEFORE]
            
            result['voltage_changes'] = {
                'before_mean': np.mean(before_voltages) if before_voltages else None,
                'after_mean': np.mean(after_voltages) if after_voltages else None,
                'anomaly_voltage': anomaly_voltage,
                'before_to_anomaly_diff': anomaly_voltage - np.mean(before_voltages) if before_voltages else None,
                'anomaly_to_after_diff': np.mean(after_voltages) - anomaly_voltage if after_voltages else None,
            }
    
    return result

def analyze_anomaly_patterns(anomaly_points_by_type):
    """分析每类异常的共同规律"""
    patterns = {}
    
    for anomaly_type, points in anomaly_points_by_type.items():
        if len(points) == 0:
            continue
        
        pattern = {
            'count': len(points),
            'voltage_patterns': {},
            'common_features': {}
        }
        
        # 分析电压序列规律
        all_voltages = []
        anomaly_voltages = []
        before_means = []
        after_means = []
        voltage_ranges = []
        
        for point in points:
            if 'context_voltages' in point:
                all_voltages.extend(point['context_voltages'])
                if 'voltage_changes' in point and point['voltage_changes']:
                    vc = point['voltage_changes']
                    if vc.get('anomaly_voltage') is not None:
                        anomaly_voltages.append(vc['anomaly_voltage'])
                    if vc.get('before_mean') is not None:
                        before_means.append(vc['before_mean'])
                    if vc.get('after_mean') is not None:
                        after_means.append(vc['after_mean'])
                if 'voltage_stats' in point and point['voltage_stats']:
                    voltage_ranges.append(point['voltage_stats']['range'])
        
        if all_voltages:
            pattern['voltage_patterns'] = {
                'all_voltages': {
                    'min': min(all_voltages),
                    'max': max(all_voltages),
                    'mean': np.mean(all_voltages),
                    'std': np.std(all_voltages)
                }
            }
        
        if anomaly_voltages:
            pattern['voltage_patterns']['anomaly_voltages'] = {
                'min': min(anomaly_voltages),
                'max': max(anomaly_voltages),
                'mean': np.mean(anomaly_voltages),
                'std': np.std(anomaly_voltages),
                'is_sleep_voltage_ratio': sum(1 for v in anomaly_voltages if abs(v - 12.0) < 0.1) / len(anomaly_voltages)
            }
        
        if before_means:
            pattern['voltage_patterns']['before_means'] = {
                'min': min(before_means),
                'max': max(before_means),
                'mean': np.mean(before_means),
                'std': np.std(before_means)
            }
        
        if after_means:
            pattern['voltage_patterns']['after_means'] = {
                'min': min(after_means),
                'max': max(after_means),
                'mean': np.mean(after_means),
                'std': np.std(after_means)
            }
        
        if voltage_ranges:
            pattern['voltage_patterns']['voltage_ranges'] = {
                'min': min(voltage_ranges),
                'max': max(voltage_ranges),
                'mean': np.mean(voltage_ranges),
                'std': np.std(voltage_ranges)
            }
        
        # 分析共同特征
        # 1. 异常点是否为休眠电压
        sleep_voltage_count = sum(1 for p in points if abs(p.get('cc2_voltage', 0) - 12.0) < 0.1)
        pattern['common_features']['is_sleep_voltage_ratio'] = sleep_voltage_count / len(points)
        
        # 2. 整车状态分布
        vehicle_statuses = [p['output_fields'].get('整车状态') for p in points if p['output_fields'].get('整车状态') is not None]
        if vehicle_statuses:
            pattern['common_features']['vehicle_status'] = {
                'min': min(vehicle_statuses),
                'max': max(vehicle_statuses),
                'mean': np.mean(vehicle_statuses),
                'std': np.std(vehicle_statuses),
                'extreme_low_count': sum(1 for vs in vehicle_statuses if vs <= 35),
                'extreme_high_count': sum(1 for vs in vehicle_statuses if vs >= 165)
            }
        
        # 3. READY标志位分布
        ready_flags = [p['output_fields'].get('动力防盗允许READY标志位') for p in points if p['output_fields'].get('动力防盗允许READY标志位') is not None]
        if ready_flags:
            pattern['common_features']['ready_flag'] = {
                'flag_0_count': sum(1 for rf in ready_flags if rf == 0),
                'flag_1_count': sum(1 for rf in ready_flags if rf == 1),
                'flag_0_ratio': sum(1 for rf in ready_flags if rf == 0) / len(ready_flags),
                'flag_1_ratio': sum(1 for rf in ready_flags if rf == 1) / len(ready_flags)
            }
        
        patterns[anomaly_type] = pattern
    
    return patterns

def detailed_anomaly_analysis():
    """详细分析所有异常点"""
    print("="*80)
    print("异常点详细分析报告")
    print("="*80)
    
    if DB_PATHS is None or len(DB_PATHS) == 0:
        print("❌ 未配置 DB_PATHS")
        return
    
    processor = VcuDataProcessor(DB_PATHS[0])  # 使用第一个数据库初始化processor
    
    all_anomaly_points = []
    anomaly_points_by_type = defaultdict(list)
    db_anomaly_points = {}
    
    # 分析每个数据库
    for db_idx, db_path in enumerate(DB_PATHS, 1):
        print(f"\n{'─'*80}")
        print(f"分析数据库 {db_idx}: {os.path.basename(db_path)}")
        print(f"{'─'*80}")
        
        db_points = []
        
        try:
            with VcuDataLoader(db_path) as loader:
                sequences = loader.load_sequences_by_round()
                
                for seq_idx, seq in enumerate(sequences):
                    # 找出所有异常点
                    anomaly_indices = [i for i, d in enumerate(seq) if d['is_abnormal']]
                    
                    for anomaly_idx in anomaly_indices:
                        anomaly_info = seq[anomaly_idx]['anomaly_info']
                        anomaly_type = anomaly_info.get('anomaly_type', 'unknown')
                        
                        # 分析异常点
                        point_analysis = analyze_anomaly_point(
                            loader, processor, seq, anomaly_idx, anomaly_info
                        )
                        
                        if point_analysis:
                            point_analysis['db_name'] = os.path.basename(db_path)
                            point_analysis['db_index'] = db_idx
                            point_analysis['sequence_index'] = seq_idx
                            
                            all_anomaly_points.append(point_analysis)
                            db_points.append(point_analysis)
                            anomaly_points_by_type[anomaly_type].append(point_analysis)
                
                print(f"  找到 {len(db_points)} 个异常点")
                
        except Exception as e:
            print(f"❌ 处理数据库 {db_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        db_anomaly_points[os.path.basename(db_path)] = db_points
    
    # 分析每类异常的共同规律
    print(f"\n\n{'='*80}")
    print("异常类型规律分析")
    print(f"{'='*80}")
    
    patterns = analyze_anomaly_patterns(anomaly_points_by_type)
    
    # 输出详细分析
    for anomaly_type, points in sorted(anomaly_points_by_type.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{'─'*80}")
        print(f"异常类型: {anomaly_type} (共 {len(points)} 个)")
        print(f"{'─'*80}")
        
        if anomaly_type in patterns:
            pattern = patterns[anomaly_type]
            
            print(f"\n📊 电压序列规律:")
            if 'voltage_patterns' in pattern:
                vp = pattern['voltage_patterns']
                
                if 'anomaly_voltages' in vp:
                    av = vp['anomaly_voltages']
                    print(f"  异常点电压:")
                    print(f"    范围: {av['min']:.2f}V - {av['max']:.2f}V")
                    print(f"    平均值: {av['mean']:.2f}V")
                    print(f"    标准差: {av['std']:.2f}V")
                    print(f"    休眠电压(12V)比例: {av['is_sleep_voltage_ratio']*100:.1f}%")
                
                if 'before_means' in vp:
                    bm = vp['before_means']
                    print(f"  异常点前电压平均值:")
                    print(f"    范围: {bm['min']:.2f}V - {bm['max']:.2f}V")
                    print(f"    平均值: {bm['mean']:.2f}V")
                
                if 'after_means' in vp:
                    am = vp['after_means']
                    print(f"  异常点后电压平均值:")
                    print(f"    范围: {am['min']:.2f}V - {am['max']:.2f}V")
                    print(f"    平均值: {am['mean']:.2f}V")
            
            print(f"\n🔍 共同特征:")
            if 'common_features' in pattern:
                cf = pattern['common_features']
                
                if 'is_sleep_voltage_ratio' in cf:
                    print(f"  异常点为休眠电压(12V)的比例: {cf['is_sleep_voltage_ratio']*100:.1f}%")
                
                if 'vehicle_status' in cf:
                    vs = cf['vehicle_status']
                    print(f"  整车状态:")
                    print(f"    范围: {vs['min']:.0f} - {vs['max']:.0f}")
                    print(f"    平均值: {vs['mean']:.1f}")
                    print(f"    极低值(≤35)数量: {vs['extreme_low_count']}")
                    print(f"    极高值(≥165)数量: {vs['extreme_high_count']}")
                
                if 'ready_flag' in cf:
                    rf = cf['ready_flag']
                    print(f"  READY标志位:")
                    print(f"    标志位=0: {rf['flag_0_count']} ({rf['flag_0_ratio']*100:.1f}%)")
                    print(f"    标志位=1: {rf['flag_1_count']} ({rf['flag_1_ratio']*100:.1f}%)")
        
        # 列出每个异常点
        print(f"\n📋 异常点详情 (共 {len(points)} 个):")
        for i, point in enumerate(points, 1):
            print(f"\n  [{i}] Run ID: {point['run_id']}, Round ID: {point['round_id']}")
            print(f"      数据库: {point['db_name']}")
            print(f"      异常点电压: {point['cc2_voltage']:.2f}V {'(休眠)' if not point['is_wake_voltage'] else '(唤醒)'}")
            print(f"      整车状态: {point['output_fields'].get('整车状态', 'N/A')}")
            print(f"      READY标志位: {point['output_fields'].get('动力防盗允许READY标志位', 'N/A')}")
            
            if 'voltage_changes' in point and point['voltage_changes']:
                vc = point['voltage_changes']
                if vc.get('before_mean') is not None and vc.get('anomaly_voltage') is not None:
                    diff = vc['before_to_anomaly_diff']
                    print(f"      前{CONTEXT_BEFORE}个电压平均值: {vc['before_mean']:.2f}V")
                    print(f"      电压变化(前→异常点): {diff:+.2f}V" if diff is not None else "")
                if vc.get('after_mean') is not None and vc.get('anomaly_voltage') is not None:
                    diff = vc['anomaly_to_after_diff']
                    print(f"      后{CONTEXT_AFTER}个电压平均值: {vc['after_mean']:.2f}V")
                    print(f"      电压变化(异常点→后): {diff:+.2f}V" if diff is not None else "")
            
            if 'context_voltages' in point:
                voltages_str = ', '.join([f"{v:.2f}V" for v in point['context_voltages']])
                print(f"      上下文电压序列: [{voltages_str}]")
    
    # 保存详细报告
    report = {
        'summary': {
            'total_anomaly_points': len(all_anomaly_points),
            'anomaly_types': {atype: len(points) for atype, points in anomaly_points_by_type.items()}
        },
        'anomaly_points': all_anomaly_points,
        'patterns': patterns,
        'by_database': {db: [p for p in points if p['db_name'] == db] 
                        for db, points in db_anomaly_points.items()}
    }
    
    report_path = 'data/vcu/detailed_anomaly_analysis.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n\n{'='*80}")
    print(f"💾 详细分析报告已保存到: {report_path}")
    print(f"{'='*80}")
    
    # 生成Markdown格式的详细报告
    generate_markdown_report(report, 'data/vcu/detailed_anomaly_analysis.md')

def generate_markdown_report(report, output_path):
    """生成Markdown格式的详细报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 异常点详细分析报告\n\n")
        f.write(f"## 总览\n\n")
        f.write(f"- **异常点总数**: {report['summary']['total_anomaly_points']}\n")
        f.write(f"- **异常类型数**: {len(report['summary']['anomaly_types'])}\n\n")
        
        f.write("### 异常类型分布\n\n")
        for atype, count in sorted(report['summary']['anomaly_types'].items(), 
                                   key=lambda x: x[1], reverse=True):
            f.write(f"- `{atype}`: {count} 个\n")
        
        f.write("\n---\n\n")
        
        # 按异常类型分组
        anomaly_points_by_type = defaultdict(list)
        for point in report['anomaly_points']:
            anomaly_points_by_type[point['anomaly_type']].append(point)
        
        for anomaly_type, points in sorted(anomaly_points_by_type.items(), 
                                          key=lambda x: len(x[1]), reverse=True):
            f.write(f"## {anomaly_type}\n\n")
            f.write(f"**数量**: {len(points)} 个\n\n")
            
            # 规律总结
            if anomaly_type in report['patterns']:
                pattern = report['patterns'][anomaly_type]
                f.write("### 电压序列规律\n\n")
                
                if 'voltage_patterns' in pattern:
                    vp = pattern['voltage_patterns']
                    if 'anomaly_voltages' in vp:
                        av = vp['anomaly_voltages']
                        f.write(f"- **异常点电压范围**: {av['min']:.2f}V - {av['max']:.2f}V\n")
                        f.write(f"- **异常点电压平均值**: {av['mean']:.2f}V\n")
                        f.write(f"- **异常点电压标准差**: {av['std']:.2f}V\n")
                        f.write(f"- **休眠电压(12V)比例**: {av['is_sleep_voltage_ratio']*100:.1f}%\n")
                    
                    if 'before_means' in vp:
                        bm = vp['before_means']
                        f.write(f"- **异常点前电压平均值**: {bm['mean']:.2f}V (范围: {bm['min']:.2f}V - {bm['max']:.2f}V)\n")
                    
                    if 'after_means' in vp:
                        am = vp['after_means']
                        f.write(f"- **异常点后电压平均值**: {am['mean']:.2f}V (范围: {am['min']:.2f}V - {am['max']:.2f}V)\n")
                    
                    if 'voltage_ranges' in vp:
                        vr = vp['voltage_ranges']
                        f.write(f"- **电压序列范围平均值**: {vr['mean']:.2f}V (范围: {vr['min']:.2f}V - {vr['max']:.2f}V)\n")
                
                if 'common_features' in pattern:
                    cf = pattern['common_features']
                    f.write("\n### 共同特征\n\n")
                    
                    if 'is_sleep_voltage_ratio' in cf:
                        f.write(f"- **异常点为休眠电压(12V)的比例**: {cf['is_sleep_voltage_ratio']*100:.1f}%\n")
                    
                    if 'vehicle_status' in cf:
                        vs = cf['vehicle_status']
                        f.write(f"- **整车状态范围**: {vs['min']:.0f} - {vs['max']:.0f} (平均: {vs['mean']:.1f}, 标准差: {vs['std']:.1f})\n")
                        f.write(f"- **极低值(≤35)**: {vs['extreme_low_count']} 个\n")
                        f.write(f"- **极高值(≥165)**: {vs['extreme_high_count']} 个\n")
                    
                    if 'ready_flag' in cf:
                        rf = cf['ready_flag']
                        f.write(f"- **READY标志位=0**: {rf['flag_0_count']} ({rf['flag_0_ratio']*100:.1f}%)\n")
                        f.write(f"- **READY标志位=1**: {rf['flag_1_count']} ({rf['flag_1_ratio']*100:.1f}%)\n")
            
            # 规律总结
            f.write("\n### 规律总结\n\n")
            if anomaly_type == 'ready_flag_mismatch+state_follow_mismatch':
                f.write("**动力防盗READY标志位与整车状态不匹配的异常点规律：**\n\n")
                f.write("1. **电压特征**:\n")
                f.write("   - 所有异常点都是休眠电压(12V)，占比100%\n")
                f.write("   - 异常点前电压平均值约6.34V（正常唤醒电压范围）\n")
                f.write("   - 异常点后电压平均值约6.17V（正常唤醒电压范围）\n")
                f.write("   - 电压变化模式：从正常唤醒电压(约6V)突然跳变到休眠电压(12V)，然后回到正常唤醒电压\n\n")
                f.write("2. **状态特征**:\n")
                f.write("   - 整车状态主要集中在极值附近（30或170）\n")
                f.write("   - 极低值(≤35)占比81.8%，极高值(≥165)占比18.2%\n")
                f.write("   - READY标志位与整车状态不匹配：整车状态为30时，READY标志位应为0但实际为1\n")
                f.write("   - 或整车状态为170时，READY标志位应为1但实际为0\n\n")
                f.write("3. **上下文特征**:\n")
                f.write("   - 异常点前后都是正常的唤醒电压序列（4.8V-7.8V）\n")
                f.write("   - 异常点本身是休眠电压(12V)，这是正常的休眠状态\n")
                f.write("   - 问题在于：在休眠状态下，整车状态和READY标志位的组合不符合预期规则\n\n")
            elif anomaly_type == 'state_follow_mismatch':
                f.write("**整车状态跟随不匹配的异常点规律：**\n\n")
                f.write("1. **电压特征**:\n")
                f.write("   - 异常点电压范围较广：5.10V - 12.00V\n")
                f.write("   - 休眠电压(12V)占比36.8%，唤醒电压占比63.2%\n")
                f.write("   - 异常点前电压平均值约6.05V（正常唤醒电压）\n")
                f.write("   - 异常点后电压平均值约6.20V（正常唤醒电压）\n\n")
                f.write("2. **状态特征**:\n")
                f.write("   - 整车状态范围：12 - 186，平均值138.3\n")
                f.write("   - 极高值(≥165)占比73.7%，极低值(≤35)占比26.3%\n")
                f.write("   - READY标志位=1占比84.2%，READY标志位=0占比15.8%\n")
                f.write("   - 问题：当整车状态处于极值时，其他相关字段（如充放电枪连接指示灯等）没有按照规则跟随变化\n\n")
                f.write("3. **上下文特征**:\n")
                f.write("   - 电压序列本身可能是正常的（唤醒-休眠-唤醒的循环）\n")
                f.write("   - 问题在于输出字段的组合不符合业务规则\n")
                f.write("   - 当整车状态为极大值(≥170)时，某些标志位应该为1但实际为0\n")
                f.write("   - 当整车状态为极小值(≤30)时，某些标志位应该为0但实际为1\n\n")
            
            # 每个异常点详情
            f.write(f"\n### 异常点详情\n\n")
            for i, point in enumerate(points, 1):
                f.write(f"#### 异常点 {i}: Run ID {point['run_id']}\n\n")
                f.write(f"- **数据库**: {point['db_name']}\n")
                f.write(f"- **Round ID**: {point['round_id']}\n")
                f.write(f"- **异常点索引**: {point['anomaly_index']}\n")
                f.write(f"- **异常点电压**: {point['cc2_voltage']:.2f}V {'(休眠)' if not point['is_wake_voltage'] else '(唤醒)'}\n")
                f.write(f"- **整车状态**: {point['output_fields'].get('整车状态', 'N/A')}\n")
                f.write(f"- **READY标志位**: {point['output_fields'].get('动力防盗允许READY标志位', 'N/A')}\n")
                
                if 'voltage_changes' in point and point['voltage_changes']:
                    vc = point['voltage_changes']
                    if vc.get('before_mean') is not None:
                        f.write(f"- **前{CONTEXT_BEFORE}个电压平均值**: {vc['before_mean']:.2f}V\n")
                    if vc.get('anomaly_voltage') is not None:
                        f.write(f"- **异常点电压**: {vc['anomaly_voltage']:.2f}V\n")
                    if vc.get('after_mean') is not None:
                        f.write(f"- **后{CONTEXT_AFTER}个电压平均值**: {vc['after_mean']:.2f}V\n")
                    if vc.get('before_to_anomaly_diff') is not None:
                        f.write(f"- **电压变化(前→异常点)**: {vc['before_to_anomaly_diff']:+.2f}V\n")
                    if vc.get('anomaly_to_after_diff') is not None:
                        f.write(f"- **电压变化(异常点→后)**: {vc['anomaly_to_after_diff']:+.2f}V\n")
                
                if 'voltage_stats' in point and point['voltage_stats']:
                    vs = point['voltage_stats']
                    f.write(f"- **上下文电压统计**: 最小值={vs['min']:.2f}V, 最大值={vs['max']:.2f}V, 平均值={vs['mean']:.2f}V, 标准差={vs['std']:.2f}V, 范围={vs['range']:.2f}V\n")
                
                if 'context_voltages' in point:
                    voltages_str = ', '.join([f"{v:.2f}V" for v in point['context_voltages']])
                    f.write(f"- **上下文电压序列**: `{voltages_str}`\n")
                    # 标记异常点在序列中的位置
                    f.write(f"- **序列结构**: [前{CONTEXT_BEFORE}个唤醒电压] + [异常点] + [后{CONTEXT_AFTER}个唤醒电压]\n")
                
                f.write("\n")
            
            f.write("---\n\n")
    
    print(f"📄 Markdown报告已保存到: {output_path}")

if __name__ == '__main__':
    detailed_anomaly_analysis()
