"""
详细分析多个数据库文件
生成完整的分析报告
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.db_loader import VcuDataLoader
from sequence.vcu_data_process import VcuDataProcessor
from configs.config_vcu import DB_PATHS
from collections import Counter
import json

def detailed_analysis():
    """详细分析所有数据库"""
    print("="*80)
    print("多数据库详细分析报告")
    print("="*80)
    
    if DB_PATHS is None or len(DB_PATHS) == 0:
        print("❌ 未配置 DB_PATHS")
        return
    
    # 1. 逐个数据库分析
    print("\n" + "="*80)
    print("第一部分：各数据库详细分析")
    print("="*80)
    
    all_anomaly_details = []
    db_summaries = []
    
    for idx, db_path in enumerate(DB_PATHS, 1):
        print(f"\n{'─'*80}")
        print(f"数据库 {idx}: {os.path.basename(db_path)}")
        print(f"{'─'*80}")
        
        try:
            with VcuDataLoader(db_path) as loader:
                # 基础统计
                stats = loader.get_statistics()
                sequences = loader.load_sequences_by_round()
                data_list = loader.load_test_data()
                
                # 异常类型详细分析
                anomaly_type_counter = Counter()
                anomaly_details = []
                
                for data in data_list:
                    if data['is_abnormal']:
                        anomaly_info = data['anomaly_info']
                        anomaly_type = anomaly_info.get('anomaly_type', 'unknown')
                        anomaly_type_counter[anomaly_type] += 1
                        
                        # 收集异常详情
                        detail = {
                            'run_id': data['run_id'],
                            'round_id': data['round_id'],
                            'cc2_voltage': data['cc2_voltage'],
                            'anomaly_type': anomaly_type,
                            'output_fields': data['output_fields']
                        }
                        anomaly_details.append(detail)
                
                # 输出统计
                print(f"\n📊 基础统计:")
                print(f"  总记录数: {stats['total_records']:,}")
                print(f"  有效记录数: {stats['valid_records']:,}")
                print(f"  异常记录数: {stats['abnormal_count']:,}")
                print(f"  正常记录数: {stats['normal_count']:,}")
                print(f"  异常率: {stats['abnormal_rate']:.2%}")
                
                if stats['voltage_stats']:
                    vs = stats['voltage_stats']
                    print(f"\n⚡ CC2 电压统计:")
                    print(f"  最小值: {vs['min']:.2f}V")
                    print(f"  最大值: {vs['max']:.2f}V")
                    print(f"  平均值: {vs['mean']:.2f}V")
                    print(f"  标准差: {vs['std']:.2f}V")
                
                print(f"\n📈 序列信息:")
                print(f"  序列数量: {len(sequences)}")
                if sequences:
                    seq_lens = [len(seq) for seq in sequences]
                    print(f"  序列长度: {min(seq_lens)} - {max(seq_lens)} (平均: {sum(seq_lens)/len(seq_lens):.1f})")
                
                print(f"\n🚨 异常类型分布:")
                if anomaly_type_counter:
                    for atype, count in anomaly_type_counter.most_common():
                        percentage = count / stats['abnormal_count'] * 100 if stats['abnormal_count'] > 0 else 0
                        print(f"  {atype:40s}: {count:3d} 条 ({percentage:5.1f}%)")
                else:
                    print("  无异常数据")
                
                # 保存摘要
                db_summaries.append({
                    'db_name': os.path.basename(db_path),
                    'db_path': db_path,
                    'total_records': stats['total_records'],
                    'valid_records': stats['valid_records'],
                    'abnormal_count': stats['abnormal_count'],
                    'normal_count': stats['normal_count'],
                    'abnormal_rate': stats['abnormal_rate'],
                    'voltage_stats': stats.get('voltage_stats', {}),
                    'sequence_count': len(sequences),
                    'anomaly_types': dict(anomaly_type_counter),
                    'anomaly_details': anomaly_details[:10]  # 只保存前10个异常详情
                })
                
                all_anomaly_details.extend(anomaly_details)
                
        except Exception as e:
            print(f"❌ 处理数据库 {db_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. 汇总分析
    print("\n\n" + "="*80)
    print("第二部分：合并数据汇总分析")
    print("="*80)
    
    total_records = sum(s['total_records'] for s in db_summaries)
    total_valid = sum(s['valid_records'] for s in db_summaries)
    total_abnormal = sum(s['abnormal_count'] for s in db_summaries)
    total_normal = sum(s['normal_count'] for s in db_summaries)
    
    print(f"\n📊 总体统计:")
    print(f"  数据库数量: {len(db_summaries)}")
    print(f"  总记录数: {total_records:,}")
    print(f"  有效记录数: {total_valid:,}")
    print(f"  异常记录数: {total_abnormal:,}")
    print(f"  正常记录数: {total_normal:,}")
    print(f"  总体异常率: {total_abnormal/total_valid*100:.2f}%" if total_valid > 0 else "  总体异常率: 0%")
    
    # 异常类型汇总
    all_anomaly_types = Counter()
    for summary in db_summaries:
        for atype, count in summary['anomaly_types'].items():
            all_anomaly_types[atype] += count
    
    print(f"\n🚨 异常类型汇总:")
    if all_anomaly_types:
        for atype, count in all_anomaly_types.most_common():
            percentage = count / total_abnormal * 100 if total_abnormal > 0 else 0
            print(f"  {atype:40s}: {count:3d} 条 ({percentage:5.1f}%)")
    
    # 3. 数据处理结果
    print("\n\n" + "="*80)
    print("第三部分：数据处理结果")
    print("="*80)
    
    try:
        processor = VcuDataProcessor(DB_PATHS)
        train_data, test_data, max_seq_len = processor.process_data()
        
        train_list = list(train_data)
        test_list = list(test_data)
        
        train_samples = sum(x[0].shape[0] for x in train_list)
        test_samples = sum(x[0].shape[0] for x in test_list)
        total_samples = train_samples + test_samples
        
        # 统计异常样本
        train_abnormal = sum(x[2].numpy().sum() for x in train_list)
        test_abnormal = sum(x[2].numpy().sum() for x in test_list)
        
        print(f"\n📦 处理后的数据:")
        print(f"  最大序列长度: {max_seq_len}")
        print(f"  训练集样本数: {train_samples:,}")
        print(f"  测试集样本数: {test_samples:,}")
        print(f"  总样本数: {total_samples:,}")
        print(f"  训练集异常样本: {int(train_abnormal):,}")
        print(f"  测试集异常样本: {int(test_abnormal):,}")
        print(f"  训练集异常率: {train_abnormal/train_samples*100:.2f}%" if train_samples > 0 else "  训练集异常率: 0%")
        print(f"  测试集异常率: {test_abnormal/test_samples*100:.2f}%" if test_samples > 0 else "  测试集异常率: 0%")
        
        if train_list:
            voltages, conditions, labels = train_list[0]
            print(f"\n📋 数据格式:")
            print(f"  电压序列形状: {voltages.shape}")
            print(f"  条件向量形状: {conditions.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  条件向量维度: {conditions.shape[1]} (应为9)")
        
        # 数据质量评估
        print(f"\n✅ 数据质量评估:")
        if total_samples > 0:
            print(f"  ✓ 成功提取 {total_samples} 个训练样本")
            if total_samples >= 20:
                print(f"  ✓ 样本数量充足（>=20）")
            else:
                print(f"  ⚠ 样本数量较少（<20），可能影响训练效果")
            
            if train_abnormal > 0 and test_abnormal > 0:
                print(f"  ✓ 训练集和测试集都包含异常样本")
            else:
                print(f"  ⚠ 部分数据集缺少异常样本")
            
            if max_seq_len <= 20:
                print(f"  ✓ 序列长度合理（<=20）")
            else:
                print(f"  ⚠ 序列长度较长（>{max_seq_len}），可能需要调整模型")
        
    except Exception as e:
        print(f"\n❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 保存分析报告
    report_path = 'data/vcu/multi_db_analysis_report.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report = {
        'summary': {
            'total_databases': len(db_summaries),
            'total_records': total_records,
            'total_valid': total_valid,
            'total_abnormal': total_abnormal,
            'total_normal': total_normal,
            'overall_abnormal_rate': total_abnormal/total_valid if total_valid > 0 else 0
        },
        'databases': db_summaries,
        'anomaly_type_summary': dict(all_anomaly_types)
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n💾 分析报告已保存到: {report_path}")
    print("="*80)

if __name__ == '__main__':
    detailed_analysis()

