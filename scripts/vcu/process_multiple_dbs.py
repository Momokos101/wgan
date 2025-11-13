"""
处理多个数据库文件并进行分析
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.db_loader import VcuDataLoader
from sequence.vcu_data_process import VcuDataProcessor
from configs.config_vcu import DB_PATHS, DB_PATH
import json

def analyze_database(db_path):
    """分析单个数据库的统计信息"""
    print(f"\n{'='*70}")
    print(f"分析数据库: {db_path}")
    print(f"{'='*70}")
    
    try:
        with VcuDataLoader(db_path) as loader:
            stats = loader.get_statistics()
            
            print(f"\n📊 数据统计:")
            print(f"  总记录数: {stats['total_records']}")
            print(f"  有效记录数: {stats['valid_records']}")
            print(f"  异常记录数: {stats['abnormal_count']}")
            print(f"  正常记录数: {stats['normal_count']}")
            print(f"  异常率: {stats['abnormal_rate']:.2%}")
            
            if stats['voltage_stats']:
                print(f"\n⚡ CC2 电压统计:")
                print(f"  最小值: {stats['voltage_stats']['min']:.2f}V")
                print(f"  最大值: {stats['voltage_stats']['max']:.2f}V")
                print(f"  平均值: {stats['voltage_stats']['mean']:.2f}V")
                print(f"  标准差: {stats['voltage_stats']['std']:.2f}V")
            
            # 加载序列数据
            sequences = loader.load_sequences_by_round()
            print(f"\n📈 序列统计:")
            print(f"  总序列数: {len(sequences)}")
            if sequences:
                seq_lens = [len(seq) for seq in sequences]
                print(f"  序列长度范围: {min(seq_lens)} - {max(seq_lens)}")
                print(f"  平均序列长度: {sum(seq_lens) / len(seq_lens):.1f}")
            
            # 分析异常类型分布
            data_list = loader.load_test_data()
            anomaly_types = {}
            for data in data_list:
                if data['is_abnormal']:
                    anomaly_type = data['anomaly_info'].get('anomaly_type', 'unknown')
                    anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            if anomaly_types:
                print(f"\n🚨 异常类型分布:")
                for atype, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {atype}: {count} 条")
            
            return stats, len(sequences)
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def process_all_databases():
    """处理所有数据库并分析"""
    print("="*70)
    print("多数据库处理和分析")
    print("="*70)
    
    # 确定要处理的数据库列表
    if DB_PATHS is not None and len(DB_PATHS) > 0:
        db_paths = DB_PATHS
        print(f"\n使用配置的 DB_PATHS，共 {len(db_paths)} 个数据库:")
    else:
        db_paths = [DB_PATH]
        print(f"\n使用单个数据库: {DB_PATH}")
    
    # 分析每个数据库
    all_stats = []
    total_sequences = 0
    
    for db_path in db_paths:
        stats, seq_count = analyze_database(db_path)
        if stats:
            all_stats.append((db_path, stats, seq_count))
            total_sequences += seq_count
    
    # 汇总统计
    print(f"\n\n{'='*70}")
    print("汇总统计")
    print(f"{'='*70}")
    
    total_records = sum(s[1]['total_records'] for s in all_stats)
    total_valid = sum(s[1]['valid_records'] for s in all_stats)
    total_abnormal = sum(s[1]['abnormal_count'] for s in all_stats)
    total_normal = sum(s[1]['normal_count'] for s in all_stats)
    
    print(f"\n📊 总体数据统计:")
    print(f"  数据库数量: {len(all_stats)}")
    print(f"  总记录数: {total_records}")
    print(f"  有效记录数: {total_valid}")
    print(f"  异常记录数: {total_abnormal}")
    print(f"  正常记录数: {total_normal}")
    print(f"  总体异常率: {total_abnormal/total_valid*100:.2f}%" if total_valid > 0 else "  总体异常率: 0%")
    print(f"  总序列数: {total_sequences}")
    
    # 处理数据
    print(f"\n\n{'='*70}")
    print("开始处理数据（合并所有数据库）")
    print(f"{'='*70}")
    
    try:
        processor = VcuDataProcessor(db_paths)
        
        # 处理数据
        train_data, test_data, max_seq_len = processor.process_data()
        
        # 统计处理后的数据
        train_list = list(train_data)
        test_list = list(test_data)
        
        train_voltages = sum(x[0].shape[0] for x in train_list)
        test_voltages = sum(x[0].shape[0] for x in test_list)
        
        print(f"\n✅ 数据处理完成!")
        print(f"\n📦 处理后的数据统计:")
        print(f"  最大序列长度: {max_seq_len}")
        print(f"  训练集样本数: {train_voltages}")
        print(f"  测试集样本数: {test_voltages}")
        print(f"  总样本数: {train_voltages + test_voltages}")
        
        # 查看一个批次的数据示例
        if train_list:
            voltages, conditions, labels = train_list[0]
            print(f"\n📋 数据格式示例:")
            print(f"  电压序列形状: {voltages.shape}")
            print(f"  条件向量形状: {conditions.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  条件向量维度: {conditions.shape[1]} (应为9)")
            print(f"  异常样本数: {int(labels.numpy().sum())}")
            print(f"  正常样本数: {int((1 - labels.numpy()).sum())}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = process_all_databases()
    if success:
        print(f"\n\n{'='*70}")
        print("✅ 所有数据库处理完成！")
        print(f"{'='*70}")
    else:
        print(f"\n\n{'='*70}")
        print("❌ 处理过程中出现错误")
        print(f"{'='*70}")

