"""
分析多个数据库的数据处理结果
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.db_loader import VcuDataLoader
from sequence.vcu_data_process import VcuDataProcessor
from configs.config_vcu import DB_PATHS, DB_PATH
import json

def analyze_databases():
    """分析多个数据库的数据"""
    print("=" * 80)
    print("多数据库数据分析")
    print("=" * 80)
    
    # 确定要分析的数据库列表
    if DB_PATHS is not None:
        db_paths = DB_PATHS
    else:
        db_paths = [DB_PATH]
    
    print(f"\n共 {len(db_paths)} 个数据库文件:")
    for i, db_path in enumerate(db_paths, 1):
        print(f"  {i}. {db_path}")
    
    # 逐个分析每个数据库
    all_stats = []
    total_records = 0
    total_valid = 0
    total_abnormal = 0
    
    print("\n" + "=" * 80)
    print("各数据库详细统计")
    print("=" * 80)
    
    for i, db_path in enumerate(db_paths, 1):
        print(f"\n【数据库 {i}/{len(db_paths)}: {db_path}】")
        print("-" * 80)
        
        if not os.path.exists(db_path):
            print(f"  ⚠️  警告: 文件不存在，跳过")
            continue
        
        try:
            with VcuDataLoader(db_path) as loader:
                stats = loader.get_statistics()
                all_stats.append({
                    'db_path': db_path,
                    'stats': stats
                })
                
                print(f"  总记录数: {stats['total_records']}")
                print(f"  有效记录数: {stats['valid_records']}")
                print(f"  异常记录数: {stats['abnormal_count']}")
                print(f"  正常记录数: {stats['normal_count']}")
                print(f"  异常率: {stats['abnormal_rate']:.2%}")
                
                if stats['voltage_stats']:
                    vs = stats['voltage_stats']
                    print(f"  CC2电压范围: {vs['min']:.2f}V ~ {vs['max']:.2f}V")
                    print(f"  CC2电压平均值: {vs['mean']:.2f}V")
                    print(f"  CC2电压标准差: {vs['std']:.2f}V")
                
                total_records += stats['total_records']
                total_valid += stats['valid_records']
                total_abnormal += stats['abnormal_count']
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    print(f"  总记录数: {total_records}")
    print(f"  总有效记录数: {total_valid}")
    print(f"  总异常记录数: {total_abnormal}")
    print(f"  总正常记录数: {total_valid - total_abnormal}")
    if total_valid > 0:
        print(f"  总异常率: {total_abnormal / total_valid:.2%}")
    
    # 处理数据并分析
    print("\n" + "=" * 80)
    print("数据处理和分析")
    print("=" * 80)
    
    try:
        processor = VcuDataProcessor(db_paths if DB_PATHS else DB_PATH)
        
        print("\n开始处理数据...")
        train_data, test_data, max_seq_len = processor.process_data()
        
        # 统计训练数据
        train_list = list(train_data)
        test_list = list(test_data)
        
        train_voltages = None
        train_conditions = None
        train_labels = None
        test_voltages = None
        test_conditions = None
        test_labels = None
        
        if train_list:
            train_voltages = [x[0].numpy() for x in train_list]
            train_conditions = [x[1].numpy() for x in train_list]
            train_labels = [x[2].numpy() for x in train_list]
        
        if test_list:
            test_voltages = [x[0].numpy() for x in test_list]
            test_conditions = [x[1].numpy() for x in test_list]
            test_labels = [x[2].numpy() for x in test_list]
        
        import numpy as np
        
        if train_voltages:
            train_voltages_all = np.concatenate(train_voltages, axis=0)
            train_conditions_all = np.concatenate(train_conditions, axis=0)
            train_labels_all = np.concatenate(train_labels, axis=0)
            
            print(f"\n训练数据统计:")
            print(f"  样本数量: {len(train_voltages_all)}")
            print(f"  序列长度: {max_seq_len}")
            print(f"  电压序列形状: {train_voltages_all.shape}")
            print(f"  条件向量形状: {train_conditions_all.shape}")
            print(f"  异常样本数: {np.sum(train_labels_all == 1.0)}")
            print(f"  正常样本数: {np.sum(train_labels_all == 0.0)}")
            print(f"  异常率: {np.sum(train_labels_all == 1.0) / len(train_labels_all):.2%}")
            
            print(f"\n  电压序列统计:")
            print(f"    最小值: {train_voltages_all.min():.4f}")
            print(f"    最大值: {train_voltages_all.max():.4f}")
            print(f"    平均值: {train_voltages_all.mean():.4f}")
            print(f"    标准差: {train_voltages_all.std():.4f}")
            
            print(f"\n  条件向量统计 (9维):")
            for i in range(train_conditions_all.shape[1]):
                print(f"    维度 {i+1}: min={train_conditions_all[:, i].min():.4f}, "
                      f"max={train_conditions_all[:, i].max():.4f}, "
                      f"mean={train_conditions_all[:, i].mean():.4f}")
        
        if test_voltages:
            test_voltages_all = np.concatenate(test_voltages, axis=0)
            test_conditions_all = np.concatenate(test_conditions, axis=0)
            test_labels_all = np.concatenate(test_labels, axis=0)
            
            print(f"\n测试数据统计:")
            print(f"  样本数量: {len(test_voltages_all)}")
            print(f"  异常样本数: {np.sum(test_labels_all == 1.0)}")
            print(f"  正常样本数: {np.sum(test_labels_all == 0.0)}")
            print(f"  异常率: {np.sum(test_labels_all == 1.0) / len(test_labels_all):.2%}")
        
        print("\n" + "=" * 80)
        print("✅ 数据处理完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 数据处理出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    analyze_databases()

