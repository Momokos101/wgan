"""
测试 VCU 数据加载功能
"""
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.db_loader import VcuDataLoader
from sequence.vcu_data_process import VcuDataProcessor
from configs.config_vcu import DB_PATH
import json

def test_db_loader():
    """测试数据库加载器"""
    print("=" * 60)
    print("测试数据库加载器")
    print("=" * 60)
    
    db_path = DB_PATH
    
    with VcuDataLoader(db_path) as loader:
        # 获取统计信息
        stats = loader.get_statistics()
        print("\n数据统计:")
        print(f"  总记录数: {stats['total_records']}")
        print(f"  有效记录数: {stats['valid_records']}")
        print(f"  异常记录数: {stats['abnormal_count']}")
        print(f"  正常记录数: {stats['normal_count']}")
        print(f"  异常率: {stats['abnormal_rate']:.2%}")
        
        if stats['voltage_stats']:
            print(f"\nCC2 电压统计:")
            print(f"  最小值: {stats['voltage_stats']['min']:.2f}V")
            print(f"  最大值: {stats['voltage_stats']['max']:.2f}V")
            print(f"  平均值: {stats['voltage_stats']['mean']:.2f}V")
            print(f"  标准差: {stats['voltage_stats']['std']:.2f}V")
        
        # 加载前 10 条数据
        print("\n前 10 条数据示例:")
        data_list = loader.load_test_data(limit=10)
        for i, data in enumerate(data_list[:5], 1):
            print(f"\n  数据 {i}:")
            print(f"    Run ID: {data['run_id']}")
            print(f"    Round ID: {data['round_id']}")
            print(f"    CC2 电压: {data['cc2_voltage']:.2f}V")
            print(f"    是否异常: {data['is_abnormal']}")
            print(f"    输出字段: {json.dumps(data['output_fields'], ensure_ascii=False, indent=6)}")
        
        # 测试序列加载
        print("\n\n测试序列加载:")
        sequences = loader.load_sequences_by_round()
        print(f"  总序列数: {len(sequences)}")
        if sequences:
            print(f"  第一个序列长度: {len(sequences[0])}")
            print(f"  第一个序列示例:")
            for i, item in enumerate(sequences[0][:3], 1):
                print(f"    步骤 {i}: CC2={item['cc2_voltage']:.2f}V, 异常={item['is_abnormal']}")


def test_data_processor():
    """测试数据处理器"""
    print("\n\n" + "=" * 60)
    print("测试数据处理器")
    print("=" * 60)
    
    db_path = 'db.db'
    processor = VcuDataProcessor(db_path, batch_size=32)
    
    # 测试归一化
    print("\n测试电压归一化:")
    test_voltages = [4.8, 6.3, 7.8, 12.0, 5.0]
    for v in test_voltages:
        norm = processor.normalize_voltage(v)
        denorm = processor.denormalize_voltage(norm)
        print(f"  {v:.2f}V -> {norm:.4f} -> {denorm:.2f}V")
    
    # 测试数据加载（限制数量）
    print("\n测试数据加载（限制 5 条）:")
    try:
        train_data, test_data, max_seq_len = processor.process_data(limit=5)
        print(f"  最大序列长度: {max_seq_len}")
        print(f"  训练数据批次: {len(list(train_data))}")
        print(f"  测试数据批次: {len(list(test_data))}")
        
        # 查看一个批次
        for voltages, conditions, labels in train_data.take(1):
            print(f"\n  批次示例:")
            print(f"    电压序列形状: {voltages.shape}")
            print(f"    条件向量形状: {conditions.shape}")
            print(f"    标签形状: {labels.shape}")
            print(f"    第一个序列的电压范围: [{voltages[0].numpy().min():.4f}, {voltages[0].numpy().max():.4f}]")
            print(f"    第一个序列的条件: {conditions[0].numpy()}")
            print(f"    第一个序列的标签: {labels[0].numpy()}")
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        test_db_loader()
        test_data_processor()
        print("\n\n测试完成！")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

