"""
测试异常点上下文提取功能
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sequence.vcu_data_process import VcuDataProcessor
from sequence.db_loader import VcuDataLoader
from configs.config_vcu import CONTEXT_BEFORE, CONTEXT_AFTER

def test_context_extraction():
    """测试上下文提取功能"""
    print("=" * 70)
    print("测试异常点上下文提取功能")
    print("=" * 70)
    
    # 加载数据
    db_path = 'database/db.db'
    processor = VcuDataProcessor(db_path)
    
    print(f"\n配置参数：")
    print(f"  CONTEXT_BEFORE = {CONTEXT_BEFORE} (向前提取 {CONTEXT_BEFORE} 个唤醒电压)")
    print(f"  CONTEXT_AFTER = {CONTEXT_AFTER} (向后提取 {CONTEXT_AFTER} 个唤醒电压)")
    
    with VcuDataLoader(db_path) as loader:
        sequences = loader.load_sequences_by_round(round_id=1)
        
        if not sequences:
            print("\n错误：没有找到数据")
            return
        
        seq = sequences[0]
        print(f"\n示例序列长度: {len(seq)} 个数据点")
        
        # 找出异常点
        anomaly_indices = [i for i, d in enumerate(seq) if d['is_abnormal']]
        print(f"异常点数量: {len(anomaly_indices)}")
        print(f"异常点位置: {anomaly_indices[:10]}... (显示前10个)")
        
        if not anomaly_indices:
            print("\n警告：没有找到异常点")
            return
        
        # 测试第一个异常点的上下文提取
        anomaly_idx = anomaly_indices[0]
        print(f"\n测试第一个异常点 (索引 {anomaly_idx}):")
        print(f"  run_id: {seq[anomaly_idx]['run_id']}")
        print(f"  CC2电压: {seq[anomaly_idx]['cc2_voltage']:.2f}V")
        print(f"  是否为唤醒电压: {processor.is_wake_voltage(seq[anomaly_idx]['cc2_voltage'])}")
        
        # 提取上下文
        context_seq = processor.extract_context_around_anomaly(
            seq, anomaly_idx, CONTEXT_BEFORE, CONTEXT_AFTER
        )
        
        if context_seq is None:
            print(f"\n无法提取足够的上下文（需要向前 {CONTEXT_BEFORE} 个，向后 {CONTEXT_AFTER} 个唤醒电压）")
        else:
            print(f"\n成功提取上下文，序列长度: {len(context_seq)}")
            print(f"  结构: [前 {CONTEXT_BEFORE} 个唤醒电压] + [异常点] + [后 {CONTEXT_AFTER} 个唤醒电压]")
            print(f"\n上下文序列详情：")
            for i, d in enumerate(context_seq):
                voltage = d['cc2_voltage']
                is_abnormal = d['is_abnormal']
                is_wake = processor.is_wake_voltage(voltage)
                marker = " <-- 异常点" if is_abnormal else ""
                wake_marker = " (唤醒)" if is_wake else " (休眠)"
                print(f"  [{i}] CC2电压: {voltage:.2f}V{wake_marker}{marker}")
        
        # 测试所有异常点的上下文提取
        print(f"\n测试所有异常点的上下文提取：")
        voltage_seqs, condition_vecs, abnormal_labels = processor.build_sequence_pairs(sequences)
        
        print(f"  成功提取的序列数量: {len(voltage_seqs)}")
        print(f"  条件向量数量: {len(condition_vecs)}")
        print(f"  异常标签数量: {len(abnormal_labels)}")
        
        if voltage_seqs:
            print(f"\n第一个提取的序列详情：")
            print(f"  序列长度: {len(voltage_seqs[0])}")
            print(f"  电压序列（归一化）: {[f'{v:.3f}' for v in voltage_seqs[0][:10]]}...")
            print(f"  条件向量维度: {len(condition_vecs[0])}")
            print(f"  异常标签: {abnormal_labels[0]}")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

if __name__ == '__main__':
    test_context_extraction()

