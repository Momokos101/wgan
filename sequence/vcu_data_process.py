"""
VCU 数据处理模块
处理 CC2 电压序列数据，识别异常，准备 GAN 训练数据
"""
import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional
from sequence.db_loader import VcuDataLoader
try:
    from configs.config_vcu import *
except ImportError:
    try:
        from config_vcu import *
    except ImportError:
        from config import *


class VcuDataProcessor:
    """VCU 数据处理器"""
    
    def __init__(self, db_path, batch_size: int = BATCH_SIZE, split_ratio: float = SPLIT_RATIO):
        """
        初始化数据处理器
        
        Args:
            db_path: 数据库路径（可以是单个路径字符串，或路径列表）
            batch_size: 批次大小
            split_ratio: 训练集比例
        """
        # 支持单个路径或路径列表
        if isinstance(db_path, str):
            self.db_paths = [db_path]
        elif isinstance(db_path, list):
            self.db_paths = db_path
        else:
            raise ValueError(f"db_path 必须是字符串或列表，当前类型: {type(db_path)}")
        
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        
        # CC2 电压范围：4.8V - 7.8V
        self.cc2_min = 4.8
        self.cc2_max = 7.8
        
        # 休眠时的固定值（12V，超出正常范围，作为特殊标记）
        self.sleep_voltage = 12.0
        
        # 异常类型编码映射
        self.anomaly_type_map = {
            'normal': 0,
            'state_follow_mismatch': 1,
            'error': 2,
            'stuck': 3,
            'ready_flag_mismatch': 4,  # 已整合到state_follow_mismatch，但保留用于兼容
        }
        
    def normalize_voltage(self, voltage: float) -> float:
        """
        归一化电压值到 [0, 1] 范围
        
        Args:
            voltage: 原始电压值
            
        Returns:
            归一化后的值
        """
        # 如果是休眠电压（12V），归一化为 1.0
        if abs(voltage - self.sleep_voltage) < 0.1:
            return 1.0
        
        # 正常范围 4.8V-7.8V 归一化到 [0, 0.9]
        if self.cc2_min <= voltage <= self.cc2_max:
            return (voltage - self.cc2_min) / (self.cc2_max - self.cc2_min) * 0.9
        else:
            # 超出范围的值，归一化到 [0.9, 1.0]
            if voltage < self.cc2_min:
                return 0.0
            else:
                return 0.95
    
    def denormalize_voltage(self, normalized: float) -> float:
        """
        反归一化电压值
        
        Args:
            normalized: 归一化后的值 [0, 1]
            
        Returns:
            原始电压值
        """
        if normalized >= 0.9:
            return self.sleep_voltage
        
        # 从 [0, 0.9] 映射回 [4.8, 7.8]
        return normalized / 0.9 * (self.cc2_max - self.cc2_min) + self.cc2_min
    
    def extract_voltage_features(self, voltages: List[float], raw_voltages: List[float]) -> List[float]:
        """
        提取CC2电压序列的特征
        
        特征包括：
        1. 是否在震荡顶峰（局部极大值）
        2. 是否连续处于极大值附近（7.6V-7.8V）
        3. 是否在边界值附近（4.8V或7.8V）
        4. 震荡强度（标准差）
        5. 电压范围（最大值-最小值）
        
        Args:
            voltages: 归一化后的电压序列
            raw_voltages: 原始电压值（用于特征计算）
            
        Returns:
            特征向量 [震荡顶峰比例, 连续极大值比例, 边界值比例, 震荡强度, 电压范围]
        """
        if len(voltages) < 2:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        features = []
        
        # 1. 震荡顶峰检测（局部极大值）
        peak_count = 0
        for i in range(1, len(raw_voltages) - 1):
            if (raw_voltages[i] > raw_voltages[i-1] and 
                raw_voltages[i] > raw_voltages[i+1] and
                raw_voltages[i] >= 7.6):  # 顶峰在7.6V以上
                peak_count += 1
        peak_ratio = peak_count / max(len(raw_voltages) - 2, 1)
        features.append(peak_ratio)
        
        # 2. 连续极大值检测（7.6V-7.8V）
        max_value_count = sum(1 for v in raw_voltages if 7.6 <= v <= 7.8)
        max_value_ratio = max_value_count / len(raw_voltages)
        features.append(max_value_ratio)
        
        # 3. 边界值检测（接近4.8V或7.8V）
        boundary_count = sum(1 for v in raw_voltages 
                           if abs(v - 4.8) < 0.2 or abs(v - 7.8) < 0.2)
        boundary_ratio = boundary_count / len(raw_voltages)
        features.append(boundary_ratio)
        
        # 4. 震荡强度（标准差）
        if len(raw_voltages) > 1:
            std = np.std(raw_voltages)
            # 归一化到[0,1]，假设最大标准差为3.0
            std_norm = min(std / 3.0, 1.0)
        else:
            std_norm = 0.0
        features.append(std_norm)
        
        # 5. 电压范围（最大值-最小值）
        if len(raw_voltages) > 0:
            voltage_range = max(raw_voltages) - min(raw_voltages)
            # 归一化到[0,1]，假设最大范围为3.0V
            range_norm = min(voltage_range / 3.0, 1.0)
        else:
            range_norm = 0.0
        features.append(range_norm)
        
        return features
    
    def encode_anomaly_type(self, anomaly_info: dict) -> int:
        """
        编码异常类型
        
        Args:
            anomaly_info: 异常信息字典
            
        Returns:
            异常类型编码（0-4）
        """
        if not anomaly_info.get('is_abnormal', False):
            return 0
        
        anomaly_type = anomaly_info.get('anomaly_type', 'normal')
        
        # 优先判断主要异常类型
        if 'state_follow_mismatch' in anomaly_type:
            return 1
        elif 'error' in anomaly_type:
            return 2
        elif 'stuck' in anomaly_type:
            return 3
        elif 'ready_flag_mismatch' in anomaly_type:
            return 4
        else:
            return 0
    
    def is_wake_voltage(self, voltage: float) -> bool:
        """
        判断是否为唤醒电压（排除休眠电压12V）
        
        Args:
            voltage: 电压值
            
        Returns:
            True 表示是唤醒电压（4.8-7.8V），False 表示是休眠电压（12V）或其他
        """
        return self.cc2_min <= voltage <= self.cc2_max
    
    def extract_context_around_anomaly(self, seq: List[dict], anomaly_idx: int, 
                                       context_before: int, context_after: int) -> Optional[List[dict]]:
        """
        以异常点为中心，提取前后上下文（只包含唤醒电压）
        
        Args:
            seq: 完整序列
            anomaly_idx: 异常点的索引
            context_before: 向前提取的唤醒电压数量（N）
            context_after: 向后提取的唤醒电压数量（M）
            
        Returns:
            提取的上下文序列，如果无法提取足够的上下文则返回 None
        """
        if anomaly_idx < 0 or anomaly_idx >= len(seq):
            return None
        
        # 向前提取 N 个唤醒电压
        before_voltages = []
        i = anomaly_idx - 1
        while i >= 0 and len(before_voltages) < context_before:
            voltage = seq[i]['cc2_voltage']
            if self.is_wake_voltage(voltage):
                before_voltages.insert(0, seq[i])  # 保持顺序
            i -= 1
        
        # 向后提取 M 个唤醒电压
        after_voltages = []
        i = anomaly_idx + 1
        while i < len(seq) and len(after_voltages) < context_after:
            voltage = seq[i]['cc2_voltage']
            if self.is_wake_voltage(voltage):
                after_voltages.append(seq[i])
            i += 1
        
        # 如果无法提取足够的上下文，返回 None
        if len(before_voltages) < context_before or len(after_voltages) < context_after:
            return None
        
        # 组合：前N个唤醒电压 + 异常点 + 后M个唤醒电压
        # 注意：如果异常点本身是休眠电压，也包含它（因为它是异常点）
        context_seq = before_voltages + [seq[anomaly_idx]] + after_voltages
        
        return context_seq
    
    def build_sequence_pairs(self, sequences: List[List[dict]], 
                             context_before: Optional[int] = None,
                             context_after: Optional[int] = None) -> Tuple[List, List, List]:
        """
        构建序列对（以异常点为中心提取上下文）
        
        Args:
            sequences: 序列列表，每个序列包含一轮测试的数据
            context_before: 向前提取的唤醒电压数量（N），默认从配置读取
            context_after: 向后提取的唤醒电压数量（M），默认从配置读取
            
        Returns:
            (voltage_sequences, condition_vectors, abnormal_labels)
            - voltage_sequences: CC2 电压序列（归一化）
            - condition_vectors: 条件向量（压缩为单个向量，取序列平均值）
            - abnormal_labels: 异常标签（1.0 表示异常，0.0 表示正常）
        """
        # 从配置读取上下文参数
        if context_before is None:
            try:
                try:
                    from configs.config_vcu import CONTEXT_BEFORE
                except ImportError:
                    from config_vcu import CONTEXT_BEFORE
                context_before = CONTEXT_BEFORE
            except ImportError:
                context_before = 4  # 默认值
        
        if context_after is None:
            try:
                try:
                    from configs.config_vcu import CONTEXT_AFTER
                except ImportError:
                    from config_vcu import CONTEXT_AFTER
                context_after = CONTEXT_AFTER
            except ImportError:
                context_after = 4  # 默认值
        
        voltage_seqs = []
        condition_vecs = []
        abnormal_labels = []
        
        for seq in sequences:
            if len(seq) < 2:
                continue
            
            # 找出所有异常点
            anomaly_indices = [i for i, d in enumerate(seq) if d['is_abnormal']]
            
            if not anomaly_indices:
                # 如果没有异常点，跳过这个序列（或者可以选择提取正常序列）
                continue
            
            # 对于每个异常点，提取上下文
            for anomaly_idx in anomaly_indices:
                context_seq = self.extract_context_around_anomaly(
                    seq, anomaly_idx, context_before, context_after
                )
                
                if context_seq is None:
                    # 无法提取足够的上下文，跳过
                    continue
                
                # 提取电压序列（归一化和原始值）
                voltages = [self.normalize_voltage(d['cc2_voltage']) for d in context_seq]
                raw_voltages = [d['cc2_voltage'] for d in context_seq]
                
                # 提取电压序列特征
                voltage_features = self.extract_voltage_features(voltages, raw_voltages)
                
                # 构建条件向量
                # 基础条件：异常标志、整车状态、READY标志位
                condition_sum = [0.0, 0.0, 0.0]
                has_abnormal = False
                anomaly_type_sum = 0.0
                
                for d in context_seq:
                    output_fields = d['output_fields']
                    vehicle_status = output_fields.get('整车状态', 0)
                    ready_flag = output_fields.get('动力防盗允许READY标志位', 0)
                    
                    vehicle_status_norm = vehicle_status / 200.0 if vehicle_status else 0.0
                    is_abnormal = 1.0 if d['is_abnormal'] else 0.0
                    if is_abnormal:
                        has_abnormal = True
                        # 编码异常类型
                        anomaly_type_code = self.encode_anomaly_type(d.get('anomaly_info', {}))
                        anomaly_type_sum += anomaly_type_code
                    
                    condition_sum[0] += is_abnormal
                    condition_sum[1] += vehicle_status_norm
                    condition_sum[2] += float(ready_flag)
                
                seq_len = len(context_seq)
                base_condition = [c / seq_len for c in condition_sum]
                
                # 异常类型编码（归一化到[0,1]）
                anomaly_type_norm = (anomaly_type_sum / seq_len) / 4.0 if seq_len > 0 else 0.0
                
                # 组合条件向量：[基础条件(3) + 异常类型(1) + 电压特征(5)] = 9维
                condition = base_condition + [anomaly_type_norm] + voltage_features
                
                voltage_seqs.append(voltages)
                condition_vecs.append(condition)
                abnormal_labels.append(1.0 if has_abnormal else 0.0)
        
        return voltage_seqs, condition_vecs, abnormal_labels
    
    def pad_sequences(self, sequences: List[List], max_length: Optional[int] = None, 
                     pad_value: float = 0.0) -> np.ndarray:
        """
        填充序列到相同长度
        
        Args:
            sequences: 序列列表
            max_length: 最大长度，None 表示使用最大序列长度
            pad_value: 填充值
            
        Returns:
            填充后的 numpy 数组
        """
        if not sequences:
            return np.array([])
            
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                padded_seq = seq + [pad_value] * (max_length - len(seq))
            else:
                padded_seq = seq[:max_length]
            padded.append(padded_seq)
            
        return np.array(padded, dtype=np.float32)
    
    def process_data(self, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        处理数据，准备训练数据
        
        Args:
            limit: 限制加载的数据量（对每个数据库分别限制，None表示不限制）
            
        Returns:
            (train_data, test_data, max_sequence_length)
            - train_data: 训练数据 (voltage_seq, condition, abnormal_label)
            - test_data: 测试数据
            - max_sequence_length: 最大序列长度
        """
        # 从多个数据库加载数据并合并
        all_sequences = []
        total_loaded = 0
        
        for i, db_path in enumerate(self.db_paths):
            print(f"正在处理数据库 {i+1}/{len(self.db_paths)}: {db_path}")
            try:
                with VcuDataLoader(db_path) as loader:
                    sequences = loader.load_sequences_by_round()
                    if limit:
                        # 如果设置了limit，对每个数据库平均分配
                        per_db_limit = limit // len(self.db_paths) if len(self.db_paths) > 1 else limit
                        if total_loaded < limit:
                            remaining = limit - total_loaded
                            sequences = sequences[:min(per_db_limit, remaining)]
                            total_loaded += len(sequences)
                        else:
                            sequences = []
                    
                    all_sequences.extend(sequences)
                    print(f"  从 {db_path} 加载了 {len(sequences)} 个序列")
            except Exception as e:
                print(f"  警告: 处理数据库 {db_path} 时出错: {e}")
                continue
        
        sequences = all_sequences
        if limit and len(sequences) > limit:
            sequences = sequences[:limit]
        
        # 构建序列对
        voltage_seqs, condition_vecs, abnormal_labels = self.build_sequence_pairs(sequences)
        
        if not voltage_seqs:
            raise ValueError("没有有效的数据序列")
        
        # 确定最大序列长度
        max_seq_len = max(len(seq) for seq in voltage_seqs)
        
        # 填充序列（条件向量已经是单个向量，不需要填充）
        voltage_seqs_padded = self.pad_sequences(voltage_seqs, max_seq_len)
        condition_vecs = np.array(condition_vecs, dtype=np.float32)  # 已经是单个向量列表
        abnormal_labels = np.array(abnormal_labels, dtype=np.float32)
        
        # 分割训练集和测试集
        n_samples = len(voltage_seqs_padded)
        n_train = int(n_samples * self.split_ratio)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_voltages = voltage_seqs_padded[train_indices]
        train_conditions = condition_vecs[train_indices]
        train_labels = abnormal_labels[train_indices]
        
        test_voltages = voltage_seqs_padded[test_indices]
        test_conditions = condition_vecs[test_indices]
        test_labels = abnormal_labels[test_indices]
        
        # 检查数据量是否足够
        if len(train_voltages) < self.batch_size:
            print(f"警告: 训练数据量 ({len(train_voltages)}) 小于批次大小 ({self.batch_size})，将使用较小的批次大小")
            actual_batch_size = max(1, len(train_voltages) // 2)  # 至少分成2个批次
        else:
            actual_batch_size = self.batch_size
        
        if len(test_voltages) < self.batch_size:
            test_batch_size = max(1, len(test_voltages))
        else:
            test_batch_size = self.batch_size
        
        # 创建 TensorFlow Dataset
        train_data = tf.data.Dataset.from_tensor_slices((
            train_voltages,
            train_conditions,
            train_labels
        )).batch(actual_batch_size, drop_remainder=False)  # 改为 False，保留不完整批次
        
        test_data = tf.data.Dataset.from_tensor_slices((
            test_voltages,
            test_conditions,
            test_labels
        )).batch(test_batch_size, drop_remainder=False)
        
        return train_data, test_data, max_seq_len
    
    def save_processed_data(self, output_dir: str, limit: Optional[int] = None):
        """
        保存处理后的数据到文件
        
        Args:
            output_dir: 输出目录
            limit: 限制加载的数据量（对每个数据库分别限制）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"开始处理数据，共 {len(self.db_paths)} 个数据库文件...")
        train_data, test_data, max_seq_len = self.process_data(limit)
        
        # 转换为 numpy 数组保存
        train_list = list(train_data)
        test_list = list(test_data)
        
        # 检查是否有数据
        if len(train_list) == 0:
            raise ValueError(f"训练数据为空！可能原因：数据量太少（需要至少 {self.batch_size} 个样本）或批次大小设置过大")
        if len(test_list) == 0:
            raise ValueError(f"测试数据为空！可能原因：数据量太少（需要至少 {self.batch_size} 个样本）或批次大小设置过大")
        
        # 提取数据
        train_voltages = np.concatenate([x[0].numpy() for x in train_list], axis=0)
        train_conditions = np.concatenate([x[1].numpy() for x in train_list], axis=0)
        train_labels = np.concatenate([x[2].numpy() for x in train_list], axis=0)
        
        test_voltages = np.concatenate([x[0].numpy() for x in test_list], axis=0)
        test_conditions = np.concatenate([x[1].numpy() for x in test_list], axis=0)
        test_labels = np.concatenate([x[2].numpy() for x in test_list], axis=0)
        
        # 确保条件向量是 2D 的 (n_samples, c_dim)
        if train_conditions.ndim == 1:
            train_conditions = train_conditions.reshape(-1, 1) if train_conditions.shape[0] == len(train_voltages) else train_conditions
        if test_conditions.ndim == 1:
            test_conditions = test_conditions.reshape(-1, 1) if test_conditions.shape[0] == len(test_voltages) else test_conditions
        
        # 保存
        np.save(os.path.join(output_dir, 'train_voltages.npy'), train_voltages)
        np.save(os.path.join(output_dir, 'train_conditions.npy'), train_conditions)
        np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(output_dir, 'test_voltages.npy'), test_voltages)
        np.save(os.path.join(output_dir, 'test_conditions.npy'), test_conditions)
        np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
        
        # 保存元数据
        metadata = {
            'max_sequence_length': int(max_seq_len),
            'cc2_min': self.cc2_min,
            'cc2_max': self.cc2_max,
            'sleep_voltage': self.sleep_voltage
        }
        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"数据已保存到 {output_dir}")
        print(f"训练集: {len(train_voltages)} 条")
        print(f"测试集: {len(test_voltages)} 条")
        print(f"最大序列长度: {max_seq_len}")


def load_vcu_data(precision: str = '2', db_path=None, 
                  output_dir: str = 'data/vcu', limit: Optional[int] = None):
    """
    加载 VCU 数据
    
    Args:
        precision: 精度（保留用于兼容性）
        db_path: 数据库路径（可以是单个路径字符串，或路径列表，None表示从配置读取）
        output_dir: 输出目录
        limit: 限制加载的数据量
        
    Returns:
        (train_data, test_data, max_sequence_length)
    """
    # 如果未指定 db_path，从配置读取
    if db_path is None:
        try:
            try:
                from configs.config_vcu import DB_PATHS, DB_PATH
            except ImportError:
                from config_vcu import DB_PATHS, DB_PATH
            # 优先使用 DB_PATHS，如果为 None 则使用 DB_PATH
            if DB_PATHS is not None:
                db_path = DB_PATHS
            else:
                db_path = DB_PATH
        except ImportError:
            db_path = 'database/db.db'  # 默认值
    
    processor = VcuDataProcessor(db_path)
    
    # 如果预处理文件不存在，先处理数据
    if not os.path.exists(os.path.join(output_dir, 'train_voltages.npy')):
        print("预处理数据文件不存在，开始处理数据...")
        processor.save_processed_data(output_dir, limit)
    
    # 加载预处理的数据
    train_voltages = np.load(os.path.join(output_dir, 'train_voltages.npy'))
    train_conditions = np.load(os.path.join(output_dir, 'train_conditions.npy'))
    train_labels = np.load(os.path.join(output_dir, 'train_labels.npy'))
    
    test_voltages = np.load(os.path.join(output_dir, 'test_voltages.npy'))
    test_conditions = np.load(os.path.join(output_dir, 'test_conditions.npy'))
    test_labels = np.load(os.path.join(output_dir, 'test_labels.npy'))
    
    # 创建 Dataset
    train_data = tf.data.Dataset.from_tensor_slices((
        train_voltages,
        train_conditions,
        train_labels
    )).batch(processor.batch_size, drop_remainder=True)
    
    test_data = tf.data.Dataset.from_tensor_slices((
        test_voltages,
        test_conditions,
        test_labels
    )).batch(processor.batch_size, drop_remainder=True)
    
    # 加载元数据
    import json
    with open(os.path.join(output_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    max_seq_len = metadata['max_sequence_length']
    
    return train_data, test_data, max_seq_len

