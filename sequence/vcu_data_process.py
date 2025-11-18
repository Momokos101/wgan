"""
VCU 数据处理模块
处理 CC2 电压序列数据，识别异常，准备 GAN 训练数据
"""
import os
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf
from configs.config_vcu_data import SEQ_LEN

from sequence.db_loader import VcuDataLoader

# ---- 导入配置 ----
try:
    from configs.config_vcu import *
except ImportError:
    try:
        from config_vcu import *
    except ImportError:
        from config import *


class VcuDataProcessor:
    """VCU 数据处理器：从数据库加载 → 序列抽取 → 构造 GAN 训练数据"""

    def __init__(self, db_path, batch_size: int = BATCH_SIZE, split_ratio: float = SPLIT_RATIO):
        """
        Args:
            db_path: 数据库路径（可以是单个路径字符串，或路径列表）
            batch_size: 训练/测试批次大小（上限）
            split_ratio: 训练集比例
        """
       
        self.seq_len = SEQ_LEN  
        self.c_dim = C_DIM if "C_DIM" in globals() else 9
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
        self.cc2_min = CC2_MIN_VOLTAGE if "CC2_MIN_VOLTAGE" in globals() else 4.8
        self.cc2_max = CC2_MAX_VOLTAGE if "CC2_MAX_VOLTAGE" in globals() else 7.8

        # 休眠时的固定值（12V，超出正常范围，作为特殊标记）
        self.sleep_voltage = SLEEP_VOLTAGE if "SLEEP_VOLTAGE" in globals() else 12.0

        # 异常类型编码映射（目前不直接进条件向量，只作为扩展预留）
        self.anomaly_type_map = {
            "normal": 0,
            "state_follow_mismatch": 1,
            "error": 2,
            "stuck": 3,
            "ready_flag_mismatch": 4,  # 已整合到 state_follow_mismatch，但保留用于兼容
        }

    # ---------------------------------------------------------------------
    # 基础工具：归一化 / 反归一化
    # ---------------------------------------------------------------------
    def normalize_voltage(self, voltage: float) -> float:
        """
        归一化电压值到 [0, 1] 范围
        """
        # 如果是休眠电压（12V），归一化为 1.0
        if abs(voltage - self.sleep_voltage) < 0.1:
            return 1.0

        # 正常范围 4.8V-7.8V 归一化到 [0, 0.9]
        if self.cc2_min <= voltage <= self.cc2_max:
            return (voltage - self.cc2_min) / (self.cc2_max - self.cc2_min) * 0.9
        else:
            # 超出范围的值
            if voltage < self.cc2_min:
                return 0.0
            else:
                return 0.95

    def denormalize_voltage(self, normalized: float) -> float:
        """
        反归一化电压值
        """
        if normalized >= 0.9:
            return self.sleep_voltage

        # 从 [0, 0.9] 映射回 [4.8, 7.8]
        return normalized / 0.9 * (self.cc2_max - self.cc2_min) + self.cc2_min

    # ---------------------------------------------------------------------
    # 特征与异常编码
    # ---------------------------------------------------------------------
    def extract_voltage_features(self, voltages: List[float], raw_voltages: List[float]) -> List[float]:
        """
        提取 CC2 电压序列的特征：
        1. 震荡顶峰比例（局部极大值且电压 >= 7.6V）
        2. 连续极大值比例（7.6V-7.8V）
        3. 边界值比例（接近 4.8V 或 7.8V）
        4. 震荡强度（标准差，归一化）
        5. 电压范围（max-min，归一化）
        """
        if len(voltages) < 2:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        features = []

        # 1. 震荡顶峰
        peak_count = 0
        for i in range(1, len(raw_voltages) - 1):
            if (
                raw_voltages[i] > raw_voltages[i - 1]
                and raw_voltages[i] > raw_voltages[i + 1]
                and raw_voltages[i] >= 7.6
            ):
                peak_count += 1
        peak_ratio = peak_count / max(len(raw_voltages) - 2, 1)
        features.append(peak_ratio)

        # 2. 连续极大值（7.6-7.8V）
        max_value_count = sum(1 for v in raw_voltages if 7.6 <= v <= 7.8)
        max_value_ratio = max_value_count / len(raw_voltages)
        features.append(max_value_ratio)

        # 3. 边界值（接近 4.8V 或 7.8V）
        boundary_count = sum(
            1 for v in raw_voltages if abs(v - 4.8) < 0.2 or abs(v - 7.8) < 0.2
        )
        boundary_ratio = boundary_count / len(raw_voltages)
        features.append(boundary_ratio)

        # 4. 震荡强度（标准差）
        if len(raw_voltages) > 1:
            std = np.std(raw_voltages)
            std_norm = min(std / 3.0, 1.0)  # 假设最大标准差约为 3.0V
        else:
            std_norm = 0.0
        features.append(std_norm)

        # 5. 电压范围
        if len(raw_voltages) > 0:
            voltage_range = max(raw_voltages) - min(raw_voltages)
            range_norm = min(voltage_range / 3.0, 1.0)  # 假设最大范围约为 3.0V
        else:
            range_norm = 0.0
        features.append(range_norm)

        return features

    def encode_anomaly_type(self, anomaly_info: dict) -> int:
        """
        将异常信息字典编码为类别 ID（预留）
        """
        if not anomaly_info.get("is_abnormal", False):
            return 0

        anomaly_type = anomaly_info.get("anomaly_type", "normal")

        if "state_follow_mismatch" in anomaly_type:
            return 1
        if "error" in anomaly_type:
            return 2
        if "stuck" in anomaly_type:
            return 3
        if "ready_flag_mismatch" in anomaly_type:
            return 4
        return 0

    # ---------------------------------------------------------------------
    # 上下文提取与序列构造
    # ---------------------------------------------------------------------
    def is_wake_voltage(self, voltage: float) -> bool:
        """
        判断是否为唤醒电压（排除休眠电压 12V）
        """
        return self.cc2_min <= voltage <= self.cc2_max

    def extract_context_around_anomaly(
        self,
        seq: List[dict],
        anomaly_idx: int,
        context_before: int,
        context_after: int,
    ) -> Optional[List[dict]]:
        """
        以异常点为中心，提取前后上下文（只包含唤醒电压）
        """
        if anomaly_idx < 0 or anomaly_idx >= len(seq):
            return None

        # 向前提取 N 个唤醒电压
        before_voltages: List[dict] = []
        i = anomaly_idx - 1
        while i >= 0 and len(before_voltages) < context_before:
            v = seq[i]["cc2_voltage"]
            if self.is_wake_voltage(v):
                before_voltages.insert(0, seq[i])  # 保持时间顺序
            i -= 1

        # 向后提取 M 个唤醒电压
        after_voltages: List[dict] = []
        i = anomaly_idx + 1
        while i < len(seq) and len(after_voltages) < context_after:
            v = seq[i]["cc2_voltage"]
            if self.is_wake_voltage(v):
                after_voltages.append(seq[i])
            i += 1

        # 上下文不足则放弃该异常点
        if len(before_voltages) < context_before or len(after_voltages) < context_after:
    # 如果一个唤醒电压都没找到，就退回到用原始索引上下文
            if not before_voltages and not after_voltages:
                start = max(0, anomaly_idx - context_before)
                end = min(len(seq), anomaly_idx + context_after + 1)
                context_seq = seq[start:end]
            else:
        # 否则就用已有的前后唤醒电压
                context_seq = before_voltages + [seq[anomaly_idx]] + after_voltages
        else:
    # 正常情况：前 N + 异常点 + 后 M
            context_seq = before_voltages + [seq[anomaly_idx]] + after_voltages

        return context_seq

    def build_sequence_pairs(self, sequences):
        """
        从 VcuDataLoader 提供的序列中构造 GAN 训练样本:
        voltage_seqs:   归一化电压窗口 (seq_len)
        condition_vecs: 条件向量（9维或 C_DIM）
        abnormal_labels: 是否异常（这里固定为 1）
        """
        voltage_seqs = []
        condition_vecs = []
        abnormal_labels = []

        seq_len = self.seq_len
        expected_c_dim = globals().get("C_DIM", 9)

        # 全局阈值（可能没定义）
        high_th = globals().get("BOUNDARY_JUDGE", None)
        low_th = globals().get("LOWER_JUDGE", None)

        for seq_idx, seq in enumerate(sequences):
            if not seq:
                continue

            raw_values = np.array(
                [item.get("cc2_voltage", item.get("voltage", 0.0)) for item in seq],
                dtype=np.float32
            )

            # 正确：逐点 normalize（不能直接处理 array）
            norm_values = np.array(
                [self.normalize_voltage(v) for v in raw_values],
                dtype=np.float32
            )

            # ----- 找异常点 -----
            if any(item.get("is_abnormal", False) for item in seq):
                abnormal_positions = [
                    i for i, item in enumerate(seq) if item.get("is_abnormal", False)
                ]
            else:
                if high_th is not None and low_th is not None:
                    mask = (raw_values >= high_th) | (raw_values <= low_th)
                else:
                    mask = (raw_values < self.cc2_min) | (raw_values > self.cc2_max)
                abnormal_positions = np.where(mask)[0].tolist()

            if not abnormal_positions:
                continue

            # 对每个异常点抽取上下文
            before = seq_len // 2
            after = seq_len - 1 - before

            for pos in abnormal_positions:

                ctx = self.extract_context_around_anomaly(
                    seq, pos, context_before=before, context_after=after
                )
                if not ctx:
                    continue

                ctx_raw = [item.get("cc2_voltage", item.get("voltage", 0.0)) for item in ctx]
                ctx_norm = [self.normalize_voltage(v) for v in ctx_raw]

                # ----- 电压统计特征 -----
                peak_ratio, max_ratio, boundary_ratio, std_norm, range_norm = \
                    self.extract_voltage_features(ctx_norm, ctx_raw)

                # ----- 条件向量 -----
                is_abnormal_flag = 1.0
                vehicle_state = ctx[-1].get("vehicle_state", 0.0)
                ready_val = ctx[-1].get("ready", ctx[-1].get("ready_flag", False))
                ready_flag = 1.0 if ready_val else 0.0
                anomaly_type_id = float(self.encode_anomaly_type(ctx[-1]))

                condition = [
                    float(is_abnormal_flag),
                    float(vehicle_state),
                    float(ready_flag),
                    float(anomaly_type_id),
                    float(peak_ratio),
                    float(max_ratio),
                    float(boundary_ratio),
                    float(std_norm),
                    float(range_norm),
                ]

                # 对齐 C_DIM
                if len(condition) != expected_c_dim:
                    if len(condition) < expected_c_dim:
                        condition = condition + [0.0] * (expected_c_dim - len(condition))
                    else:
                        condition = condition[:expected_c_dim]

                voltage_seqs.append(ctx_norm)
                condition_vecs.append(condition)
                abnormal_labels.append(1)

        voltage_seqs = np.array(voltage_seqs, dtype=np.float32)
        condition_vecs = np.array(condition_vecs, dtype=np.float32)
        abnormal_labels = np.array(abnormal_labels, dtype=np.int32)

        print(f"[build_sequence_pairs] 提取了 {len(voltage_seqs)} 条异常上下文窗口")

        return voltage_seqs, condition_vecs, abnormal_labels

    
    # ---------------------------------------------------------------------
    # 序列填充 / 主处理流程
    # ---------------------------------------------------------------------
    def pad_sequences(
        self,
        sequences: List[List[float]],
        max_length: Optional[int] = None,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """
        将不同长度序列填充到相同长度
        """
        if sequences is None or len(sequences) == 0:
            return np.zeros((0, max_len), dtype=np.float32)

        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                padded_seq = seq + [pad_value] * (max_length - len(seq))
            else:
                padded_seq = seq[:max_length]
            padded.append(padded_seq)

        return np.asarray(padded, dtype=np.float32)

    def process_data(
        self,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        处理数据，直接返回 numpy 数组（不创建 Dataset）
        Returns:
            voltage_array: (N, T)
            condition_array: (N, C_DIM)
            label_array: (N,)
            max_seq_len: int
        """
        # ---------------- 加载原始序列 ----------------
        all_sequences: List[List[dict]] = []
        total_loaded = 0

        for i, db_path in enumerate(self.db_paths):
            print(f"正在处理数据库 {i+1}/{len(self.db_paths)}: {db_path}")
            try:
                with VcuDataLoader(db_path) as loader:
                    seqs = loader.load_sequences_by_round()
                    if limit:
                        per_db_limit = limit // len(self.db_paths) if len(self.db_paths) > 1 else limit
                        if total_loaded < limit:
                            remaining = limit - total_loaded
                            seqs = seqs[: min(per_db_limit, remaining)]
                            total_loaded += len(seqs)
                        else:
                            seqs = []

                    all_sequences.extend(seqs)
                    print(f"  从 {db_path} 加载了 {len(seqs)} 个序列")
            except Exception as e:
                print(f"  警告: 处理数据库 {db_path} 时出错: {e}")
                continue

        sequences = all_sequences
        if limit and len(sequences) > limit:
            sequences = sequences[:limit]

        # ---------------- 构建样本 ----------------
        voltage_seqs, condition_vecs, abnormal_labels = self.build_sequence_pairs(sequences)

        if voltage_seqs.size == 0:
            print("警告: 未找到任何序列片段，返回空结果")
            return None, None, None, 0
        # 
        max_seq_len = 8

        voltage_array = self.pad_sequences(voltage_seqs, max_seq_len)
        condition_array = np.asarray(condition_vecs, dtype=np.float32)
        label_array = np.asarray(abnormal_labels, dtype=np.float32)

        # 再次检查条件维度
        if "C_DIM" in globals():
            if condition_array.shape[1] != C_DIM:
                raise ValueError(
                    f"条件向量维度 {condition_array.shape[1]} 与 C_DIM({C_DIM}) 不一致，"
                   f"请检查特征构造逻辑与配置。"
                )

        return voltage_array, condition_array, label_array, max_seq_len

    # ---------------------------------------------------------------------
    # 保存到 .npy（供 GAN 训练阶段复用）
    # ---------------------------------------------------------------------
    def save_processed_data(self, output_dir: str, limit: Optional[int] = None) -> None:
        """
        完整处理数据并保存到 output_dir 下的 .npy 文件
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"开始处理数据，共 {len(self.db_paths)} 个数据库文件...")
        voltage_array, condition_array, label_array, max_seq_len = self.process_data(limit)

        n_samples = voltage_array.shape[0]
        n_train = int(n_samples * self.split_ratio)
        if n_train <= 0 or n_train >= n_samples:
            # 至少要保证 train 和 test 都有数据
            n_train = max(1, n_samples - 1)

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_voltages = voltage_array[train_indices]
        train_conditions = condition_array[train_indices]
        train_labels = label_array[train_indices]

        test_voltages = voltage_array[test_indices]
        test_conditions = condition_array[test_indices]
        test_labels = label_array[test_indices]

        if train_voltages.shape[0] == 0:
            raise ValueError("训练集样本数为 0，请检查 split_ratio 或数据量是否足够。")
        if test_voltages.shape[0] == 0:
            raise ValueError("测试集样本数为 0，请检查 split_ratio 或数据量是否足够。")

        # 保存为 .npy
        np.save(os.path.join(output_dir, "train_voltages.npy"), train_voltages)
        np.save(os.path.join(output_dir, "train_conditions.npy"), train_conditions)
        np.save(os.path.join(output_dir, "train_labels.npy"), train_labels)

        np.save(os.path.join(output_dir, "test_voltages.npy"), test_voltages)
        np.save(os.path.join(output_dir, "test_conditions.npy"), test_conditions)
        np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)

        # 保存元数据
        metadata = {
            "max_sequence_length": int(8),
            "cc2_min": float(self.cc2_min),
            "cc2_max": float(self.cc2_max),
            "sleep_voltage": float(self.sleep_voltage),
        }
        import json

        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"数据已保存到 {output_dir}")
        print(f"训练集: {train_voltages.shape[0]} 条")
        print(f"测试集: {test_voltages.shape[0]} 条")
        print(f"最大序列长度: {max_seq_len}")


# -------------------------------------------------------------------------
# 对外入口：train_vcu.py 直接调用的 API
# -------------------------------------------------------------------------
def load_vcu_data(
    precision: str = "2",
    db_path=None,
    output_dir: str = "data/vcu",
    limit: Optional[int] = None,
):
    """
    加载 / 预处理 VCU 数据，返回 TensorFlow Dataset 与最大序列长度

    Returns:
        train_data: tf.data.Dataset((voltages, conditions, labels))
        test_data: tf.data.Dataset((voltages, conditions, labels))
        max_seq_len: int
    """
    # precision 目前只保留参数接口，实际由数据库与上下文控制

    # 从配置读取 db_path
    if db_path is None:
        try:
            from configs.config_vcu import DB_PATHS, DB_PATH
        except ImportError:
            try:
                from config_vcu import DB_PATHS, DB_PATH
            except ImportError:
                DB_PATHS, DB_PATH = None, "database/db.db"

        if DB_PATHS is not None:
            db_path = DB_PATHS
        else:
            db_path = DB_PATH

    processor = VcuDataProcessor(db_path)

    # 如果预处理文件不存在则先跑一遍完整 pipeline
    train_vol_path = os.path.join(output_dir, "train_voltages.npy")
    if not os.path.exists(train_vol_path):
        print("预处理数据文件不存在，开始处理数据...")
        processor.save_processed_data(output_dir, limit)

    # 读取 .npy
    train_voltages = np.load(os.path.join(output_dir, "train_voltages.npy"))
    train_conditions = np.load(os.path.join(output_dir, "train_conditions.npy"))
    train_labels = np.load(os.path.join(output_dir, "train_labels.npy"))

    test_voltages = np.load(os.path.join(output_dir, "test_voltages.npy"))
    test_conditions = np.load(os.path.join(output_dir, "test_conditions.npy"))
    test_labels = np.load(os.path.join(output_dir, "test_labels.npy"))

    # Dataset 的 batch_size 不能超过样本数
    n_train = train_voltages.shape[0]
    n_test = test_voltages.shape[0]
    batch_size_train = min(processor.batch_size, n_train)
    batch_size_test = min(processor.batch_size, n_test)

    train_data = (
        tf.data.Dataset.from_tensor_slices(
            (train_voltages, train_conditions, train_labels)
        )
        .batch(batch_size_train, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_data = (
        tf.data.Dataset.from_tensor_slices(
            (test_voltages, test_conditions, test_labels)
        )
        .batch(batch_size_test, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 加载元数据
    import json

    with open(os.path.join(output_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    max_seq_len = int(metadata["max_sequence_length"])

    return train_data, test_data, max_seq_len
