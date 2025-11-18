import numpy as np

def score_voltage_sequence(seq):
    """
    根据真实异常窗口（0 → 高 → 0）形态进行评分：
    1. 脉冲结构奖励
    2. 多点连续上升 → 峰值 → 下降奖励
    3. 全 0 / 单点高惩罚
    """

    seq = np.array(seq).astype(float)
    L = len(seq)

    # -----------------------
    # 1. 全 0 惩罚
    # -----------------------
    if np.all(seq == 0):
        return 0.10

    # -----------------------
    # 2. 单点高值惩罚
    # -----------------------
    high_points = np.sum(seq > 0)
    if high_points == 1:
        return 0.20

    # -----------------------
    # 3. 脉冲形态评分 (主目标)
    # -----------------------
    # 找峰值
    peak_idx = np.argmax(seq)
    peak_val = seq[peak_idx]

    # 峰必须高于周围
    if peak_val <= 0:
        return 0.15

    # 上升段
    rise = seq[:peak_idx]
    if len(rise) > 1:
        rise_smooth = np.sum(np.diff(rise) > 0) / (len(rise) - 1)
    else:
        rise_smooth = 0

    # 下降段
    fall = seq[peak_idx:]
    if len(fall) > 1:
        fall_smooth = np.sum(np.diff(fall) < 0) / (len(fall) - 1)
    else:
        fall_smooth = 0

    # -----------------------
    # 4. 整体脉冲对称性（形状越像“山峰”越高）
    # -----------------------
    symmetry = 1 - abs((peak_idx / L) - 0.5)   # 顶点居中更好

    # -----------------------
    # 5. 归一化加权
    # -----------------------
    score = (
        0.35 * rise_smooth +
        0.35 * fall_smooth +
        0.20 * symmetry +
        0.10 * (peak_val / (peak_val + 1))
    )

    # 限制范围 0~1
    return float(max(0.0, min(1.0, score)))
