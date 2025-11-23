import numpy as np

def denormalize_to_voltage(normalized_value, cc2_min=4.8, cc2_max=7.8, sleep_voltage=12.0):
    """将归一化值转换为实际电压值（伏特）"""
    if normalized_value >= 0.9:
        return sleep_voltage
    return normalized_value / 0.9 * (cc2_max - cc2_min) + cc2_min

def score_voltage_sequence(seq, actual_voltages=None):
    """
    改进的评分函数，针对高风险区间和多样化模式：
    1. 高风险区间检测（7.6-7.8V 和 5.7-7.1V）
    2. 峰/谷/转折点检测
    3. 振荡强度评分
    4. 上下文宽度评分
    """
    seq = np.array(seq).astype(float)
    L = len(seq)
    
    # 转换为实际电压值（如果未提供）
    if actual_voltages is None:
        actual_voltages = [denormalize_to_voltage(v) for v in seq]
    actual_voltages = np.array(actual_voltages)
    
    # -----------------------
    # 1. 全 0 或无效序列惩罚
    # -----------------------
    if np.all(seq == 0) or np.all(np.isnan(seq)):
        return 0.10
    
    # -----------------------
    # 2. 高风险区间检测（加权高分）
    # -----------------------
    # 上边界高风险区间：7.6-7.8V
    upper_risk_mask = (actual_voltages >= 7.6) & (actual_voltages <= 7.8)
    upper_risk_count = np.sum(upper_risk_mask)
    upper_risk_score = min(upper_risk_count / L, 1.0) * 0.3  # 最高0.3分
    
    # 低-中段高风险区间：5.7-7.1V
    mid_risk_mask = (actual_voltages >= 5.7) & (actual_voltages <= 7.1)
    mid_risk_count = np.sum(mid_risk_mask)
    mid_risk_score = min(mid_risk_count / L, 1.0) * 0.25  # 最高0.25分
    
    # -----------------------
    # 3. 峰/谷/转折点检测
    # -----------------------
    peak_valley_score = 0.0
    if L >= 3:
        # 检测局部极大值（峰）
        peaks = []
        for i in range(1, L - 1):
            if actual_voltages[i] > actual_voltages[i-1] and actual_voltages[i] > actual_voltages[i+1]:
                peaks.append(i)
        
        # 检测局部极小值（谷）
        valleys = []
        for i in range(1, L - 1):
            if actual_voltages[i] < actual_voltages[i-1] and actual_voltages[i] < actual_voltages[i+1]:
                valleys.append(i)
        
        # 检测转折点（上升→下降 或 下降→上升）
        turn_points = []
        for i in range(1, L - 1):
            diff1 = actual_voltages[i] - actual_voltages[i-1]
            diff2 = actual_voltages[i+1] - actual_voltages[i]
            if (diff1 > 0 and diff2 < 0) or (diff1 < 0 and diff2 > 0):
                turn_points.append(i)
        
        # 峰/谷/转折点落在高风险区间 → 加权高分
        peak_risk_score = sum(1 for p in peaks if (7.6 <= actual_voltages[p] <= 7.8) or (5.7 <= actual_voltages[p] <= 7.1)) / max(len(peaks), 1) * 0.2
        valley_risk_score = sum(1 for v in valleys if (7.6 <= actual_voltages[v] <= 7.8) or (5.7 <= actual_voltages[v] <= 7.1)) / max(len(valleys), 1) * 0.15
        turn_risk_score = sum(1 for t in turn_points if (7.6 <= actual_voltages[t] <= 7.8) or (5.7 <= actual_voltages[t] <= 7.1)) / max(len(turn_points), 1) * 0.15
        
        peak_valley_score = peak_risk_score + valley_risk_score + turn_risk_score
    
    # -----------------------
    # 4. 振荡强度评分（宽度≥2.5V）
    # -----------------------
    if len(actual_voltages) > 0:
        voltage_range = np.max(actual_voltages) - np.min(actual_voltages)
        oscillation_score = min(voltage_range / 3.0, 1.0) * 0.15  # 宽度≥2.5V时接近满分
        if voltage_range >= 2.5:
            oscillation_score = 0.15  # 强振荡额外奖励
    else:
        oscillation_score = 0.0
    
    # -----------------------
    # 5. 上下文宽度评分（确保非平稳）
    # -----------------------
    if L >= 3:
        # 计算变化率
        diffs = np.abs(np.diff(actual_voltages))
        avg_change = np.mean(diffs)
        context_width_score = min(avg_change / 1.0, 1.0) * 0.1  # 变化越大越好
    else:
        context_width_score = 0.0
    
    # -----------------------
    # 6. 序列模式多样性奖励
    # -----------------------
    # 上边界型：连续在7.6-7.8V
    if upper_risk_count >= 2:
        pattern_bonus = 0.05
    # 强振荡型：电压范围大且包含高风险区间
    elif voltage_range >= 2.5 and (upper_risk_count > 0 or mid_risk_count > 0):
        pattern_bonus = 0.05
    # 转折型：有明显的下降→谷→反弹
    elif len(valleys) > 0 and len(peaks) > 0:
        pattern_bonus = 0.05
    else:
        pattern_bonus = 0.0
    
    # -----------------------
    # 7. 综合评分
    # -----------------------
    total_score = (
        upper_risk_score +
        mid_risk_score +
        peak_valley_score +
        oscillation_score +
        context_width_score +
        pattern_bonus
    )
    
    # 限制范围 0~1
    return float(max(0.0, min(1.0, total_score)))
