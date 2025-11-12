"""
数据库读取模块
从 SQLite 数据库中读取 VCU 测试数据
"""
import sqlite3
import json
import numpy as np
from typing import List, Tuple, Dict, Optional


class VcuDataLoader:
    """VCU 测试数据加载器"""
    
    def __init__(self, db_path: str):
        """
        初始化数据加载器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """连接数据库"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _parse_numeric(self, v):
        """将字符串或其他类型安全转换为数字"""
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    
    def _get_input_value(self, item: Dict) -> Optional[float]:
        """
        根据 in_type/in_range 解析输入项的取值语义并给出代表值
        
        规则说明（来自用户）：
        - in_type: 1--范围内取值，2--固定值
        - in_range: 1--不包含边界值，2--固定值单值
        
        解析策略：
        - 若存在明确的 value 字段，优先使用 value
        - 若为范围型（in_type==1），且存在 min/max：
            - 取中点 (min+max)/2 作为代表值；对于“不包含边界值”同样使用中点
        - 若为固定值型（in_type==2）且存在 value，则返回 value
        """
        if not isinstance(item, dict):
            return None
        
        # 优先使用 value
        value = self._parse_numeric(item.get('value'))
        if value is not None:
            return value
        
        in_type = item.get('in_type')
        in_range = item.get('in_range')
        v_min = self._parse_numeric(item.get('min'))
        v_max = self._parse_numeric(item.get('max'))
        
        # 范围型：取中点
        if in_type == 1 and v_min is not None and v_max is not None:
            return (v_min + v_max) / 2.0
        
        # 固定值型但未提供 value 时，回退到 min 或 max
        if in_type == 2:
            if v_min is not None and v_max is not None and abs(v_min - v_max) < 1e-9:
                return v_min
            if v_min is not None:
                return v_min
            if v_max is not None:
                return v_max
        
        return None
    
    def _get_output_condition(self, item: Dict) -> Optional[Dict]:
        """
        根据 out_type/out_range 解析输出项的判定语义
        
        规则说明（来自用户）：
        - out_type: 1--阈值类型
        - out_range: 1--大于等于 2--等于 3--小于等于
        
        返回：
        { 'op': '>=/==/<=' , 'threshold': number } 或 None
        """
        if not isinstance(item, dict):
            return None
        out_type = item.get('out_type')
        out_range = item.get('out_range')
        threshold = self._parse_numeric(item.get('value') if 'value' in item else item.get('threshold'))
        if out_type != 1 or threshold is None:
            return None
        if out_range == 1:
            op = '>='
        elif out_range == 2:
            op = '=='
        elif out_range == 3:
            op = '<='
        else:
            return None
        return {'op': op, 'threshold': threshold}
    
    def extract_cc2_voltage(self, actual_input: str) -> Optional[float]:
        """
        从输入 JSON 中提取 CC2 电压值
        
        Args:
            actual_input: 输入的 JSON 字符串
            
        Returns:
            CC2 电压值，如果不存在则返回 None
        """
        try:
            input_data = json.loads(actual_input)
            if isinstance(input_data, list):
                for item in input_data:
                    if item.get('name') == 'CC2电压':
                        # 支持范围/固定值两种输入定义
                        v = self._get_input_value(item)
                        return None if v is None else float(v)
            elif isinstance(input_data, dict):
                if input_data.get('name') == 'CC2电压':
                    v = self._get_input_value(input_data)
                    return None if v is None else float(v)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"解析输入数据错误: {e}")
        return None
    
    def extract_output_fields(self, actual_output: str) -> Dict:
        """
        从输出 JSON 中提取关键字段
        
        Args:
            actual_output: 输出的 JSON 字符串
            
        Returns:
            包含关键字段的字典
        """
        result = {
            '动力防盗允许READY标志位': None,
            '整车状态': None,
            'CC2电压值': None
        }
        
        try:
            output_data = json.loads(actual_output)
            if output_data.get('status') == 'success' and 'data' in output_data:
                for item in output_data['data']:
                    name = item.get('name')
                    if name in result:
                        # 直接读 value，兼容布尔/数值
                        result[name] = item.get('value')
                    # 也解析输出阈值语义，若后续需要可使用
                    cond = self._get_output_condition(item)
                    if cond:
                        # 将条件以 name+'.cond' 的形式附加，避免破坏原有结构
                        result[name + '.cond'] = cond
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析输出数据错误: {e}")
            
        return result
    
    def _parse_expected_output(self, expected_json: Optional[str]) -> Dict[str, Dict]:
        """
        解析期望输出 JSON，返回字段名到期望条件的映射
        
        Args:
            expected_json: 期望输出的 JSON 字符串
            
        Returns:
            字典，key 为字段名，value 为 {'op': '>=/==/<=', 'threshold': number}
        """
        result = {}
        if not expected_json:
            return result
        
        try:
            expected_list = json.loads(expected_json)
            if isinstance(expected_list, list):
                for item in expected_list:
                    name = item.get('name')
                    if name:
                        cond = self._get_output_condition(item)
                        if cond:
                            result[name] = cond
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析期望输出错误: {e}")
        
        return result
    
    def _check_value_against_condition(self, actual_value: any, condition: Dict) -> bool:
        """
        检查实际值是否符合条件
        
        Args:
            actual_value: 实际值
            condition: 条件字典 {'op': '>=/==/<=', 'threshold': number}
            
        Returns:
            True 表示符合条件，False 表示不符合
        """
        if condition is None:
            return False
        
        op = condition.get('op')
        threshold = condition.get('threshold')
        
        if threshold is None:
            return False
        
        actual_num = self._parse_numeric(actual_value)
        if actual_num is None:
            return False
        
        if op == '>=':
            return actual_num >= threshold
        elif op == '==':
            # 对于浮点数，使用小的容差
            return abs(actual_num - threshold) < 1e-6
        elif op == '<=':
            return actual_num <= threshold
        else:
            return False
    
    def _extract_actual_values(self, actual_output: str) -> Dict[str, any]:
        """
        从 actual_output 中提取所有字段的值
        
        Args:
            actual_output: 实际输出的 JSON 字符串
            
        Returns:
            字典，key 为字段名，value 为实际值
        """
        result = {}
        try:
            output_data = json.loads(actual_output)
            if output_data.get('status') == 'success' and 'data' in output_data:
                for item in output_data['data']:
                    name = item.get('name')
                    if name:
                        result[name] = item.get('value')
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析实际输出错误: {e}")
        
        return result
    
    def detect_anomalies(self, actual_output: str, expected_output: Optional[str], 
                        expected_error_output: Optional[str], 
                        expected_stuck_output: Optional[str]) -> Dict:
        """
        检测异常，返回详细的异常信息
        
        Args:
            actual_output: 实际输出 JSON 字符串
            expected_output: 期望输出 JSON 字符串
            expected_error_output: 期望错误输出 JSON 字符串
            expected_stuck_output: 期望卡死输出 JSON 字符串
            
        Returns:
            异常信息字典，包含：
            - is_abnormal: bool, 是否异常
            - anomaly_type: str, 异常类型 ('range_violation', 'error', 'stuck', 'ready_flag_mismatch', 'normal')
            - violated_fields: List[str], 不符合取值范围要求的字段列表
            - matched_error_fields: List[str], 符合 error 条件的字段列表
            - matched_stuck_fields: List[str], 符合 stuck 条件的字段列表
        """
        result = {
            'is_abnormal': False,
            'anomaly_type': 'normal',
            'matched_error_fields': [],
            'matched_stuck_fields': []
        }
        
        # 提取实际值
        actual_values = self._extract_actual_values(actual_output)
        if not actual_values:
            return result
        
        # 1. 检查是否符合 expected_error_output
        # 只有当 expected_error_output 中提到的所有字段都符合条件时，才标记为 error
        error_conditions = self._parse_expected_output(expected_error_output)
        if error_conditions:
            matched_error_fields = []
            unmatched_error_fields = []
            all_error_matched = True
            
            for field_name, condition in error_conditions.items():
                actual_value = actual_values.get(field_name)
                if actual_value is None:
                    # 字段在 actual_output 中不存在，不匹配
                    all_error_matched = False
                    unmatched_error_fields.append(f"{field_name}(字段不存在)")
                elif self._check_value_against_condition(actual_value, condition):
                    # 字段符合条件
                    matched_error_fields.append(f"{field_name}(值={actual_value}, 条件={condition['op']}{condition['threshold']})")
                else:
                    # 字段存在但不符合条件
                    all_error_matched = False
                    unmatched_error_fields.append(f"{field_name}(值={actual_value}, 不符合条件={condition['op']}{condition['threshold']})")
            
            # 只有当所有字段都符合条件时，才标记为 error
            if all_error_matched and len(matched_error_fields) == len(error_conditions):
                result['is_abnormal'] = True
                if result['anomaly_type'] == 'normal':
                    result['anomaly_type'] = 'error'
                elif result['anomaly_type'] == 'stuck':
                    result['anomaly_type'] = 'error+stuck'
                result['matched_error_fields'] = matched_error_fields
        
        # 2. 检查是否符合 expected_stuck_output
        # 只有当 expected_stuck_output 中提到的所有字段都符合条件时，才标记为 stuck
        stuck_conditions = self._parse_expected_output(expected_stuck_output)
        if stuck_conditions:
            matched_stuck_fields = []
            unmatched_stuck_fields = []
            all_stuck_matched = True
            
            for field_name, condition in stuck_conditions.items():
                actual_value = actual_values.get(field_name)
                if actual_value is None:
                    # 字段在 actual_output 中不存在，不匹配
                    all_stuck_matched = False
                    unmatched_stuck_fields.append(f"{field_name}(字段不存在)")
                elif self._check_value_against_condition(actual_value, condition):
                    # 字段符合条件
                    matched_stuck_fields.append(f"{field_name}(值={actual_value}, 条件={condition['op']}{condition['threshold']})")
                else:
                    # 字段存在但不符合条件
                    all_stuck_matched = False
                    unmatched_stuck_fields.append(f"{field_name}(值={actual_value}, 不符合条件={condition['op']}{condition['threshold']})")
            
            # 只有当所有字段都符合条件时，才标记为 stuck
            if all_stuck_matched and len(matched_stuck_fields) == len(stuck_conditions):
                result['is_abnormal'] = True
                if result['anomaly_type'] == 'normal':
                    result['anomaly_type'] = 'stuck'
                elif result['anomaly_type'] == 'error':
                    result['anomaly_type'] = 'error+stuck'
                result['matched_stuck_fields'] = matched_stuck_fields
        
        # 3. 检查 READY 标志位与整车状态的匹配（原有逻辑）
        ready_flag = actual_values.get('动力防盗允许READY标志位')
        vehicle_status = actual_values.get('整车状态')
        
        if ready_flag is not None and vehicle_status is not None:
            # 导入配置（延迟导入避免循环依赖）
            try:
                try:
                    from configs.config_vcu import VEHICLE_STATUS_MIN, VEHICLE_STATUS_MAX, VEHICLE_STATUS_TOLERANCE
                except ImportError:
                    from config_vcu import VEHICLE_STATUS_MIN, VEHICLE_STATUS_MAX, VEHICLE_STATUS_TOLERANCE
            except ImportError:
                # 默认值
                VEHICLE_STATUS_MIN = 30
                VEHICLE_STATUS_MAX = 170
                VEHICLE_STATUS_TOLERANCE = 5
            
            # 异常情况1：整车状态接近极小值（30）时，READY标志位应该为0，若为1即错误
            if abs(vehicle_status - VEHICLE_STATUS_MIN) <= VEHICLE_STATUS_TOLERANCE:
                if ready_flag == 1:
                    result['is_abnormal'] = True
                    if result['anomaly_type'] == 'normal':
                        result['anomaly_type'] = 'ready_flag_mismatch'
                    else:
                        result['anomaly_type'] += '+ready_flag_mismatch'
                    if 'ready_flag_mismatch' not in result:
                        result['ready_flag_mismatch'] = []
                    result['ready_flag_mismatch'].append(f"整车状态={vehicle_status}(接近极小值{VEHICLE_STATUS_MIN})，但READY标志位={ready_flag}(应为0)")
            
            # 异常情况2：整车状态接近极大值（170）时，READY标志位应该为1，若为0即错误
            if abs(vehicle_status - VEHICLE_STATUS_MAX) <= VEHICLE_STATUS_TOLERANCE:
                if ready_flag == 0:
                    result['is_abnormal'] = True
                    if result['anomaly_type'] == 'normal':
                        result['anomaly_type'] = 'ready_flag_mismatch'
                    else:
                        result['anomaly_type'] += '+ready_flag_mismatch'
                    if 'ready_flag_mismatch' not in result:
                        result['ready_flag_mismatch'] = []
                    result['ready_flag_mismatch'].append(f"整车状态={vehicle_status}(接近极大值{VEHICLE_STATUS_MAX})，但READY标志位={ready_flag}(应为1)")
        
        # 4. 扩展：整车状态为极大/极小时，其他关键字段是否随之变化（用户规则）
        # 规则：
        # - 当整车状态 >= VEHICLE_STATUS_MAX（170以上）时：
        #     充放电枪连接指示灯、动力防盗允许READY标志位、快充唤醒信号、整车禁止READY标志位 应为 1
        #     直流充电枪连接状态、PDCU输出快充唤醒信号状态 应为 2
        # - 当整车状态 <= VEHICLE_STATUS_MIN（30以下）时：
        #     上述四个应为 0；直流充电枪连接状态、PDCU输出快充唤醒信号状态 应为 1
        vs = actual_values.get('整车状态')
        if vs is not None:
            # 阈值按“以上/以下”解读，不使用容差
            high = vs >= VEHICLE_STATUS_MAX
            low = vs <= VEHICLE_STATUS_MIN
            if high or low:
                follow_mismatch = []
                follow_mismatch_by_field = {}
                should_be_1_fields = ['充放电枪连接指示灯', '动力防盗允许READY标志位', '快充唤醒信号', '整车禁止READY标志位']
                specific_fields = {
                    '直流充电枪连接状态': 2 if high else 1,
                    'PDCU输出快充唤醒信号状态': 2 if high else 1
                }
                # 检查四个布尔/标志位
                expected_flag_val = 1 if high else 0
                for name in should_be_1_fields:
                    actual_val = actual_values.get(name)
                    if actual_val is None:
                        continue  # 缺失则跳过，不判异常
                    if actual_val != expected_flag_val:
                        state_desc = f"极大值(>= {VEHICLE_STATUS_MAX})" if high else f"极小值(<= {VEHICLE_STATUS_MIN})"
                        msg = f"整车状态={vs}处于{state_desc}时，{name}={actual_val}(应为{expected_flag_val})"
                        follow_mismatch.append(msg)
                        follow_mismatch_by_field.setdefault(name, []).append(msg)
                # 检查两个状态枚举
                for name, expected_val in specific_fields.items():
                    actual_val = actual_values.get(name)
                    if actual_val is None:
                        continue
                    if actual_val != expected_val:
                        state_desc = f"极大值(>= {VEHICLE_STATUS_MAX})" if high else f"极小值(<= {VEHICLE_STATUS_MIN})"
                        msg = f"整车状态={vs}处于{state_desc}时，{name}={actual_val}(应为{expected_val})"
                        follow_mismatch.append(msg)
                        follow_mismatch_by_field.setdefault(name, []).append(msg)
                if follow_mismatch:
                    result['is_abnormal'] = True
                    if result['anomaly_type'] == 'normal':
                        result['anomaly_type'] = 'state_follow_mismatch'
                    else:
                        result['anomaly_type'] += '+state_follow_mismatch'
                    result['state_follow_mismatch'] = follow_mismatch
                    result['state_follow_mismatch_by_field'] = follow_mismatch_by_field
                    result['mismatch_fields'] = sorted(list(follow_mismatch_by_field.keys()))
        
        return result
    
    def is_abnormal(self, output_fields: Dict) -> bool:
        """
        判断数据是否异常（兼容旧接口）
        异常条件：
        1. 整车状态为极小值（30）时，动力防盗允许READY标志位应该为0，若为1即错误
        2. 整车状态为极大值（170）时，动力防盗允许READY标志位应该为1，若为0即错误
        
        Args:
            output_fields: 输出字段字典
            
        Returns:
            True 表示异常，False 表示正常
        """
        ready_flag = output_fields.get('动力防盗允许READY标志位')
        vehicle_status = output_fields.get('整车状态')
        
        if ready_flag is None or vehicle_status is None:
            return False
        
        # 导入配置（延迟导入避免循环依赖）
        try:
            try:
                from configs.config_vcu import VEHICLE_STATUS_MIN, VEHICLE_STATUS_MAX, VEHICLE_STATUS_TOLERANCE
            except ImportError:
                from config_vcu import VEHICLE_STATUS_MIN, VEHICLE_STATUS_MAX, VEHICLE_STATUS_TOLERANCE
        except ImportError:
            # 默认值
            VEHICLE_STATUS_MIN = 30
            VEHICLE_STATUS_MAX = 170
            VEHICLE_STATUS_TOLERANCE = 5
        
        # 异常情况1：整车状态接近极小值（30）时，READY标志位应该为0，若为1即错误
        if abs(vehicle_status - VEHICLE_STATUS_MIN) <= VEHICLE_STATUS_TOLERANCE:
            if ready_flag == 1:
                return True  # 异常：极小值状态时READY标志位不应该为1
        
        # 异常情况2：整车状态接近极大值（170）时，READY标志位应该为1，若为0即错误
        if abs(vehicle_status - VEHICLE_STATUS_MAX) <= VEHICLE_STATUS_TOLERANCE:
            if ready_flag == 0:
                return True  # 异常：极大值状态时READY标志位不应该为0
            
        return False
    
    def load_test_data(self, limit: Optional[int] = None) -> List[Dict]:
        """
        从数据库加载测试数据
        
        Args:
            limit: 限制加载的数据条数，None 表示加载全部
            
        Returns:
            测试数据列表，每个元素包含：
            - run_id: 运行ID
            - round_id: 轮次ID
            - cc2_voltage: CC2电压值
            - output_fields: 输出字段
            - is_abnormal: 是否异常
            - anomaly_info: 详细异常信息（包含异常类型、不符合的字段等）
            - status: 状态码
        """
        if not self.conn:
            self.connect()
            
        query = """
            SELECT run_id, round_id, actual_input, actual_output, 
                   expected_output, expected_error_output, expected_stuck_output, status
            FROM test_runs
            WHERE actual_input IS NOT NULL AND actual_output IS NOT NULL
            ORDER BY run_id
        """
        if limit:
            query += f" LIMIT {limit}"
            
        cursor = self.conn.execute(query)
        rows = cursor.fetchall()
        
        data_list = []
        for row in rows:
            cc2_voltage = self.extract_cc2_voltage(row['actual_input'])
            if cc2_voltage is None:
                continue
                
            output_fields = self.extract_output_fields(row['actual_output'])
            
            # 使用新的异常检测方法
            anomaly_info = self.detect_anomalies(
                row['actual_output'],
                row['expected_output'],
                row['expected_error_output'],
                row['expected_stuck_output']
            )
            
            data_list.append({
                'run_id': row['run_id'],
                'round_id': row['round_id'],
                'cc2_voltage': cc2_voltage,
                'output_fields': output_fields,
                'is_abnormal': anomaly_info['is_abnormal'],
                'anomaly_info': anomaly_info,
                'status': row['status']
            })
            
        return data_list
    
    def load_sequences_by_round(self, round_id: Optional[int] = None) -> List[List[Dict]]:
        """
        按轮次加载 CC2 电压序列
        每个序列包含一次唤醒和一次休眠（交替进行）
        
        Args:
            round_id: 指定轮次ID，None 表示加载所有轮次
            
        Returns:
            序列列表，每个序列是一个列表，包含该轮次的所有测试数据
        """
        if not self.conn:
            self.connect()
            
        query = """
            SELECT run_id, round_id, actual_input, actual_output,
                   expected_output, expected_error_output, expected_stuck_output, status
            FROM test_runs
            WHERE actual_input IS NOT NULL AND actual_output IS NOT NULL
        """
        
        if round_id is not None:
            query += f" AND round_id = {round_id}"
            
        query += " ORDER BY round_id, run_id"
        
        cursor = self.conn.execute(query)
        rows = cursor.fetchall()
        
        # 按 round_id 分组
        sequences = {}
        for row in rows:
            cc2_voltage = self.extract_cc2_voltage(row['actual_input'])
            if cc2_voltage is None:
                continue
                
            output_fields = self.extract_output_fields(row['actual_output'])
            
            # 使用新的异常检测方法
            anomaly_info = self.detect_anomalies(
                row['actual_output'],
                row['expected_output'],
                row['expected_error_output'],
                row['expected_stuck_output']
            )
            
            data_item = {
                'run_id': row['run_id'],
                'round_id': row['round_id'],
                'cc2_voltage': cc2_voltage,
                'output_fields': output_fields,
                'is_abnormal': anomaly_info['is_abnormal'],
                'anomaly_info': anomaly_info,
                'status': row['status']
            }
            
            rid = row['round_id']
            if rid not in sequences:
                sequences[rid] = []
            sequences[rid].append(data_item)
            
        return list(sequences.values())
    
    def get_statistics(self) -> Dict:
        """
        获取数据统计信息
        
        Returns:
            统计信息字典
        """
        if not self.conn:
            self.connect()
            
        # 总数据量
        total = self.conn.execute("SELECT COUNT(*) FROM test_runs").fetchone()[0]
        
        # 加载数据进行分析
        data_list = self.load_test_data()
        
        total_valid = len(data_list)
        abnormal_count = sum(1 for d in data_list if d['is_abnormal'])
        normal_count = total_valid - abnormal_count
        
        # CC2 电压范围统计
        voltages = [d['cc2_voltage'] for d in data_list]
        if voltages:
            voltage_stats = {
                'min': min(voltages),
                'max': max(voltages),
                'mean': np.mean(voltages),
                'std': np.std(voltages)
            }
        else:
            voltage_stats = {}
            
        return {
            'total_records': total,
            'valid_records': total_valid,
            'abnormal_count': abnormal_count,
            'normal_count': normal_count,
            'abnormal_rate': abnormal_count / total_valid if total_valid > 0 else 0,
            'voltage_stats': voltage_stats
        }

