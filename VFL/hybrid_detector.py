#!/usr/bin/env python3
"""
修复方案：添加基于规则的辅助检测
因为特征分布差异,纯ML模型效果不佳，添加规则引擎辅助检测
"""

import numpy as np
from collections import Counter

class HybridAttackDetector:
    """混合攻击检测器 - 结合机器学习和规则引擎"""
    
    def __init__(self, ml_classifier, flow_tracker):
        self.ml_classifier = ml_classifier
        self.flow_tracker = flow_tracker
        
        # 检测阈值（调整后避免误判）
        self.dos_threshold = {
            'same_dst_count': 40,   # 进一步提高，避免误判浏览器并发连接
            'serror_rate': 0.85,    # 提高SYN错误率阈值
            'syn_rate': 0.9         # SYN包占比
        }
        
        self.probe_threshold = {
            'port_diversity': 15,    # 必须扫描至少15个不同端口
            'same_dst_count': 100,   # 大幅提高至100个连接
            'diff_srv_rate': 0.9,    # 提高到0.9，几乎所有连接都是不同服务
            'min_ports': 12,         # 最少扫描12个端口
            'max_time_window': 10.0  # 时间窗口内（秒）
        }
        
        self.r2l_threshold = {
            'failed_login_ports': [21, 22, 23, 3389],  # 常见登录服务
            'same_srv_count': 3,   # 同服务重复尝试
            'large_payload': 80    # 较大payload
        }
        
        # 正常流量特征（用于优先识别）
        self.normal_indicators = {
            'common_ports': [80, 443, 53, 8080, 8443, 8000, 9000],  # 常见正常服务端口
            'max_same_dst': 80,     # 大幅提高正常流量连接数上限（浏览器可能并发很多请求）
            'typical_flags': [0x02, 0x10, 0x18],  # SYN, ACK, PSH+ACK
            'min_error_for_attack': 0.6  # 攻击通常有更高的错误率
        }
    
    def detect(self, base_features, packet_info, flow_stats):
        """
        混合检测
        
        Returns:
            (attack_type, confidence, method)
        """
        # 1. 基于规则的快速检测
        rule_result = self._rule_based_detection(base_features, packet_info, flow_stats)
        
        if rule_result:
            attack_type, confidence = rule_result
            return attack_type, confidence, 'rule'
        
        # 2. 机器学习检测（作为后备）
        enhanced_features = self.flow_tracker.features_to_vector(flow_stats, base_features)
        processed = self.ml_classifier.preprocess_flow(enhanced_features)
        ml_pred, ml_conf, _ = self.ml_classifier.classify(processed)

        # 对 ML 输出的 probe 结果做额外校验，避免将正常流量误判为探测
        if ml_pred == 'probe':
            protocol = packet_info.get('protocol', 0)
            dst_port = packet_info.get('dst_port', 0)
            same_dst = flow_stats.get('same_dst_count', 0)
            diff_srv_rate = flow_stats.get('diff_srv_rate', 0)
            serror_rate = flow_stats.get('serror_rate', 0)

            # 端口扫描的一般模式：
            # - 短时间内同一目标有较多连接（same_dst 较高）
            # - 访问大量不同服务/端口（diff_srv_rate 高）
            # - 错误率较高（serror_rate 高）
            # - 目的端口通常不是典型的 Web / DNS 正常端口
            is_scan_like = (
                protocol == 6 and  # 仅对 TCP 流量做这一判断
                same_dst >= 20 and
                diff_srv_rate >= 0.7 and
                serror_rate >= 0.5 and
                dst_port not in self.normal_indicators['common_ports']
            )

            # 如果不符合扫描模式，则将 probe 判定降级为 normal，
            # 以牺牲部分 Probe 召回为代价，降低正常流量被误报为 probe 的概率
            if not is_scan_like:
                return 'normal', min(0.8, max(ml_conf, 0.6)), 'ml'

        return ml_pred, ml_conf, 'ml'
    
    def _rule_based_detection(self, base_features, packet_info, flow_stats):
        """基于规则的检测（优先识别正常流量，避免误判）"""
        
        # 提取关键特征
        protocol = packet_info.get('protocol', 0)
        dst_port = packet_info.get('dst_port', 0)
        tcp_flags = packet_info.get('tcp_flags', 0)
        packet_size = packet_info.get('packet_size', 0)
        
        same_dst = flow_stats.get('same_dst_count', 0)
        serror_rate = flow_stats.get('serror_rate', 0)
        same_srv_count = flow_stats.get('same_srv_count', 0)
        same_srv_rate = flow_stats.get('same_srv_rate', 0)
        diff_srv_rate = flow_stats.get('diff_srv_rate', 0)
        syn_count = flow_stats.get('syn_count', 0)
        
        # 【优先级0: 识别正常流量】避免误判
        # 正常HTTP/HTTPS流量特征：
        # - 访问常见端口（80, 443等）
        # - 连接数在合理范围内
        # - 主要访问同一服务（same_srv_rate高）
        # - 错误率低（非扫描/攻击行为）
        if protocol == 6:  # TCP
            is_common_port = dst_port in self.normal_indicators['common_ports']
            reasonable_conn = same_dst < self.normal_indicators['max_same_dst']
            same_service = same_srv_rate > 0.5  # 降低阈值，允许更多变化
            low_error = serror_rate < self.normal_indicators['min_error_for_attack']
            
            # 条件1：访问常见端口 + 合理连接数 + 低错误率 → 很可能是正常流量
            if is_common_port and reasonable_conn and low_error:
                return 'normal', 0.90
            
            # 条件2：合理连接数 + 主要访问同一服务 + 低错误率 → 也可能是正常流量
            if reasonable_conn and same_service and low_error:
                return 'normal', 0.85
            
            # 条件3：即使连接较多，但主要访问同一服务且错误率低 → 也是正常
            if same_dst < 150 and same_service and low_error:
                return 'normal', 0.80
        
        # UDP DNS查询等正常流量
        if protocol == 17 and dst_port == 53 and same_dst < 50:
            return 'normal', 0.85
        
        # 【规则1: DoS攻击检测 (SYN Flood)】
        if protocol == 6 and tcp_flags & 0x02:  # TCP SYN
            # 计算一个简单的 SYN 占比估计（在当前目标上的 SYN-only 连接比例）
            syn_only_ratio = serror_rate  # 在 FlowTracker 中 serror_rate 已近似为"只有 SYN 的连接占比"

            # 强条件：大量连接 + 非常高的错误率 → 明确判定为 DoS
            if same_dst >= self.dos_threshold['same_dst_count'] and serror_rate >= self.dos_threshold['serror_rate']:
                confidence = min(0.9 + serror_rate * 0.1, 1.0)
                return 'dos', confidence

            # 次强条件：连接数非常多(>=100) 且 错误率中等偏高(>=0.6)，
            # 在真实环境下这类模式也极少是正常流量，直接归为 DoS，
            # 避免被 ML 判成 normal。
            if same_dst >= 100 and serror_rate >= 0.6:
                confidence = 0.85 + 0.1 * min(1.0, syn_only_ratio)
                return 'dos', min(confidence, 0.99)
        
        # 【规则2: R2L攻击检测 (暴力破解)】
        if protocol == 6 and dst_port in self.r2l_threshold['failed_login_ports']:
            # 检查是否有payload（PSH flag）且不是纯SYN
            has_payload = (tcp_flags & 0x08) or packet_size >= self.r2l_threshold['large_payload']
            if same_dst >= 2 and has_payload:  # 至少2次尝试
                confidence = 0.75 + min(same_dst * 0.05, 0.2)  # 随尝试次数增加
                return 'r2l', confidence
        
        # 【规则3: 端口扫描检测 (Probe)】- 极其严格的条件
        # 必须满足ALL条件才认定为扫描（避免误判正常流量）
        if protocol == 6 and tcp_flags & 0x02:  # TCP SYN
            # 条件1: 非常大量的连接到同一目标
            very_many_connections = same_dst >= self.probe_threshold['same_dst_count']
            # 条件2: 访问非常多不同服务/端口（几乎每个连接都是不同端口）
            very_diverse_services = diff_srv_rate >= self.probe_threshold['diff_srv_rate']
            # 条件3: 高错误率（大部分连接失败/无响应）
            high_error = serror_rate > 0.7
            # 条件4: 不是访问常见服务端口
            not_common_service = dst_port not in self.normal_indicators['common_ports']
            
            # 必须同时满足所有条件
            if very_many_connections and very_diverse_services and high_error and not_common_service:
                confidence = 0.88 + diff_srv_rate * 0.1
                return 'probe', confidence
        
        # 【规则4: ICMP Flood】
        if protocol == 1:  # ICMP
            if same_dst >= 50:  # 提高阈值
                return 'dos', 0.85
        
        # 【规则5: UDP Flood】
        if protocol == 17:  # UDP
            # 排除DNS（端口53）
            if dst_port != 53 and same_dst >= 50:
                return 'dos', 0.80
        
        # 没有匹配任何攻击规则：
        # - 如果连接数很少，大概率是正常流量
        #   （注意：上面已经对高 same_dst 的情况做了 DoS / Probe 判定，
        #    这里不要再把高并发场景误降级为 normal）
        if same_dst < 5:
            return 'normal', 0.70
        
        # 其他情况返回None，让ML模型接管
        return None


# 测试代码
if __name__ == '__main__':
    print("="*80)
    print("混合攻击检测器测试")
    print("="*80)
    
    from realtime_monitor import VFLFlowClassifier
    from flow_tracker import FlowTracker
    import torch
    import time
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    classifier = VFLFlowClassifier('models/vfl_network', device)
    tracker = FlowTracker()
    detector = HybridAttackDetector(classifier, tracker)
    
    # 测试1: SYN Flood
    print("\n【测试1: SYN Flood】")
    for i in range(50):
        packet = {
            'src_ip': f'10.0.0.{i}',
            'dst_ip': '192.168.1.1',
            'src_port': 10000 + i,
            'dst_port': 80,
            'protocol': 6,
            'tcp_flags': 0x02,
            'packet_size': 60,
            'ttl': 64,
            'timestamp': time.time() + i * 0.001
        }
        
        base_features = np.zeros(41)
        base_features[0] = packet['packet_size']
        base_features[1] = packet['protocol']
        base_features[2] = packet['src_port']
        base_features[3] = packet['dst_port']
        base_features[4] = packet['tcp_flags']
        
        flow_stats = tracker.update(packet)
        attack_type, confidence, method = detector.detect(base_features, packet, flow_stats)
        
        if i % 10 == 0:
            print(f"  包{i+1:2d}: {attack_type:8s} (置信度={confidence:.3f}, 方法={method})")
    
    # 重置
    tracker = FlowTracker()
    detector = HybridAttackDetector(classifier, tracker)
    
    # 测试2: R2L攻击
    print("\n【测试2: FTP暴力破解 (R2L)】")
    for i in range(20):
        packet = {
            'src_ip': '10.0.0.1',
            'dst_ip': '192.168.1.100',
            'src_port': 40000 + i,
            'dst_port': 21,
            'protocol': 6,
            'tcp_flags': 0x18,
            'packet_size': 200,
            'ttl': 64,
            'timestamp': time.time() + i * 1.0
        }
        
        base_features = np.zeros(41)
        base_features[0] = packet['packet_size']
        base_features[1] = packet['protocol']
        base_features[2] = packet['src_port']
        base_features[3] = packet['dst_port']
        base_features[4] = packet['tcp_flags']
        
        flow_stats = tracker.update(packet)
        attack_type, confidence, method = detector.detect(base_features, packet, flow_stats)
        
        if i % 5 == 0:
            print(f"  包{i+1:2d}: {attack_type:8s} (置信度={confidence:.3f}, 方法={method})")
    
    # 重置
    tracker = FlowTracker()
    detector = HybridAttackDetector(classifier, tracker)
    
    # 测试3: 端口扫描
    print("\n【测试3: 端口扫描 (Probe)】")
    ports = [21, 22, 23, 25, 80, 110, 443, 3306, 8080, 5432] * 2
    for i, port in enumerate(ports):
        packet = {
            'src_ip': '10.0.0.1',
            'dst_ip': '192.168.1.1',
            'src_port': 55000,
            'dst_port': port,
            'protocol': 6,
            'tcp_flags': 0x02,
            'packet_size': 60,
            'ttl': 64,
            'timestamp': time.time() + i * 0.1
        }
        
        base_features = np.zeros(41)
        base_features[0] = packet['packet_size']
        base_features[1] = packet['protocol']
        base_features[2] = packet['src_port']
        base_features[3] = packet['dst_port']
        base_features[4] = packet['tcp_flags']
        
        flow_stats = tracker.update(packet)
        attack_type, confidence, method = detector.detect(base_features, packet, flow_stats)
        
        if i % 5 == 0:
            print(f"  包{i+1:2d}: {attack_type:8s} (置信度={confidence:.3f}, 方法={method}, port={port})")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("✓ 混合检测器结合了规则引擎和机器学习")
    print("✓ 对于特征明显的攻击，使用规则快速检测")
    print("✓ 对于复杂模式，使用ML模型")
