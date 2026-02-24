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
        
        # 检测阈值
        self.dos_threshold = {
            'same_dst_count': 20,  # 短时间内同目标连接数
            'serror_rate': 0.7,    # SYN错误率
            'syn_rate': 0.8        # SYN包占比
        }
        
        self.probe_threshold = {
            'port_diversity': 5,   # 扫描的不同端口数
            'same_dst_count': 10,  # 同目标连接数
            'diff_srv_rate': 0.7   # 不同服务率
        }
        
        self.r2l_threshold = {
            'failed_login_ports': [21, 22, 23, 3389],  # 常见登录服务
            'same_srv_count': 3,   # 同服务重复尝试（降低阈值）
            'large_payload': 80    # 较大payload（降低阈值）
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
        
        return ml_pred, ml_conf, 'ml'
    
    def _rule_based_detection(self, base_features, packet_info, flow_stats):
        """基于规则的检测"""
        
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
        
        # 规则1: DoS攻击检测 (SYN Flood)
        if protocol == 6 and tcp_flags & 0x02:  # TCP SYN
            if same_dst >= self.dos_threshold['same_dst_count']:
                if serror_rate >= self.dos_threshold['serror_rate']:
                    confidence = min(0.9 + serror_rate * 0.1, 1.0)
                    return 'dos', confidence
        
        # 规则2: R2L攻击检测 (暴力破解) - 优先级高于Probe
        if protocol == 6 and dst_port in self.r2l_threshold['failed_login_ports']:
            # 检查是否有payload（PSH flag）且不是纯SYN
            has_payload = (tcp_flags & 0x08) or packet_size >= self.r2l_threshold['large_payload']
            if same_dst >= 2 and has_payload:  # 至少2次尝试
                confidence = 0.75 + min(same_dst * 0.05, 0.2)  # 随尝试次数增加
                return 'r2l', confidence
        
        # 规则3: 端口扫描检测 (Probe)
        if protocol == 6 and tcp_flags & 0x02:  # TCP SYN
            if same_dst >= self.probe_threshold['same_dst_count']:
                if diff_srv_rate >= self.probe_threshold['diff_srv_rate']:
                    confidence = 0.85 + diff_srv_rate * 0.1
                    return 'probe', confidence
        
        # 规则4: ICMP Flood
        if protocol == 1:  # ICMP
            if same_dst >= 30:
                return 'dos', 0.85
        
        # 规则5: UDP Flood
        if protocol == 17:  # UDP
            if same_dst >= 30:
                return 'dos', 0.80
        
        # 没有匹配规则，返回None让ML接管
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
