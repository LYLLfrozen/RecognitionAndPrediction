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
        
        # 检测阈值（调整后提高DoS检测灵敏度）
        self.dos_threshold = {
            'same_dst_count': 20,   # 降低阈值以便更早检测到SYN flood
            'serror_rate': 0.5,     # 降低SYN错误率阈值
            'syn_rate': 0.7,        # SYN包占比
            'min_syn_count': 10,    # 最少SYN包数量
            'small_packet_size': 100  # SYN包通常很小
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
        混合检测（ML优先策略）

        Returns:
            (attack_type, confidence, method)
        """
        # 1. 优先使用机器学习检测（主要检测方式）
        enhanced_features = self.flow_tracker.features_to_vector(flow_stats, base_features)
        processed = self.ml_classifier.preprocess_flow(enhanced_features)
        ml_pred, ml_conf, _ = self.ml_classifier.classify(processed)

        # 2. 使用规则对ML结果进行验证和调整
        rule_result = self._rule_based_verification(base_features, packet_info, flow_stats, ml_pred, ml_conf)

        if rule_result:
            # 规则检测到强特征，覆盖ML结果
            attack_type, confidence = rule_result
            return attack_type, confidence, 'rule'

        # 3. 对ML结果进行后处理
        # 对 ML 输出的 probe 结果做额外校验，避免将正常流量误判为探测
        if ml_pred == 'probe':
            protocol = packet_info.get('protocol', 0)
            dst_port = packet_info.get('dst_port', 0)
            same_dst = flow_stats.get('same_dst_count', 0)
            diff_srv_rate = flow_stats.get('diff_srv_rate', 0)
            serror_rate = flow_stats.get('serror_rate', 0)

            # 端口扫描的一般模式
            is_scan_like = (
                protocol == 6 and
                same_dst >= 15 and  # 降低阈值，让更多流量被ML识别
                diff_srv_rate >= 0.6 and  # 降低阈值
                serror_rate >= 0.4 and  # 降低阈值
                dst_port not in self.normal_indicators['common_ports']
            )

            # 如果不符合扫描模式，降级为normal
            if not is_scan_like:
                return 'normal', min(0.8, max(ml_conf, 0.6)), 'ml'

        return ml_pred, ml_conf, 'ml'
    
    def _rule_based_verification(self, base_features, packet_info, flow_stats, ml_pred, ml_conf):
        """基于规则的验证（改进的DoS检测）"""

        # 提取关键特征
        protocol = packet_info.get('protocol', 0)
        dst_port = packet_info.get('dst_port', 0)
        tcp_flags = packet_info.get('tcp_flags', 0)
        packet_size = packet_info.get('packet_size', 0)

        same_dst = flow_stats.get('same_dst_count', 0)
        serror_rate = flow_stats.get('serror_rate', 0)
        diff_srv_rate = flow_stats.get('diff_srv_rate', 0)
        syn_count = flow_stats.get('syn_count', 0)
        psh_count = flow_stats.get('psh_count', 0)
        fin_count = flow_stats.get('fin_count', 0)

        # 【规则1: SYN Flood DoS攻击检测 - 改进版】
        if protocol == 6:  # TCP
            is_syn = tcp_flags & 0x02  # 当前包是SYN
            is_small_packet = packet_size <= self.dos_threshold['small_packet_size']
            
            # 条件1: 高频SYN包 + 无数据传输（经典SYN flood）
            # serror_rate表示"只有SYN但没有PSH/FIN的连接"占比
            if is_syn:
                # 大量相同目标 + 较高错误率（半连接）
                if same_dst >= self.dos_threshold['same_dst_count'] and serror_rate >= self.dos_threshold['serror_rate']:
                    confidence = min(0.85 + serror_rate * 0.1 + same_dst * 0.001, 0.98)
                    return 'dos', confidence
                
                # 中等数量的SYN到同一目标 + 高错误率（更敏感的检测）
                if same_dst >= 10 and serror_rate >= 0.7:
                    confidence = min(0.80 + serror_rate * 0.1, 0.92)
                    return 'dos', confidence
            
            # 条件2: 小包高频（即使不是只有SYN也可能是flood）
            if is_small_packet and same_dst >= 30 and serror_rate >= 0.6:
                confidence = 0.82 + serror_rate * 0.1
                return 'dos', confidence

        # 【规则2: 明显的暴力破解】
        if protocol == 6 and dst_port in self.r2l_threshold['failed_login_ports']:
            has_payload = (tcp_flags & 0x08) or packet_size >= self.r2l_threshold['large_payload']
            # 提高阈值，只在非常明显时触发
            if same_dst >= 5 and has_payload:
                confidence = 0.80 + min(same_dst * 0.03, 0.15)
                return 'r2l', confidence

        # 【规则3: 极明显的端口扫描】
        if protocol == 6 and tcp_flags & 0x02:
            # 必须满足极严格的条件
            if (same_dst >= 200 and
                diff_srv_rate >= 0.95 and
                serror_rate > 0.85 and
                dst_port not in self.normal_indicators['common_ports']):
                confidence = 0.92 + diff_srv_rate * 0.05
                return 'probe', confidence

        # 【强规则4: ICMP/UDP Flood】
        if protocol == 1 and same_dst >= 100:  # ICMP
            return 'dos', 0.90

        if protocol == 17 and dst_port != 53 and same_dst >= 100:  # UDP非DNS
            return 'dos', 0.88

        # 其他情况：信任ML的判断
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
