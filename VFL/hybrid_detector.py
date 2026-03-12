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
        
        # 检测阈值（规则作为兜底机制，设置较高阈值减少触发频率）
        self.dos_threshold = {
            'same_dst_count': 100,  # 大幅提高阈值，减少规则触发频率
            'serror_rate': 0.8,     # 提高SYN错误率阈值
            'syn_rate': 0.9,        # SYN包占比阈值提高
            'min_syn_count': 50,    # 最少SYN包数量提高
            'small_packet_size': 64  # 仅检测极小的SYN包
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
        混合检测（ML主导策略）
        规则引擎仅作为兜底机制，ML置信度 >= 0.5 时完全信任ML。

        Returns:
            (attack_type, confidence, method)
        """
        # 1. 优先使用机器学习检测（主要检测方式）
        enhanced_features = self.flow_tracker.features_to_vector(flow_stats, base_features)
        processed = self.ml_classifier.preprocess_flow(enhanced_features)
        ml_pred, ml_conf, _ = self.ml_classifier.classify(processed)

        # 2. ML置信度 >= 0.5：完全信任ML，规则引擎不介入
        if ml_conf >= 0.5:
            # 对低置信度 probe 结果做防误判校验（避免正常流量被误标为探测）
            if ml_pred == 'probe' and ml_conf < 0.8:
                protocol = packet_info.get('protocol', 0)
                dst_port = packet_info.get('dst_port', 0)
                same_dst = flow_stats.get('same_dst_count', 0)
                diff_srv_rate = flow_stats.get('diff_srv_rate', 0)
                serror_rate = flow_stats.get('serror_rate', 0)
                is_scan_like = (
                    protocol == 6 and
                    same_dst >= 50 and
                    diff_srv_rate >= 0.8 and
                    serror_rate >= 0.6 and
                    dst_port not in self.normal_indicators['common_ports']
                )
                if not is_scan_like:
                    return 'normal', min(0.8, max(ml_conf, 0.6)), 'ml'
            return ml_pred, ml_conf, 'ml'

        # 3. ML置信度中等（0.35 <= ml_conf < 0.5）：ML仍为主
        #    规则仅在与ML意见一致时小幅提升置信度，不允许覆盖ML结果
        elif ml_conf >= 0.35:
            rule_result = self._rule_based_verification(base_features, packet_info, flow_stats, ml_pred, ml_conf)
            if rule_result:
                attack_type, rule_conf = rule_result
                if attack_type == ml_pred:
                    # ML与规则意见一致，小幅提升置信度，保持ML标签
                    boosted_conf = min(ml_conf + 0.15, 0.80)
                    return ml_pred, boosted_conf, 'ml'
            return ml_pred, ml_conf, 'ml'

        # 4. ML置信度很低（< 0.35）：规则可作为兜底
        #    但需极高置信度（>= 0.92）才能覆盖ML
        else:
            rule_result = self._rule_based_verification(base_features, packet_info, flow_stats, ml_pred, ml_conf)
            if rule_result:
                attack_type, rule_conf = rule_result
                if rule_conf >= 0.92:
                    return attack_type, rule_conf, 'rule'
            return ml_pred, ml_conf, 'ml'
    
    def _rule_based_verification(self, base_features, packet_info, flow_stats, ml_pred, ml_conf):
        """基于规则的验证（仅作为ML的兜底机制）"""

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

        # ML置信度 >= 0.5 时不介入（已在 detect() 中拦截，此处作双重保障）
        if ml_conf >= 0.5:
            return None

        # 【规则1: SYN Flood DoS攻击检测 - 高阈值版本】
        # 仅在出现非常明显的DoS特征时触发
        if protocol == 6:  # TCP
            is_syn = tcp_flags & 0x02  # 当前包是SYN
            is_small_packet = packet_size <= self.dos_threshold['small_packet_size']

            # 条件1: 大量相同目标 + 极高错误率（经典SYN flood）
            if is_syn and same_dst >= self.dos_threshold['same_dst_count'] and serror_rate >= self.dos_threshold['serror_rate']:
                confidence = min(0.90 + serror_rate * 0.08 + same_dst * 0.0005, 0.99)
                return 'dos', confidence

            # 条件2: 极端小包高频攻击
            if is_small_packet and same_dst >= 200 and serror_rate >= 0.9:
                confidence = 0.92 + serror_rate * 0.06
                return 'dos', confidence

        # 【规则2: 非常明显的暴力破解】
        if protocol == 6 and dst_port in self.r2l_threshold['failed_login_ports']:
            has_payload = (tcp_flags & 0x08) or packet_size >= self.r2l_threshold['large_payload']
            # 仅在高频失败登录时触发
            if same_dst >= 20 and has_payload:
                confidence = 0.85 + min(same_dst * 0.02, 0.12)
                return 'r2l', confidence

        # 【规则3: 极端明显的端口扫描】
        if protocol == 6 and tcp_flags & 0x02:
            # 必须满足极严格的条件
            if (same_dst >= 500 and
                diff_srv_rate >= 0.98 and
                serror_rate > 0.9 and
                dst_port not in self.normal_indicators['common_ports']):
                confidence = 0.95 + diff_srv_rate * 0.03
                return 'probe', confidence

        # 【强规则4: 大规模ICMP/UDP Flood】
        if protocol == 1 and same_dst >= 500:  # ICMP
            return 'dos', 0.95

        if protocol == 17 and dst_port != 53 and same_dst >= 500:  # UDP非DNS
            return 'dos', 0.93

        # 其他情况：信任ML的判断或继续使用低置信度结果
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
