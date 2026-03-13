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
            'same_dst_count': 50,   # 降低阈值以提高DoS检测灵敏度
            'serror_rate': 0.7,     # 降低SYN错误率阈值
            'syn_rate': 0.9,        # SYN包占比阈值
            'min_syn_count': 30,    # 最少SYN包数量
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
            # 常见正常服务端口（含本地代理常用端口）
            'common_ports': [
                80, 443, 53, 8080, 8443, 8000, 9000,
                10808, 7890, 7891, 1080, 1086, 1087, 1088,  # 常见本地代理端口（Clash/V2Ray等）
                2222, 2022,                                   # 非标准SSH端口
            ],
            'max_same_dst': 80,
            'typical_flags': [0x02, 0x10, 0x18],
            'min_error_for_attack': 0.6
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

        # 1.5 DoS/Probe 冲突消解：
        #     DoS通常集中攻击同一服务（diff_srv_rate低），Probe通常扫描多服务（diff_srv_rate高）。
        #     若ML把集中洪泛流量判为probe，优先修正为dos。
        resolved = self._resolve_probe_dos_conflict(ml_pred, ml_conf, packet_info, flow_stats)
        if resolved is not None:
            return resolved

        # 2. DoS强特征优先兜底：
        #    修复场景：本地SYN Flood时，ML可能高置信度误判为normal。
        #    当流量统计满足典型DoS模式时，允许规则覆盖ML。
        dos_rescue = self._check_dos_pattern(packet_info, flow_stats)
        if dos_rescue:
            dos_type, dos_conf = dos_rescue
            # 仅在 ML 未明确给出高置信度 dos 时覆盖，避免无意义抖动。
            if not (ml_pred == 'dos' and ml_conf >= 0.80):
                return dos_type, max(dos_conf, 0.80), 'rule'

        # 3. ML置信度 >= 0.5：完全信任ML，规则引擎不介入
        if ml_conf >= 0.5:
            # probe/u2r 无论置信度多高，都进行流量特征核验以防误判
            # （正常流量不应出现 diff_srv_rate>=0.6 且 serror_rate>=0.5 的组合）
            if ml_pred in ('probe', 'u2r'):
                if not self._verify_probe_u2r(ml_pred, packet_info, flow_stats):
                    # 验证失败时检查是否为DoS（防止回环DoS被误判为probe后归入normal）
                    dos_rescue = self._check_dos_pattern(packet_info, flow_stats)
                    if dos_rescue:
                        return dos_rescue[0], dos_rescue[1], 'rule'
                    return 'normal', min(0.8, max(ml_conf, 0.6)), 'ml'
            return ml_pred, ml_conf, 'ml'

        # 4. ML置信度中等（0.35 <= ml_conf < 0.5）：ML仍为主
        #    规则仅在与ML意见一致时小幅提升置信度，不允许覆盖ML结果
        elif ml_conf >= 0.35:
            # 中等置信度下 probe/u2r 更容易误判，同样需要流量验证
            if ml_pred in ('probe', 'u2r'):
                if not self._verify_probe_u2r(ml_pred, packet_info, flow_stats):
                    dos_rescue = self._check_dos_pattern(packet_info, flow_stats)
                    if dos_rescue:
                        return dos_rescue[0], dos_rescue[1], 'rule'
                    return 'normal', 0.65, 'ml'
            rule_result = self._rule_based_verification(base_features, packet_info, flow_stats, ml_pred, ml_conf)
            if rule_result:
                attack_type, rule_conf = rule_result
                if attack_type == ml_pred:
                    # ML与规则意见一致，小幅提升置信度，保持ML标签
                    boosted_conf = min(ml_conf + 0.15, 0.80)
                    return ml_pred, boosted_conf, 'ml'
            return ml_pred, ml_conf, 'ml'

        # 5. ML置信度很低（< 0.35）：规则可作为兜底
        #    但需极高置信度（>= 0.92）才能覆盖ML
        else:
            rule_result = self._rule_based_verification(base_features, packet_info, flow_stats, ml_pred, ml_conf)
            if rule_result:
                attack_type, rule_conf = rule_result
                if rule_conf >= 0.92:
                    return attack_type, rule_conf, 'rule'
            return ml_pred, ml_conf, 'ml'

    def _resolve_probe_dos_conflict(self, ml_pred, ml_conf, packet_info, flow_stats):
        """
        解决 DoS 与 Probe 的边界冲突。
        典型SYN Flood: same_dst高 + serror高 + diff_srv低（攻击集中在单端口）。
        典型Probe: same_dst高 + diff_srv高（多端口/多服务扫描）。
        """
        protocol = packet_info.get('protocol', 0)
        same_dst = flow_stats.get('same_dst_count', 0)
        serror_rate = flow_stats.get('serror_rate', 0)
        diff_srv_rate = flow_stats.get('diff_srv_rate', 0)

        if protocol != 6:
            return None

        # 仅在ML预测probe或normal时进行修正，避免覆盖稳定的dos预测
        if ml_pred not in ('probe', 'normal'):
            return None

        # 集中式洪泛（单服务）应判为DoS，不应判为Probe
        if same_dst >= 15 and serror_rate >= 0.45 and diff_srv_rate <= 0.35:
            conf = min(0.78 + 0.12 * serror_rate + 0.002 * same_dst, 0.96)
            # 当ML高置信probe但统计形态明显是DoS时，仍强制纠正
            if ml_pred == 'probe' or ml_conf < 0.80:
                return 'dos', conf, 'rule'

        return None
    
    def _verify_probe_u2r(self, ml_pred, packet_info, flow_stats):
        """
        验证 probe/u2r 预测是否有足够的流量特征支撑。
        返回 True 表示预测可信，False 表示很可能是正常流量误判。
        """
        protocol = packet_info.get('protocol', 0)
        src_port = packet_info.get('src_port', 0)
        dst_port = packet_info.get('dst_port', 0)
        # 归一化服务端口：回包时 dst_port=临时端口，需取 src_port（服务端真实端口）
        service_port = self.flow_tracker._get_service_port(src_port, dst_port)
        same_dst = flow_stats.get('same_dst_count', 0)
        diff_srv_rate = flow_stats.get('diff_srv_rate', 0)
        serror_rate = flow_stats.get('serror_rate', 0)

        if ml_pred == 'probe':
            # 端口扫描的真实特征：
            # 1. 针对非临时的固定服务端口（service_port < 32768），不会扫描随机高端口
            # 2. diff_srv_rate 高（flow_tracker 修复后此值将真实反映多目标端口）
            # 3. serror_rate 高（大多数连接被RST拒绝）—— 修复后此值不含握手中连接
            # 正常流量（修复后）：diff_srv_rate ≈ 0，serror_rate ≈ 0
            return (
                protocol == 6 and
                same_dst >= 20 and
                diff_srv_rate >= 0.6 and
                serror_rate >= 0.5 and
                service_port < 32768 and              # 真实扫描目标服务端口，非临时端口
                service_port not in self.normal_indicators['common_ports']  # 白名单放行（Problem 3）
            )

        elif ml_pred == 'u2r':
            # U2R：低频（非批量）+ 大载荷 + 针对具体服务端口
            # 正常流量不会携带超大 payload 去攻击本地提权漏洞
            packet_size = packet_info.get('packet_size', 0)
            tcp_flags = packet_info.get('tcp_flags', 0)
            src_bytes = flow_stats.get('src_bytes', 0)
            has_large_payload = packet_size > 500 or src_bytes > 10000
            is_service_port = dst_port in [21, 22, 23, 25, 80, 110, 111, 513, 514, 2049]
            is_low_freq = same_dst < 15  # u2r 通常不产生大量连接
            return is_service_port and has_large_payload and is_low_freq

        return True  # 其他类型默认信任 ML

    def _check_dos_pattern(self, packet_info, flow_stats):
        """
        当ML预测probe/u2r验证失败时，检查是否为DoS攻击。
        DoS特征：高same_dst_count + 高serror_rate + 低diff_srv_rate（集中攻击同一服务）。
        与 _verify_probe_u2r 互补：probe需要高diff_srv_rate，DoS恰好相反（低diff_srv_rate）。

        Returns: (attack_type, confidence) 或 None
        """
        protocol = packet_info.get('protocol', 0)
        same_dst = flow_stats.get('same_dst_count', 0)
        serror_rate = flow_stats.get('serror_rate', 0)
        diff_srv_rate = flow_stats.get('diff_srv_rate', 0)
        rerror_rate = flow_stats.get('rerror_rate', 0)

        # TCP DoS：大量同目标连接 + 高失败率 + 集中攻击同一服务（低diff_srv_rate）
        # 本地回环仿真和真实SYN flood都满足此条件
        if protocol == 6 and same_dst >= 20 and serror_rate >= 0.6 and diff_srv_rate <= 0.3:
            conf = min(0.70 + serror_rate * 0.15 + same_dst * 0.001, 0.95)
            return 'dos', conf

        # ICMP flood
        if protocol == 1 and same_dst >= 30:
            return 'dos', min(0.80 + same_dst * 0.001, 0.95)

        # UDP flood（非DNS）
        if protocol == 17 and same_dst >= 30 and (serror_rate + rerror_rate) >= 0.5:
            return 'dos', min(0.75 + same_dst * 0.001, 0.93)

        return None

    def _rule_based_verification(self, base_features, packet_info, flow_stats, ml_pred, ml_conf):
        """基于规则的验证（仅作为ML的兜底机制）"""

        # 提取关键特征
        protocol = packet_info.get('protocol', 0)
        src_port = packet_info.get('src_port', 0)
        dst_port = packet_info.get('dst_port', 0)
        # 归一化服务端口：回包时 dst_port=临时端口，需取 src_port（服务端真实端口）（Problem 3）
        service_port = self.flow_tracker._get_service_port(src_port, dst_port)
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
            # 必须满足极严格的条件；使用 service_port 匹配白名单（Problem 3 修复）
            if (same_dst >= 500 and
                diff_srv_rate >= 0.98 and
                serror_rate > 0.9 and
                service_port not in self.normal_indicators['common_ports']):
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
