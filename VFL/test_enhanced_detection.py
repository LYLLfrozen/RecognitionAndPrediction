#!/usr/bin/env python3
"""
综合测试脚本 - 验证增强特征提取后的攻击检测能力
"""

import numpy as np
import time
from realtime_monitor import VFLFlowClassifier
from flow_tracker import FlowTracker
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classifier = VFLFlowClassifier('models/vfl_network', device)
tracker = FlowTracker()

print("="*80)
print("增强特征攻击检测测试")
print("="*80)

def test_attack(attack_name, packets):
    """测试一组攻击包"""
    print(f"\n【{attack_name}】")
    predictions = []
    
    for i, packet in enumerate(packets):
        # 提取基础特征
        features = np.zeros(41)
        features[0] = packet.get('packet_size', 60)
        features[1] = packet['protocol']
        features[2] = packet.get('src_port', 0)
        features[3] = packet.get('dst_port', 0)
        features[4] = packet.get('tcp_flags', 0)
        features[5] = packet.get('ttl', 64)
        features[6] = packet.get('packet_size', 60)
        
        # 更新流跟踪器
        flow_stats = tracker.update(packet)
        
        # 合并统计特征
        enhanced = tracker.features_to_vector(flow_stats, features)
        
        # 分类
        processed = classifier.preprocess_flow(enhanced)
        pred_class, confidence, probs = classifier.classify(processed)
        
        predictions.append(pred_class)
        
        if i % 10 == 0:
            print(f"  包{i+1}: 预测={pred_class:8s}, 置信度={confidence:.3f}, same_dst={flow_stats.get('same_dst_count', 0)}, serror_rate={flow_stats.get('serror_rate', 0):.2f}")
    
    # 统计结果
    from collections import Counter
    pred_counter = Counter(predictions)
    print(f"\n  预测分布:")
    for pred, count in pred_counter.most_common():
        print(f"    {pred:8s}: {count}/{len(predictions)} ({count/len(predictions)*100:.1f}%)")
    
    return predictions


# 测试1: SYN Flood (DoS攻击)
print("\n" + "="*80)
dos_packets = []
for i in range(100):
    dos_packets.append({
        'src_ip': f'10.0.{i//256}.{i%256}',
        'dst_ip': '192.168.1.1',
        'src_port': 10000 + i,
        'dst_port': 80,
        'protocol': 6,
        'tcp_flags': 0x02,  # SYN
        'packet_size': 60,
        'ttl': 64,
        'timestamp': time.time() + i * 0.001
    })

dos_preds = test_attack("SYN Flood (DoS攻击)", dos_packets)

# 重置tracker
tracker = FlowTracker()

# 测试2: FTP暴力破解 (R2L攻击)
print("\n" + "="*80)
r2l_packets = []
for i in range(50):
    r2l_packets.append({
        'src_ip': '10.0.0.1',
        'dst_ip': '192.168.1.100',
        'src_port': 40000 + i,
        'dst_port': 21,  # FTP
        'protocol': 6,
        'tcp_flags': 0x18,  # PSH+ACK
        'packet_size': 200,  # 较大的包（包含payload）
        'ttl': 64,
        'timestamp': time.time() + i * 1.0  # 间隔1秒
    })

r2l_preds = test_attack("FTP暴力破解 (R2L攻击)", r2l_packets)

# 重置tracker
tracker = FlowTracker()

# 测试3: 端口扫描 (Probe攻击)
print("\n" + "="*80)
probe_packets = []
ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 3306, 3389, 5432, 8080]
for i, port in enumerate(ports * 3):  # 扫描3轮
    probe_packets.append({
        'src_ip': '10.0.0.1',
        'dst_ip': '192.168.1.1',
        'src_port': 50000,
        'dst_port': port,
        'protocol': 6,
        'tcp_flags': 0x02,  # SYN
        'packet_size': 60,
        'ttl': 64,
        'timestamp': time.time() + i * 0.1
    })

probe_preds = test_attack("端口扫描 (Probe攻击)", probe_packets)

# 重置tracker
tracker = FlowTracker()

# 测试4: 正常HTTP流量
print("\n" + "="*80)
normal_packets = []
for i in range(20):
    normal_packets.append({
        'src_ip': '192.168.1.100',
        'dst_ip': '8.8.8.8',
        'src_port': 50000 + i,
        'dst_port': 80,
        'protocol': 6,
        'tcp_flags': 0x18,  # PSH+ACK
        'packet_size': 1000,
        'ttl': 64,
        'timestamp': time.time() + i * 0.5
    })

normal_preds = test_attack("正常HTTP流量 (Normal)", normal_packets)

# 总结
print("\n" + "="*80)
print("测试总结")
print("="*80)

from collections import Counter

def get_dominant_prediction(preds):
    counter = Counter(preds)
    return counter.most_common(1)[0][0]

results = [
    ("DoS (SYN Flood)", "dos", dos_preds),
    ("R2L (FTP暴力破解)", "r2l", r2l_preds),
    ("Probe (端口扫描)", "probe", probe_preds),
    ("Normal (正常流量)", "normal", normal_preds)
]

print(f"\n{'攻击类型':<20s} {'期望':<10s} {'实际':<10s} {'准确率':<10s}")
print("-" * 60)

for attack_name, expected, preds in results:
    dominant = get_dominant_prediction(preds)
    correct_count = sum(1 for p in preds if p == expected)
    accuracy = correct_count / len(preds) * 100
    
    status = "✓" if dominant == expected else "✗"
    print(f"{attack_name:<20s} {expected:<10s} {dominant:<10s} {accuracy:>5.1f}%  {status}")

print("\n💡 关键改进:")
print("  1. 添加了FlowTracker，计算连接级统计特征")
print("  2. 特征包括: 同目标连接数、错误率、同服务率等")
print("  3. 这些统计特征能够有效区分不同攻击模式")
print("  4. DoS: 大量SYN -> 高serror_rate")
print("  5. R2L: 同端口重复尝试 -> 高same_srv_rate + 大包")
print("  6. Probe: 多端口扫描 -> 高same_dst + 低same_srv_rate")
