#!/usr/bin/env python3
"""深入调试：检查特征预处理后的实际输入"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '.')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

from realtime_monitor import VFLFlowClassifier
from flow_tracker import FlowTracker

classifier = VFLFlowClassifier('models/vfl_network', device)
tracker = FlowTracker()

df = pd.read_csv('normal.csv')

print("=== 详细特征分析（前5行）===")
for i, row in df.iterrows():
    if i >= 5:
        break
    base_features = np.zeros(41)
    base_features[0] = row.get('packet_size', 0)
    base_features[1] = row.get('protocol', 0)
    base_features[2] = row.get('src_port', 0)
    base_features[3] = row.get('dst_port', 0)
    base_features[4] = row.get('tcp_flags', 0)
    base_features[5] = row.get('ttl', 0)
    base_features[6] = row.get('ip_len', 0)

    flow_stats = {
        'duration': row.get('duration', 0),
        'src_bytes': row.get('src_bytes', 0),
        'flow_count': row.get('flow_count', 1),
        'same_dst_count': row.get('same_dst_count', 0),
        'same_srv_count': row.get('same_srv_count', 0),
        'serror_rate': row.get('serror_rate', 0),
        'rerror_rate': row.get('rerror_rate', 0),
        'same_srv_rate': row.get('same_srv_rate', 1),
        'diff_srv_rate': row.get('diff_srv_rate', 0),
        'syn_count': row.get('syn_count', 0),
        'fin_count': row.get('fin_count', 0),
        'rst_count': row.get('rst_count', 0),
        'psh_count': row.get('psh_count', 0),
    }

    enhanced = tracker.features_to_vector(flow_stats, base_features)
    processed = classifier.preprocess_flow(enhanced)  # (1,1,11,11)

    print(f"\n行{i}:")
    print(f"  原始base_features(前10): {base_features[:10]}")
    print(f"  enhanced(前26): {enhanced[:26]}")
    print(f"  enhanced原始维度: {len(enhanced)}")
    
    # 检查归一化后的121维向量
    # preprocess_flow的输出是reshape之前的121维
    # 用手工复现归一化
    original_dim = enhanced.shape[-1]
    print(f"  original_dim={original_dim}")
    
    if original_dim == 41:
        normalized = np.zeros(41)
        normalized[0] = min(enhanced[0] / 1500.0, 1.0)   # packet_size
        normalized[1] = enhanced[1] / 20.0                # protocol
        normalized[2] = enhanced[2] / 65535.0             # src_port
        normalized[3] = enhanced[3] / 65535.0             # dst_port
        normalized[4] = enhanced[4] / 255.0               # tcp_flags
        normalized[5] = enhanced[5] / 255.0               # ttl
        normalized[6] = min(enhanced[6] / 1500.0, 1.0)    # ip_len
        normalized[7:13] = enhanced[7:13]
        if enhanced.shape[0] > 13:
            normalized[13:] = enhanced[13:]
        
        # 扩展到121维
        full = np.concatenate([normalized, np.zeros(121 - 41)])
        print(f"  归一化后的关键特征(13-26): {full[13:26]}")
        print(f"  查看是否有异常大的值: max={full.max():.4f} min={full.min():.4f}")
    
    md_pred, md_conf, all_probs = classifier.classify(processed)
    cls_names = classifier.class_names
    prob_str = ' '.join(f"{cls_names[j]}:{all_probs[j]:.3f}" for j in range(len(all_probs)))
    print(f"  预测: {md_pred}({md_conf:.3f})")
    print(f"  概率: {prob_str}")

# 对比：用全零特征测试
print("\n=== 全零特征测试 ===")
zero_feat = np.zeros(41)
processed_zero = classifier.preprocess_flow(zero_feat)
pred, conf, probs = classifier.classify(processed_zero)
print(f"全零: {pred} ({conf:.3f})")
cls_names = classifier.class_names
print(' '.join(f"{cls_names[j]}:{probs[j]:.3f}" for j in range(len(probs))))

# 测试：用模拟正常tcp流量（PSH+ACK, 普通端口）
print("\n=== 模拟正常TCP(PSH|ACK, port=443) ===")
normal_feat = np.zeros(41)
normal_feat[0] = 500   # packet_size
normal_feat[1] = 6     # TCP
normal_feat[2] = 54321 # src_port (临时端口)
normal_feat[3] = 443   # dst_port (HTTPS)
normal_feat[4] = 0x18  # PSH|ACK
normal_feat[5] = 64    # TTL
normal_feat[6] = 496   # IP len
# 流统计保持为0
processed_normal = classifier.preprocess_flow(normal_feat)
pred, conf, probs = classifier.classify(processed_normal)
print(f"模拟正常TCP: {pred} ({conf:.3f})")
print(' '.join(f"{cls_names[j]}:{probs[j]:.3f}" for j in range(len(probs))))
