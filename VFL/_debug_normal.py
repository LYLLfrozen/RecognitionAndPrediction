#!/usr/bin/env python3
"""调试：用normal.csv测试分类器，找出误判probe的原因"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '.')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

from realtime_monitor import VFLFlowClassifier
from hybrid_detector import HybridAttackDetector
from flow_tracker import FlowTracker

classifier = VFLFlowClassifier('models/vfl_network', device)
tracker = FlowTracker()
detector = HybridAttackDetector(classifier, tracker)

df = pd.read_csv('normal.csv')
print(f'列数: {len(df.columns)}, 行数: {len(df)}')
print('列名:', list(df.columns))
print()

counts = {'normal': 0, 'probe': 0, 'dos': 0, 'r2l': 0, 'u2r': 0, 'other': 0}
probe_examples = []

for i, row in df.iterrows():
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
    packet_info = {
        'src_ip': str(row.get('src_ip', '127.0.0.1')),
        'dst_ip': str(row.get('dst_ip', '127.0.0.1')),
        'src_port': int(row.get('src_port', 0)),
        'dst_port': int(row.get('dst_port', 0)),
        'protocol': int(row.get('protocol', 6)),
        'packet_size': int(row.get('packet_size', 0)),
        'tcp_flags': int(row.get('tcp_flags', 0)),
    }

    # 直接测试ML分类器（绕过混合检测器）
    enhanced = tracker.features_to_vector(flow_stats, base_features)
    processed = classifier.preprocess_flow(enhanced)
    ml_pred, ml_conf, all_probs = classifier.classify(processed)

    # 然后测试混合检测器最终结果
    final_pred, final_conf, method = detector.detect(base_features, packet_info, flow_stats)

    if ml_pred in counts:
        counts[ml_pred] += 1
    else:
        counts['other'] += 1

    if ml_pred == 'probe':
        probe_examples.append({
            'idx': i,
            'ml_pred': ml_pred, 'ml_conf': ml_conf,
            'final_pred': final_pred, 'final_conf': final_conf,
            'method': method,
            'src_port': int(row.get('src_port', 0)),
            'dst_port': int(row.get('dst_port', 0)),
            'diff_srv_rate': row.get('diff_srv_rate', 0),
            'serror_rate': row.get('serror_rate', 0),
            'same_dst_count': row.get('same_dst_count', 0),
            'all_probs': all_probs.tolist(),
            'tcp_flags': int(row.get('tcp_flags', 0)),
            'enhanced_key': enhanced[18:22].tolist(),  # serror,rerror,same_srv,diff_srv
        })

print("=== ML直接分类结果统计 ===")
for k, v in counts.items():
    print(f"  {k}: {v}")

print(f"\n=== probe误判样本 (共{len(probe_examples)}个) ===")
for ex in probe_examples[:10]:
    print(f"  行{ex['idx']:3d}: ML={ex['ml_pred']}({ex['ml_conf']:.3f}) FINAL={ex['final_pred']}({ex['final_conf']:.3f}) [{ex['method']}]")
    print(f"         src={ex['src_port']} dst={ex['dst_port']} diff_srv={ex['diff_srv_rate']:.3f} serr={ex['serror_rate']:.3f} same_dst={ex['same_dst_count']}")
    print(f"         增强特征[18:22]={ex['enhanced_key']} (serr,rerr,same_srv,diff_srv)")
    probs = ex['all_probs']
    cls_names = classifier.class_names
    prob_str = ' '.join(f"{cls_names[j]}:{probs[j]:.3f}" for j in range(len(probs)))
    print(f"         概率分布: {prob_str}")
    print()
