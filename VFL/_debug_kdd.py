#!/usr/bin/env python3
"""查看所有115个KDD特征名，以及scaler的完整参数"""
import numpy as np, pickle, os, sys
sys.path.insert(0, '.')

with open('data/processed_data/processor.pkl', 'rb') as f:
    proc = pickle.load(f)

scaler = proc['scaler']
names = proc['feature_names']
print(f"共{len(names)}个特征:")
for i, n in enumerate(names):
    print(f"  [{i:3d}] {n}  mean={scaler.mean_[i]:.5g}  scale={scaler.scale_[i]:.5g}")

# 找出"0均值, 1方差"时预测结果
# 即如果我们传入全0（对应StandardScaler(0)=(0-mean)/scale=-mean/scale）
print("\n全0原始值经过scaler的结果（前20个）:")
x_zero = np.zeros(115)
x_scaled = (x_zero - scaler.mean_) / scaler.scale_
print(x_scaled[:20])
print(f"scaler输出的统计: max={x_scaled.max():.4f} min={x_scaled.min():.4f}")

# 找出对应normal流量的大概特征值
# normal流量在KDD99中的典型特征
# duration=0, count=1-2, same_srv_rate=1, diff_srv_rate=0, serror_rate=0, rerror_rate=0
normal_kdd = np.zeros(115)
# duration=0 [idx=0]
# src_bytes≈500 [idx=1]
# count=2 [idx=19]
# srv_count=2 [idx=20]
# same_srv_rate=1 [idx=25]
# diff_srv_rate=0 [idx=26]
# dst_host_count=255 [idx=28]
# dst_host_srv_count=255 [idx=29]
# logged_in=1 [idx=8]
idx_map = {n: i for i, n in enumerate(names)}
normal_kdd[idx_map.get('duration', 0)] = 0
normal_kdd[idx_map.get('src_bytes', 1)] = 500
normal_kdd[idx_map.get('logged_in', 8)] = 1
normal_kdd[idx_map.get('count', 19)] = 2
normal_kdd[idx_map.get('srv_count', 20)] = 2
normal_kdd[idx_map.get('serror_rate', 21)] = 0.0
normal_kdd[idx_map.get('same_srv_rate', 25)] = 1.0
normal_kdd[idx_map.get('diff_srv_rate', 26)] = 0.0
normal_kdd[idx_map.get('dst_host_count', 28)] = 10
normal_kdd[idx_map.get('dst_host_srv_count', 29)] = 10
normal_kdd[idx_map.get('dst_host_same_srv_rate', 30) if 'dst_host_same_srv_rate' in idx_map else 30] = 1.0

print("\n模拟KDD正常样本（部分赋值）经scaler后:")
kdd_scaled = scaler.transform(normal_kdd.reshape(1, -1))[0]
print(kdd_scaled[:30])
