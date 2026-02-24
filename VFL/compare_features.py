#!/usr/bin/env python3
"""
特征分布对比 - 比较训练数据与模拟流量的特征分布
"""

import numpy as np
import matplotlib.pyplot as plt
from flow_tracker import FlowTracker
import time

# 加载训练数据
X_train = np.load('data/processed_data/train_images.npy')
y_train = np.load('data/processed_data/train_labels.npy')

print("="*80)
print("特征分布对比分析")
print("="*80)

# 选择DOS类别的训练样本
dos_indices = np.where(y_train == 0)[0][:100]
dos_train_samples = X_train[dos_indices]

print(f"\n训练集DOS样本: {dos_train_samples.shape}")
print(f"  均值: {dos_train_samples.mean():.4f}")
print(f"  标准差: {dos_train_samples.std():.4f}")
print(f"  最小值: {dos_train_samples.min():.4f}")
print(f"  最大值: {dos_train_samples.max():.4f}")

# 模拟DOS攻击流量
print("\n生成模拟DOS攻击流量...")
tracker = FlowTracker()
simulated_dos = []

for i in range(100):
    packet = {
        'src_ip': f'10.0.{i//256}.{i%256}',
        'dst_ip': '192.168.1.1',
        'src_port': 10000 + i,
        'dst_port': 80,
        'protocol': 6,
        'tcp_flags': 0x02,
        'packet_size': 60,
        'ttl': 64,
        'timestamp': time.time() + i * 0.001
    }
    
    # 基础特征
    features = np.zeros(41)
    features[0] = packet['packet_size'] / 1500.0
    features[1] = packet['protocol'] / 20.0
    features[2] = packet['src_port'] / 65535.0
    features[3] = packet['dst_port'] / 65535.0
    features[4] = packet['tcp_flags'] / 255.0
    features[5] = packet['ttl'] / 255.0
    features[6] = packet['packet_size'] / 1500.0
    
    # 流统计
    flow_stats = tracker.update(packet)
    enhanced = tracker.features_to_vector(flow_stats, features)
    
    # 扩展到121维并重塑
    padding = np.zeros(121 - 41)
    full_features = np.concatenate([enhanced, padding])
    reshaped = full_features.reshape(1, 1, 11, 11)
    
    simulated_dos.append(reshaped)

simulated_dos = np.array(simulated_dos)
print(f"\n模拟DOS样本: {simulated_dos.shape}")
print(f"  均值: {simulated_dos.mean():.4f}")
print(f"  标准差: {simulated_dos.std():.4f}")
print(f"  最小值: {simulated_dos.min():.4f}")
print(f"  最大值: {simulated_dos.max():.4f}")

# 详细对比前10个特征
print("\n" + "="*80)
print("前10个特征值对比 (训练 vs 模拟)")
print("="*80)

train_flat = dos_train_samples.reshape(len(dos_train_samples), -1)
sim_flat = simulated_dos.reshape(len(simulated_dos), -1)

print(f"\n{'特征':>10s} {'训练均值':>12s} {'模拟均值':>12s} {'差异':>12s}")
print("-" * 50)

for i in range(min(20, train_flat.shape[1])):
    train_mean = train_flat[:, i].mean()
    sim_mean = sim_flat[:, i].mean()
    diff = abs(train_mean - sim_mean)
    
    marker = "❗" if diff > 0.5 else ""
    print(f"{i:10d} {train_mean:12.4f} {sim_mean:12.4f} {diff:12.4f} {marker}")

print("\n" + "="*80)
print("关键发现")
print("="*80)
print("1. 训练数据是标准化后的KDD Cup 99特征（115维)")
print("2. 模拟数据是手工归一化的网络包特征（41维扩展）")
print("3. 特征分布差异巨大，导致模型无法正确分类")
print("")
print("💡 解决方案:")
print("1. 使用训练数据的scaler来归一化模拟特征")
print("2. 或者：在真实数据上微调(fine-tune)模型")
print("3. 或者：重新训练模型，使用与真实流量更接近的特征")
print("4. 推荐：创建一个特征映射层，将41维包特征映射到115维KDD特征空间")
