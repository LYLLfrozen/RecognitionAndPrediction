#!/usr/bin/env python3
"""
测试攻击检测 - 比较模拟流量特征与训练数据
"""

import numpy as np
import pickle
import torch
from scapy.all import IP, TCP, UDP, ICMP, send
from realtime_monitor import VFLFlowClassifier
import time

# 加载分类器
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classifier = VFLFlowClassifier('models/vfl_network', device)

print("="*80)
print("攻击流量特征对比测试")
print("="*80)

# 1. 测试训练集中的真实DOS样本
print("\n【1】测试训练集中的DOS样本")
X_test = np.load('data/processed_data/test_images.npy')
y_test = np.load('data/processed_data/test_labels.npy')

dos_indices = np.where(y_test == 0)[0]  # 0 = dos
sample_idx = dos_indices[0]
dos_sample = X_test[sample_idx]

print(f"DOS样本形状: {dos_sample.shape}")
print(f"数据范围: min={dos_sample.min():.3f}, max={dos_sample.max():.3f}, mean={dos_sample.mean():.3f}")

pred_class, confidence, _ = classifier.classify(dos_sample)
print(f"预测结果: {pred_class} (置信度: {confidence:.4f})")

# 2. 模拟一个DOS SYN包的特征
print("\n【2】模拟DOS SYN包特征")
syn_features = np.zeros(41)
syn_features[0] = 60    # 包长度 (小SYN包)
syn_features[1] = 6     # TCP协议
syn_features[2] = 50000  # 随机源端口
syn_features[3] = 80    # 目标端口80
syn_features[4] = 0x02  # SYN flag
syn_features[5] = 64    # TTL=64
syn_features[6] = 60    # IP长度

print(f"SYN包特征 (前10维): {syn_features[:10]}")

# 预处理并预测
syn_processed = classifier.preprocess_flow(syn_features)
print(f"预处理后形状: {syn_processed.shape}")
print(f"预处理后范围: min={syn_processed.min():.3f}, max={syn_processed.max():.3f}")

pred_class, confidence, probs = classifier.classify(syn_processed)
print(f"预测结果: {pred_class} (置信度: {confidence:.4f})")
print(f"所有类别概率:")
for i, class_name in enumerate(classifier.class_names):
    print(f"  {class_name:8s}: {probs[i]:.4f}")

# 3. 模拟R2L攻击特征
print("\n【3】模拟R2L攻击特征 (FTP暴力破解)")
r2l_features = np.zeros(41)
r2l_features[0] = 150   # 较大的包（包含payload）
r2l_features[1] = 6     # TCP
r2l_features[2] = 45000 # 源端口
r2l_features[3] = 21    # FTP端口
r2l_features[4] = 0x18  # PSH+ACK flags
r2l_features[5] = 64    # TTL
r2l_features[6] = 150   # IP长度
r2l_features[12] = 100  # payload大小

print(f"R2L包特征 (前13维): {r2l_features[:13]}")

r2l_processed = classifier.preprocess_flow(r2l_features)
pred_class, confidence, probs = classifier.classify(r2l_processed)
print(f"预测结果: {pred_class} (置信度: {confidence:.4f})")
print(f"所有类别概率:")
for i, class_name in enumerate(classifier.class_names):
    print(f"  {class_name:8s}: {probs[i]:.4f}")

# 4. 对比测试集中的R2L样本
print("\n【4】测试训练集中的R2L样本")
r2l_indices = np.where(y_test == 3)[0]  # 3 = r2l
if len(r2l_indices) > 0:
    r2l_test_sample = X_test[r2l_indices[0]]
    print(f"R2L测试样本范围: min={r2l_test_sample.min():.3f}, max={r2l_test_sample.max():.3f}")
    
    pred_class, confidence, _ = classifier.classify(r2l_test_sample)
    print(f"预测结果: {pred_class} (置信度: {confidence:.4f})")

print("\n" + "="*80)
print("关键发现:")
print("="*80)
print("1. 预处理函数将41维真实特征扩展到121维")
print("2. 但训练数据是115维KDD特征")
print("3. 特征分布和尺度差异很大")
print("4. 需要改进特征提取，使其与训练数据更匹配")
