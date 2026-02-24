#!/usr/bin/env python3
"""
垂直联邦学习简单示例
展示如何使用VFL系统进行训练
"""

import torch
import numpy as np
from federated_learning.privbox import PrivBoxProtocol
from federated_learning.vfl_utils import split_features_for_cnn, create_vfl_model_split
from federated_learning.vfl_server import VFLServer
from federated_learning.vfl_client import create_vfl_parties

# 设置
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cpu")

print("="*70)
print("垂直联邦学习简单示例")
print("="*70)

# 1. 创建模拟数据
print("\n【1】创建模拟数据...")
n_samples = 1000
X = np.random.randn(n_samples, 1, 11, 11)
y = np.random.randint(0, 5, n_samples)
print(f"数据形状: {X.shape}, 标签: {y.shape}")

# 2. 垂直分割数据
print("\n【2】垂直分割数据（2个参与方）...")
X_parties, shapes = split_features_for_cnn(X, num_parties=2)
print(f"参与方1: {X_parties[0].shape}")
print(f"参与方2: {X_parties[1].shape}")

# 3. 创建参与方
print("\n【3】创建参与方...")
parties = create_vfl_parties(X_parties, y, device, batch_size=64, active_party_id=0)
print(f"主动方: 参与方1 (拥有标签)")
print(f"被动方: 参与方2 (仅特征)")

# 4. 创建模型
print("\n【4】创建VFL模型...")
bottom_models, top_model = create_vfl_model_split(2, shapes, num_classes=5)
for i, (party, model) in enumerate(zip(parties, bottom_models)):
    party.set_bottom_model(model)
print("底层模型已分配给各方")

# 5. 创建服务器
print("\n【5】创建VFL服务器...")
server = VFLServer(top_model, device, num_parties=2, use_privbox=True)
print("服务器已启动，PrivBox隐私保护已启用")

# 6. 训练一个批次
print("\n【6】训练一个批次...")
batches = list(parties[0].get_all_batches())
batch_indices, labels = batches[0]

# 各方前向传播
embeddings = []
for party in parties:
    emb = party.forward(batch_indices)
    embeddings.append(emb)
print(f"嵌入向量已计算: {[e.shape for e in embeddings]}")

# 服务器聚合
combined = server.aggregate_embeddings(embeddings)
combined.requires_grad_(True)
combined.retain_grad()
print(f"嵌入向量已聚合: {combined.shape}")

# 顶层前向传播
outputs = server.forward_top_model(combined)
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
print(f"损失: {loss.item():.4f}")

# 6. 演示训练流程（简化版）
print("\n【6】演示训练流程...")
print("实际训练中，各方嵌入向量通过计算图连接，可以正常反向传播")
print("示例中展示了前向传播过程")

# 模拟计算
print(f"\n训练批次数: {len(batches)}")
print(f"批次大小: {batch_indices.shape[0]}")
print(f"嵌入维度: {embeddings[0].shape[1]} (每个参与方)")
print(f"组合后: {combined.shape[1]} (总维度)")
print(f"分类输出: {outputs.shape[1]} (类别数)")

# 7. 总结
print("\n" + "="*70)
print("示例完成! ✓")
print("="*70)
print("\n核心概念:")
print("1. 垂直分割数据给多个参与方")
print("2. 各方计算嵌入向量（本地特征提取）")
print("3. 服务器安全聚合嵌入（PrivBox保护）")
print("4. 顶层模型分类并计算损失")
print("5. 分割梯度并添加差分隐私噪声")
print("6. 各方反向传播更新底层模型")
print("\n完整训练请运行: python3 train_vfl.py")
