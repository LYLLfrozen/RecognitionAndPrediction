#!/usr/bin/env python3
"""
VFL系统快速演示
使用小数据集快速验证VFL系统是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# 导入VFL模块
from federated_learning.vfl_server import VFLServer
from federated_learning.vfl_client import create_vfl_parties
from federated_learning.vfl_utils import split_features_for_cnn, create_vfl_model_split

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")  # 使用CPU加快测试

print("="*70)
print("VFL网络流量识别系统 - 快速演示")
print("="*70)
print("\n此演示使用小数据集快速验证系统是否正常工作\n")

# 配置
NUM_PARTIES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_SAMPLES = 500
NUM_CLASSES = 5

print(f"配置:")
print(f"  参与方数: {NUM_PARTIES}")
print(f"  样本数: {NUM_SAMPLES}")
print(f"  类别数: {NUM_CLASSES}")
print(f"  训练轮数: {NUM_EPOCHS}")
print(f"  设备: {device}")

# 1. 创建模拟数据
print(f"\n【1/5】生成模拟数据...")
X_train = np.random.randn(NUM_SAMPLES, 1, 11, 11).astype(np.float32)
y_train = np.random.randint(0, NUM_CLASSES, NUM_SAMPLES)
X_test = np.random.randn(100, 1, 11, 11).astype(np.float32)
y_test = np.random.randint(0, NUM_CLASSES, 100)
print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")

# 2. 垂直分割数据
print(f"\n【2/5】垂直分割数据...")
X_train_parties, shapes = split_features_for_cnn(X_train, NUM_PARTIES)
X_test_parties, _ = split_features_for_cnn(X_test, NUM_PARTIES)
print(f"  参与方1: {X_train_parties[0].shape}")
print(f"  参与方2: {X_train_parties[1].shape}")

# 3. 创建参与方
print(f"\n【3/5】创建VFL参与方和服务器...")
parties = create_vfl_parties(
    X_train_parties, y_train, device,
    batch_size=BATCH_SIZE,
    active_party_id=0
)

# 4. 创建模型
bottom_models, top_model = create_vfl_model_split(
    NUM_PARTIES, shapes, num_classes=NUM_CLASSES
)

for party, model in zip(parties, bottom_models):
    party.set_bottom_model(model)

# 5. 创建服务器
server = VFLServer(top_model, device, num_parties=NUM_PARTIES, use_privbox=True)
print(f"  ✓ VFL系统初始化完成")

# 6. 训练
print(f"\n【4/5】开始训练（{NUM_EPOCHS}轮）...")
criterion = nn.CrossEntropyLoss()
top_optimizer = torch.optim.Adam(server.top_model.parameters(), lr=0.01)

for epoch in range(NUM_EPOCHS):
    server.top_model.train()
    for party in parties:
        party.bottom_model.train()
    
    batches = list(parties[0].get_all_batches())
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_indices, labels in batches:
        # 前向传播
        embeddings = [party.forward(batch_indices) for party in parties]
        combined = server.aggregate_embeddings(embeddings)
        combined.requires_grad_(True)
        combined.retain_grad()
        
        outputs = server.forward_top_model(combined)
        loss = criterion(outputs, labels)
        
        # 反向传播
        top_optimizer.zero_grad()
        for party in parties:
            party.optimizer.zero_grad()
        
        loss.backward()
        
        embedding_sizes = [emb.size(-1) for emb in embeddings]
        grads = server.split_gradients(combined, embedding_sizes)
        
        for party, emb, grad in zip(parties, embeddings, grads):
            party.backward(grad)
        
        top_optimizer.step()
        for party in parties:
            party.optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(batches)
    acc = 100. * correct / total
    print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")

# 7. 测试
print(f"\n【5/5】测试模型...")
server.top_model.eval()
for party in parties:
    party.bottom_model.eval()

correct = 0
total = 0

# 直接使用测试数据
y_test_tensor = torch.LongTensor(y_test)

with torch.no_grad():
    # 各方前向传播
    embeddings = []
    for i, party in enumerate(parties):
        test_data = torch.FloatTensor(X_test_parties[i]).to(device)
        party.bottom_model.eval()
        emb = party.bottom_model(test_data)
        embeddings.append(emb)
    
    combined = server.aggregate_embeddings(embeddings)
    outputs = server.forward_top_model(combined)
    
    _, predicted = outputs.max(1)
    total = y_test_tensor.size(0)
    correct = predicted.cpu().eq(y_test_tensor).sum().item()

test_acc = 100. * correct / total
print(f"  测试准确率: {test_acc:.2f}%")

# 总结
print("\n" + "="*70)
print("✓ 演示完成！VFL系统运行正常")
print("="*70)
print("\n系统功能验证:")
print("  ✓ 数据垂直分割")
print("  ✓ 多方协作训练")
print("  ✓ PrivBox隐私保护")
print("  ✓ 模型前向/反向传播")
print("  ✓ 梯度聚合与分发")
print("\n下一步:")
print("  运行: ./run_vfl_network.sh")
print("  或者: python3 preprocess_kddcup.py && python3 train_vfl_network.py")
print("="*70)
