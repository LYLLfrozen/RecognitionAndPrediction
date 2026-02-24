#!/usr/bin/env python3
"""
模型诊断脚本
检查模型是否能正确识别各种攻击类型
"""

import torch
import numpy as np
import pickle
import os
from collections import Counter

from federated_learning.vfl_utils import split_features_for_cnn

# 加载配置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载模型配置
with open('models/vfl_network/config.pkl', 'rb') as f:
    config = pickle.load(f)

print(f"\n模型类别: {config['class_names']}")
print(f"类别数: {len(config['class_names'])}")

# 加载测试数据
X_test = np.load('data/processed_data/test_images.npy')
y_test = np.load('data/processed_data/test_labels.npy')

print(f"\n测试集形状: {X_test.shape}")
print(f"标签形状: {y_test.shape}")

# 标签分布
print(f"\n测试集标签分布:")
label_counts = Counter(y_test)
for label, count in sorted(label_counts.items()):
    print(f"  {label} ({config['class_names'][label]:8s}): {count:6d} ({count/len(y_test)*100:.2f}%)")

# 加载模型
from federated_learning.vfl_utils import create_vfl_model_split
bottom_models, top_model = create_vfl_model_split(
    config['num_parties'], config['shapes'], num_classes=len(config['class_names'])
)

top_model.load_state_dict(
    torch.load('models/vfl_network/top_model.pth', map_location=device)
)

for i, model in enumerate(bottom_models):
    model.load_state_dict(
        torch.load(f'models/vfl_network/bottom_model_party{i+1}.pth', map_location=device)
    )

top_model.eval()
for model in bottom_models:
    model.eval()

bottom_models = [m.to(device) for m in bottom_models]
top_model = top_model.to(device)

print("\n模型已加载")

# 测试每个类别的样本
print("\n" + "="*80)
print("测试各类别样本的预测情况")
print("="*80)

for target_label in range(len(config['class_names'])):
    class_name = config['class_names'][target_label]
    
    # 找到这个类别的所有样本
    indices = np.where(y_test == target_label)[0]
    
    if len(indices) == 0:
        print(f"\n[{class_name}] 无测试样本")
        continue
    
    print(f"\n[{class_name}] 共 {len(indices)} 个样本")
    
    # 随机选择最多100个样本进行测试
    test_indices = np.random.choice(indices, min(100, len(indices)), replace=False)
    
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for idx in test_indices:
            flow = X_test[idx:idx+1]
            
            # 垂直分割
            X_parties, _ = split_features_for_cnn(flow, config['num_parties'])
            
            # 各方计算嵌入
            embeddings = []
            for i, model in enumerate(bottom_models):
                X_tensor = torch.FloatTensor(X_parties[i]).to(device)
                emb = model(X_tensor)
                embeddings.append(emb)
            
            # 聚合
            combined = torch.cat(embeddings, dim=-1)
            
            # 预测
            outputs = top_model(combined)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
            
            predictions.append(predicted.item())
            confidences.append(confidence.item())
    
    # 统计预测结果
    pred_counts = Counter(predictions)
    correct = sum(1 for p in predictions if p == target_label)
    accuracy = correct / len(predictions) * 100
    
    print(f"  准确率: {accuracy:.2f}% ({correct}/{len(predictions)})")
    print(f"  平均置信度: {np.mean(confidences):.4f}")
    print(f"  预测分布:")
    for pred_label, count in sorted(pred_counts.items()):
        pred_name = config['class_names'][pred_label]
        marker = "✓" if pred_label == target_label else "✗"
        print(f"    {marker} {pred_name:8s}: {count:3d} ({count/len(predictions)*100:.1f}%)")

print("\n" + "="*80)
print("诊断完成")
print("="*80)

# 总结问题
print("\n🔍 问题分析:")
print("  如果DOS、R2L等攻击类型的准确率很低，可能原因:")
print("  1. 数据不平衡 - 某些类别样本太少（如R2L只有225个，U2R只有10个）")
print("  2. 模型过拟合到多数类（normal和dos占98%）")
print("  3. 特征提取不充分 - 真实流量特征和训练数据格式不匹配")
print("  4. 模型训练不充分 - 需要更多训练轮次或调整损失函数")
