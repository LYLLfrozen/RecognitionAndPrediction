#!/usr/bin/env python3
"""
检查模型偏差 - 分析模型是否过度偏向DOS类别
"""

import torch
import numpy as np
from realtime_monitor import VFLFlowClassifier

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classifier = VFLFlowClassifier('models/vfl_network', device)

print("="*80)
print("模型偏差检查")
print("="*80)

# 测试不同的输入
test_cases = [
    ("全零输入", np.zeros((1, 1, 11, 11))),
    ("全1输入", np.ones((1, 1, 11, 11))),
    ("随机小值", np.random.randn(1, 1, 11, 11) * 0.1),
    ("随机大值", np.random.randn(1, 1, 11, 11) * 10),
]

print("\n不同输入的预测结果:")
print(f"{'输入类型':<15s} {'预测':<10s} {'置信度':<10s} {'DOS概率':<10s}")
print("-" * 60)

for name, input_data in test_cases:
    pred_class, confidence, probs = classifier.classify(input_data.astype(np.float32))
    print(f"{name:<15s} {pred_class:<10s} {confidence:.4f}     {probs[0]:.4f}")

# 加载测试集并检查各类别的预测情况
X_test = np.load('data/processed_data/test_images.npy')
y_test = np.load('data/processed_data/test_labels.npy')

print("\n" + "="*80)
print("测试集各类别的预测情况（每类随机抽取10个）")
print("="*80)

for target_label in range(5):
    class_name = classifier.class_names[target_label]
    indices = np.where(y_test == target_label)[0]
    
    if len(indices) == 0:
        continue
    
    sample_indices = np.random.choice(indices, min(10, len(indices)), replace=False)
    predictions = []
    confidences = []
    dos_probs = []
    
    for idx in sample_indices:
        sample = X_test[idx:idx+1]
        pred, conf, probs = classifier.classify(sample)
        predictions.append(pred)
        confidences.append(conf)
        dos_probs.append(probs[0])
    
    from collections import Counter
    pred_counter = Counter(predictions)
    
    print(f"\n[{class_name}] 预测分布:")
    for pred, count in pred_counter.most_common():
        print(f"  {pred:8s}: {count}/10")
    print(f"  平均DOS概率: {np.mean(dos_probs):.4f}")
    print(f"  平均置信度: {np.mean(confidences):.4f}")

print("\n" + "="*80)
print("诊断结果:")
print("="*80)
print("如果所有输入都预测为DOS，说明:")
print("1. 模型过拟合到多数类（DOS占79%）")
print("2. 训练时可能没有使用类别权重平衡")
print("3. 需要重新训练，使用以下策略:")
print("   - 使用类别权重 (class_weight)")
print("   - 使用Focal Loss")
print("   - 过采样少数类或欠采样多数类")
print("   - 增加少数类的训练样本")
