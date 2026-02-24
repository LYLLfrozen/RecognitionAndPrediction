#!/usr/bin/env python3
"""
完整版CNN-LSTM模型预测脚本
测试5类攻击识别效果
"""

import torch
import numpy as np
import pickle
import os
from model.fl_woa_cnn_lstm.cnn_lstm_model import CNNLSTMModel
from sklearn.metrics import classification_report, confusion_matrix
import random

# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_data():
    """加载模型和测试数据"""
    print("="*70)
    print("完整版CNN-LSTM模型 - 5类攻击识别预测")
    print("="*70)
    
    # 加载模型
    print("\n【1/3】加载模型...")
    model_path = 'model/saved_models/cnn_lstm_full.pth'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    processor = checkpoint['processor']
    
    # 创建模型
    model_config = {
        'input_shape': (1, 11, 11),
        'num_classes': processor['n_classes'],
        'cnn_filters': [32, 64, 128],
        'lstm_units': 64,
        'dropout_rate': 0.5
    }
    model = CNNLSTMModel(config=model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  模型训练轮数: {checkpoint['epoch']}")
    print(f"  最佳验证准确率: {checkpoint['val_acc']:.4f}")
    print(f"  类别: {processor['class_names']}")
    
    # 加载测试数据
    print("\n【2/3】加载测试数据...")
    data_dir = 'data/processed_data'
    X_test = np.load(os.path.join(data_dir, 'test_images.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
    
    print(f"  测试集大小: {X_test.shape}")
    print(f"  标签类型: {processor['class_names']}")
    
    return model, X_test, y_test, processor

def predict_samples(model, X_test, y_test, processor, n_samples=30):
    """预测样本"""
    print("\n【3/3】开始预测...")
    
    # 随机选择样本
    indices = random.sample(range(len(X_test)), n_samples)
    X_samples = X_test[indices]
    y_samples = y_test[indices]
    
    # 转换为张量
    X_tensor = torch.FloatTensor(X_samples).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    # 显示结果
    print("\n" + "="*70)
    print("预测结果")
    print("="*70)
    print(f"{'序号':<6} {'真实标签':<15} {'预测标签':<15} {'置信度':<10} {'结果':<10}")
    print("-"*70)
    
    correct = 0
    class_names = processor['class_names']
    
    for i, (true_label, pred_label, prob) in enumerate(zip(y_samples, predictions.cpu().numpy(), probabilities.cpu().numpy())):
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        confidence = prob[pred_label]
        is_correct = true_label == pred_label
        
        if is_correct:
            correct += 1
            result = "✓"
        else:
            result = "✗"
        
        print(f"{i+1:<6} {true_name:<15} {pred_name:<15} {confidence:<10.4f} {result:<10}")
    
    accuracy = correct / n_samples
    print("-"*70)
    print(f"准确率: {accuracy:.2%} ({correct}/{n_samples})")
    print("="*70)
    
    return accuracy

def evaluate_all(model, X_test, y_test, processor):
    """评估全部测试数据"""
    print("\n评估全部测试数据...")
    
    # 分批预测
    batch_size = 128
    all_preds = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            X_tensor = torch.FloatTensor(batch).to(device)
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 整体评估
    print("\n" + "="*70)
    print("整体评估结果")
    print("="*70)
    
    accuracy = np.mean(all_preds == y_test)
    print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 各类别统计
    print(f"\n各类别详细分析:")
    print(f"{'类别':<15} {'样本数':<10} {'正确预测':<12} {'准确率':<10} {'平均置信度':<12}")
    print("-"*70)
    
    class_names = processor['class_names']
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_samples = np.sum(mask)
            class_correct = np.sum((all_preds == y_test) & mask)
            class_acc = class_correct / class_samples
            class_conf = np.mean(all_probs[mask, i])
            
            print(f"{class_name:<15} {class_samples:<10} {class_correct:<12} {class_acc:<10.4f} {class_conf:<12.4f}")
    
    # 混淆统计
    print(f"\n混淆情况分析:")
    print(f"{'真实类别':<15} → {'预测类别':<15} {'数量':<10}")
    print("-"*70)
    
    cm = confusion_matrix(y_test, all_preds)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                print(f"{class_names[i]:<15} → {class_names[j]:<15} {cm[i, j]:<10}")
    
    print("="*70)
    
    return accuracy, all_preds, all_probs

def main():
    # 加载模型和数据
    model, X_test, y_test, processor = load_model_and_data()
    
    # 预测样本
    sample_acc = predict_samples(model, X_test, y_test, processor, n_samples=30)
    
    # 询问是否评估全部
    print("\n是否评估全部测试数据? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        overall_acc, all_preds, all_probs = evaluate_all(model, X_test, y_test, processor)
        
        # 展示一些有趣的统计
        print("\n" + "="*70)
        print("额外统计信息")
        print("="*70)
        
        # 高置信度预测
        max_probs = np.max(all_probs, axis=1)
        high_conf_mask = max_probs > 0.95
        high_conf_acc = np.mean(all_preds[high_conf_mask] == y_test[high_conf_mask])
        print(f"\n高置信度预测 (>95%):")
        print(f"  样本数: {np.sum(high_conf_mask)} ({np.sum(high_conf_mask)/len(y_test)*100:.2f}%)")
        print(f"  准确率: {high_conf_acc:.4f}")
        
        # 低置信度预测
        low_conf_mask = max_probs < 0.7
        if np.sum(low_conf_mask) > 0:
            low_conf_acc = np.mean(all_preds[low_conf_mask] == y_test[low_conf_mask])
            print(f"\n低置信度预测 (<70%):")
            print(f"  样本数: {np.sum(low_conf_mask)} ({np.sum(low_conf_mask)/len(y_test)*100:.2f}%)")
            print(f"  准确率: {low_conf_acc:.4f}")
        
        print("\n预测完成！")

if __name__ == '__main__':
    main()
