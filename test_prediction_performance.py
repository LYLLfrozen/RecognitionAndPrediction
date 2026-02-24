#!/usr/bin/env python3
"""
模型预测性能全面测试脚本
测试模型在测试集上的完整性能，包括准确率、精确率、召回率、F1值等指标
"""

import torch
import numpy as np
import os
from model.fl_woa_cnn_lstm.cnn_lstm_model import CNNLSTMModel
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_data():
    """加载模型和测试数据"""
    print("="*70)
    print("CNN-LSTM模型 - 完整性能评估")
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
    
    print(f"  ✓ 模型训练轮数: {checkpoint['epoch']}")
    print(f"  ✓ 最佳验证准确率: {checkpoint['val_acc']:.4f}")
    print(f"  ✓ 攻击类别: {processor['class_names']}")
    
    # 加载测试数据
    print("\n【2/3】加载测试数据...")
    data_dir = 'data/processed_data'
    X_test = np.load(os.path.join(data_dir, 'test_images.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
    
    print(f"  ✓ 测试集大小: {X_test.shape}")
    print(f"  ✓ 测试样本数: {len(X_test)}")
    
    return model, X_test, y_test, processor


def evaluate_model(model, X_test, y_test, processor, batch_size=256):
    """在完整测试集上评估模型"""
    print("\n【3/3】开始完整评估...")
    
    class_names = processor['class_names']
    n_samples = len(X_test)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    all_probabilities = []
    
    print(f"  处理 {n_batches} 个批次...")
    
    # 分批预测
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = torch.FloatTensor(X_test[i:end_idx]).to(device)
            
            outputs = model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"    进度: {i // batch_size + 1}/{n_batches} 批次")
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    return all_predictions, all_probabilities


def print_results(y_test, predictions, probabilities, class_names):
    """打印详细评估结果"""
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    
    # 1. 总体准确率
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n【总体准确率】: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 2. 各类别统计
    print("\n【各类别准确率】:")
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_accuracy = accuracy_score(y_test[class_mask], predictions[class_mask])
            print(f"  {class_name:10s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    # 3. 详细分类报告
    print("\n【分类报告】:")
    print(classification_report(y_test, predictions, target_names=class_names, digits=4))
    
    # 4. 混淆矩阵
    print("\n【混淆矩阵】:")
    cm = confusion_matrix(y_test, predictions)
    print(f"\n{'':10s}", end='')
    for name in class_names:
        print(f"{name:10s}", end='')
    print()
    print("-" * (10 + 10 * len(class_names)))
    
    for i, name in enumerate(class_names):
        print(f"{name:10s}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i][j]:10d}", end='')
        print()
    
    # 5. 平均置信度
    print("\n【平均置信度】:")
    avg_confidence = np.mean(np.max(probabilities, axis=1))
    print(f"  平均置信度: {avg_confidence:.4f}")
    
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_probs = probabilities[class_mask]
            class_confidence = np.mean(np.max(class_probs, axis=1))
            print(f"  {class_name:10s}: {class_confidence:.4f}")
    
    return cm


def save_confusion_matrix_plot(cm, class_names, save_path='confusion_matrix.png'):
    """保存混淆矩阵图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('混淆矩阵 - CNN-LSTM模型')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n混淆矩阵图已保存到: {save_path}")


def analyze_errors(y_test, predictions, X_test, processor, n_errors=10):
    """分析错误预测案例"""
    print("\n" + "="*70)
    print("错误案例分析（前10个）")
    print("="*70)
    
    class_names = processor['class_names']
    
    # 找到错误预测
    error_indices = np.where(y_test != predictions)[0]
    print(f"\n总错误数: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")
    
    if len(error_indices) > 0:
        print(f"\n展示前 {min(n_errors, len(error_indices))} 个错误案例:")
        print(f"{'序号':<6} {'真实标签':<15} {'预测标签':<15} {'错误类型':<20}")
        print("-"*70)
        
        for i, idx in enumerate(error_indices[:n_errors]):
            true_label = class_names[y_test[idx]]
            pred_label = class_names[predictions[idx]]
            error_type = f"{true_label} → {pred_label}"
            print(f"{i+1:<6} {true_label:<15} {pred_label:<15} {error_type:<20}")


def main():
    """主函数"""
    # 1. 加载模型和数据
    model, X_test, y_test, processor = load_model_and_data()
    
    # 2. 评估模型
    predictions, probabilities = evaluate_model(model, X_test, y_test, processor)
    
    # 3. 打印结果
    cm = print_results(y_test, predictions, probabilities, processor['class_names'])
    
    # 4. 保存混淆矩阵图
    try:
        save_confusion_matrix_plot(cm, processor['class_names'])
    except Exception as e:
        print(f"\n保存混淆矩阵图失败: {e}")
    
    # 5. 分析错误案例
    analyze_errors(y_test, predictions, X_test, processor)
    
    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)


if __name__ == "__main__":
    main()
