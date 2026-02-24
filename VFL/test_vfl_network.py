#!/usr/bin/env python3
"""
VFL网络流量识别测试脚本
加载训练好的模型并进行详细评估
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# 导入VFL模块
from federated_learning.vfl_server import VFLServer
from federated_learning.vfl_client import create_vfl_parties
from federated_learning.vfl_utils import create_vfl_model_split, split_features_for_cnn

# 设置
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("VFL网络流量识别 - 模型测试")
print("="*80)
print(f"使用设备: {device}")


def load_test_data():
    """加载测试数据"""
    print("\n【1/5】加载测试数据...")
    
    data_dir = 'data/processed_data'
    
    X_test = np.load(os.path.join(data_dir, 'test_images.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
    
    with open(os.path.join(data_dir, 'processor.pkl'), 'rb') as f:
        processor = pickle.load(f)
    
    print(f"  测试集: {X_test.shape}")
    print(f"  标签: {y_test.shape}")
    print(f"  类别: {processor['class_names']}")
    
    return X_test, y_test, processor


def load_vfl_model(model_dir):
    """加载VFL模型"""
    print(f"\n【2/5】加载VFL模型...")
    
    # 加载配置
    with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    num_parties = config['num_parties']
    num_classes = config['num_classes']
    shapes = config['shapes']
    
    print(f"  参与方数: {num_parties}")
    print(f"  类别数: {num_classes}")
    
    # 创建模型架构
    bottom_models, top_model = create_vfl_model_split(
        num_parties, shapes, num_classes=num_classes
    )
    
    # 加载权重
    top_model.load_state_dict(
        torch.load(os.path.join(model_dir, 'top_model.pth'), 
                  map_location=device)
    )
    
    for i, model in enumerate(bottom_models):
        model.load_state_dict(
            torch.load(os.path.join(model_dir, f'bottom_model_party{i+1}.pth'),
                      map_location=device)
        )
    
    print("  ✓ 模型加载成功")
    
    return bottom_models, top_model, config


def predict_vfl(parties, server, X_test_parties, y_test, device, batch_size=256):
    """
    使用VFL模型进行预测
    """
    print("\n【3/5】进行预测...")
    
    server.top_model.eval()
    for party in parties:
        party.bottom_model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    num_samples = len(y_test)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            labels = torch.LongTensor(y_test[start_idx:end_idx]).to(device)
            
            # 各方前向传播
            embeddings = []
            for i, party in enumerate(parties):
                test_data = torch.FloatTensor(X_test_parties[i][start_idx:end_idx]).to(device)
                emb = party.bottom_model(test_data)
                embeddings.append(emb)
            
            # 聚合
            combined = server.aggregate_embeddings(embeddings)
            
            # 顶层预测
            outputs = server.forward_top_model(combined)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def evaluate_model(y_true, y_pred, y_probs, class_names):
    """
    评估模型性能
    """
    print("\n【4/5】评估模型性能...")
    
    # 整体准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n整体准确率: {accuracy*100:.2f}%")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, 
                               target_names=class_names, 
                               digits=4))
    
    # 各类别指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    print("\n各类别详细指标:")
    print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<10}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(f"{name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} "
              f"{f1[i]:<10.4f} {support[i]:<10}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """
    绘制混淆矩阵
    """
    print("\n【5/5】生成可视化...")
    
    plt.figure(figsize=(10, 8))
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '比例'})
    
    plt.title('VFL模型混淆矩阵（归一化）', fontsize=14, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  混淆矩阵已保存到: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics, class_names, save_path):
    """
    绘制各类别指标对比图
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, metrics['precision'], width, 
                    label='精确率', alpha=0.8)
    rects2 = ax.bar(x, metrics['recall'], width, 
                    label='召回率', alpha=0.8)
    rects3 = ax.bar(x + width, metrics['f1'], width, 
                    label='F1分数', alpha=0.8)
    
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('VFL模型各类别性能指标', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  指标对比图已保存到: {save_path}")
    plt.close()


def plot_class_distribution(y_true, y_pred, class_names, save_path):
    """
    绘制类别分布对比
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 真实分布
    true_counts = np.bincount(y_true, minlength=len(class_names))
    ax1.bar(range(len(class_names)), true_counts, alpha=0.7, color='skyblue')
    ax1.set_xlabel('类别', fontsize=12)
    ax1.set_ylabel('样本数', fontsize=12)
    ax1.set_title('真实类别分布', fontsize=13)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 预测分布
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    ax2.bar(range(len(class_names)), pred_counts, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('类别', fontsize=12)
    ax2.set_ylabel('样本数', fontsize=12)
    ax2.set_title('预测类别分布', fontsize=13)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  类别分布图已保存到: {save_path}")
    plt.close()


def main():
    """主函数"""
    MODEL_DIR = 'models/vfl_network'
    RESULTS_DIR = 'results'
    BATCH_SIZE = 256
    
    # 创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. 加载测试数据
    X_test, y_test, processor = load_test_data()
    class_names = processor['class_names']
    
    # 2. 加载模型
    bottom_models, top_model, config = load_vfl_model(MODEL_DIR)
    
    # 3. 垂直分割测试数据
    X_test_parties, _ = split_features_for_cnn(X_test, config['num_parties'])
    
    # 4. 创建测试用的参与方
    parties = create_vfl_parties(
        X_test_parties, y_test, device,
        batch_size=BATCH_SIZE,
        active_party_id=0
    )
    
    # 为各方设置模型
    for party, model in zip(parties, bottom_models):
        party.set_bottom_model(model)
    
    # 5. 创建服务器
    server = VFLServer(top_model, device, 
                      num_parties=config['num_parties'],
                      use_privbox=config['use_privbox'])
    
    # 6. 预测
    y_pred, y_true, y_probs = predict_vfl(parties, server, X_test_parties, y_test, device, BATCH_SIZE)
    
    # 7. 评估
    metrics = evaluate_model(y_true, y_pred, y_probs, class_names)
    
    # 8. 可视化
    plot_confusion_matrix(
        metrics['confusion_matrix'], class_names,
        os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    )
    
    plot_metrics_comparison(
        metrics, class_names,
        os.path.join(RESULTS_DIR, 'metrics_comparison.png')
    )
    
    plot_class_distribution(
        y_true, y_pred, class_names,
        os.path.join(RESULTS_DIR, 'class_distribution.png')
    )
    
    # 9. 保存结果
    results = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'metrics': metrics,
        'class_names': class_names,
        'config': config
    }
    
    with open(os.path.join(RESULTS_DIR, 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*80)
    print("测试完成! ✓")
    print("="*80)
    print(f"\n最终测试准确率: {metrics['accuracy']*100:.2f}%")
    print(f"\n结果已保存到: {RESULTS_DIR}")
    print("  - confusion_matrix.png      (混淆矩阵)")
    print("  - metrics_comparison.png    (指标对比)")
    print("  - class_distribution.png    (类别分布)")
    print("  - test_results.pkl          (详细结果)")


if __name__ == '__main__':
    main()
