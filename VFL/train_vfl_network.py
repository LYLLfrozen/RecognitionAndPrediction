#!/usr/bin/env python3
"""
VFL网络流量识别训练脚本
模拟多个参与方进行垂直联邦学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# 导入VFL模块
from federated_learning.vfl_server import VFLServer
from federated_learning.vfl_client import VFLActiveParty, VFLPassiveParty, create_vfl_parties
from federated_learning.vfl_utils import (
    split_features_for_cnn, 
    create_vfl_model_split,
    print_vfl_data_distribution,
    calculate_communication_cost
)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
# 强制使用MPS (Apple Silicon GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ 使用MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ 使用CUDA GPU")
else:
    device = torch.device("cpu")
    print("✓ 使用CPU")

print("="*80)
print("垂直联邦学习 (VFL) - 网络流量识别")
print("="*80)
print("垂直联邦学习 (VFL) - 网络流量识别")
print("="*80)
print(f"使用设备: {device}")


def load_data():
    """加载处理后的数据"""
    print("\n【1/6】加载数据...")
    
    data_dir = 'data/processed_data'
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请先运行: python preprocess_kddcup.py")
        sys.exit(1)
    
    X_train = np.load(os.path.join(data_dir, 'train_images.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    X_test = np.load(os.path.join(data_dir, 'test_images.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
    
    with open(os.path.join(data_dir, 'processor.pkl'), 'rb') as f:
        processor = pickle.load(f)
    
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
    print(f"  类别: {processor['class_names']}")
    print(f"  类别数: {processor['n_classes']}")
    
    return X_train, y_train, X_test, y_test, processor


def split_data_vertical(X_train, X_test, num_parties):
    """
    垂直分割数据 - 不同参与方获得相同样本的不同特征
    """
    print(f"\n【2/6】垂直分割数据给 {num_parties} 个参与方...")
    
    # 分割训练数据
    X_train_parties, train_shapes = split_features_for_cnn(X_train, num_parties)
    
    # 分割测试数据
    X_test_parties, test_shapes = split_features_for_cnn(X_test, num_parties)
    
    print(f"\n垂直分割结果:")
    for i, (X_party, shape) in enumerate(zip(X_train_parties, train_shapes)):
        print(f"  参与方 {i+1}: {X_party.shape} -> 特征区域: {shape}")
    
    return X_train_parties, X_test_parties, train_shapes


def train_vfl(parties, server, epochs, batch_size, X_test_parties, y_test, device):
    """
    训练VFL模型
    """
    print(f"\n【3/6】开始VFL训练...")
    print(f"  轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  参与方数: {len(parties)}")
    
    active_party = parties[0]  # 第一个参与方是主动方
    criterion = nn.CrossEntropyLoss()
    
    # 为顶层模型创建优化器
    top_optimizer = optim.Adam(server.top_model.parameters(), lr=0.001)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        # 训练阶段
        server.top_model.train()
        for party in parties:
            party.bottom_model.train()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # 获取批次
        batches = list(active_party.get_all_batches())
        
        with tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_indices, labels in pbar:
                # 确保labels在正确的设备上
                labels = labels.to(device)
                
                # 1. 各参与方前向传播（计算嵌入向量）
                embeddings = []
                for party in parties:
                    emb = party.forward(batch_indices)
                    embeddings.append(emb)
                
                # 2. 服务器聚合嵌入向量（使用PrivBox）
                combined = server.aggregate_embeddings(embeddings)
                
                # 保留计算图
                combined.requires_grad_(True)
                combined.retain_grad()
                
                # 3. 顶层模型前向传播
                outputs = server.forward_top_model(combined)
                loss = criterion(outputs, labels)
                
                # 4. 反向传播
                # 清零梯度
                top_optimizer.zero_grad()
                for party in parties:
                    party.optimizer.zero_grad()
                
                # 顶层反向传播
                loss.backward()
                
                # 5. 获取并分割梯度
                embedding_sizes = [emb.size(-1) for emb in embeddings]
                grads = server.split_gradients(combined, embedding_sizes)
                
                # 6. 各参与方反向传播
                for party, emb, grad in zip(parties, embeddings, grads):
                    party.backward(grad)
                
                # 7. 更新参数
                top_optimizer.step()
                for party in parties:
                    party.optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # 计算平均损失和准确率
        avg_loss = epoch_loss / len(batches)
        train_acc = 100. * correct / total
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        
        # 测试阶段
        test_loss, test_acc = evaluate_vfl(parties, server, criterion, X_test_parties, y_test, device)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  训练 - 损失: {avg_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"  测试 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")
    
    return history


def evaluate_vfl(parties, server, criterion, X_test_parties, y_test, device, batch_size=256):
    """
    评估VFL模型
    """
    server.top_model.eval()
    for party in parties:
        party.bottom_model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 分批测试
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
            
            combined = server.aggregate_embeddings(embeddings)
            outputs = server.forward_top_model(combined)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def save_model(parties, server, save_dir):
    """
    保存模型
    """
    print(f"\n【4/6】保存模型...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存顶层模型
    torch.save(server.top_model.state_dict(), 
               os.path.join(save_dir, 'top_model.pth'))
    
    # 保存各参与方的底层模型
    for i, party in enumerate(parties):
        torch.save(party.bottom_model.state_dict(),
                  os.path.join(save_dir, f'bottom_model_party{i+1}.pth'))
    
    print(f"  模型已保存到: {save_dir}")


def plot_history(history, save_path):
    """
    绘制训练历史
    """
    print(f"\n【5/6】生成训练曲线...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(history['train_loss'], label='训练损失')
    ax1.plot(history['test_loss'], label='测试损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('VFL训练损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(history['train_acc'], label='训练准确率')
    ax2.plot(history['test_acc'], label='测试准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('VFL训练准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  训练曲线已保存到: {save_path}")
    plt.close()


def main():
    """主函数"""
    # ==================== 配置 ====================
    NUM_PARTIES = 3       # 参与方数量（2-4）
    BATCH_SIZE = 512      # 批次大小（更大的批次加快CPU训练）
    EPOCHS = 10           # 训练轮数
    USE_PRIVBOX = True    # 是否使用PrivBox隐私保护
    SAVE_DIR = 'models/vfl_network'
    # ============================================
    
    start_time = time.time()
    
    # 1. 加载数据
    X_train, y_train, X_test, y_test, processor = load_data()
    num_classes = processor['n_classes']
    
    # 2. 垂直分割数据
    X_train_parties, X_test_parties, shapes = split_data_vertical(
        X_train, X_test, NUM_PARTIES
    )
    
    # 3. 创建参与方
    print(f"\n【2.5/6】创建 {NUM_PARTIES} 个参与方...")
    parties = create_vfl_parties(
        X_train_parties, y_train, device, 
        batch_size=BATCH_SIZE, 
        active_party_id=0
    )
    print(f"  主动方: 参与方1 (拥有标签)")
    print(f"  被动方: 参与方2-{NUM_PARTIES} (仅特征)")
    
    # 4. 创建模型
    print(f"\n创建VFL模型...")
    bottom_models, top_model = create_vfl_model_split(
        NUM_PARTIES, shapes, num_classes=num_classes
    )
    
    # 为各参与方分配底层模型
    for party, model in zip(parties, bottom_models):
        party.set_bottom_model(model)
    print("  底层模型已分配给各参与方")
    
    # 5. 创建服务器
    server = VFLServer(top_model, device, num_parties=NUM_PARTIES, 
                      use_privbox=USE_PRIVBOX)
    print(f"  VFL服务器已启动")
    print(f"  隐私保护: {'✓ PrivBox启用' if USE_PRIVBOX else '✗ 未启用'}")
    
    # 6. 训练
    history = train_vfl(parties, server, EPOCHS, BATCH_SIZE, X_test_parties, y_test, device)
    
    # 7. 保存模型
    save_model(parties, server, SAVE_DIR)
    
    # 8. 绘制训练曲线
    plot_history(history, os.path.join(SAVE_DIR, 'training_history.png'))
    
    # 9. 最终评估
    print(f"\n【6/6】最终评估...")
    final_test_loss, final_test_acc = evaluate_vfl(
        parties, server, nn.CrossEntropyLoss(), X_test_parties, y_test, device
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("VFL训练完成! ✓")
    print("="*80)
    print(f"\n配置:")
    print(f"  参与方数: {NUM_PARTIES}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  隐私保护: {'PrivBox' if USE_PRIVBOX else '无'}")
    print(f"\n最终结果:")
    print(f"  测试准确率: {final_test_acc:.2f}%")
    print(f"  测试损失: {final_test_loss:.4f}")
    print(f"  训练时间: {elapsed_time/60:.1f} 分钟")
    print(f"\n模型保存位置: {SAVE_DIR}")
    
    # 保存配置信息
    config = {
        'num_parties': NUM_PARTIES,
        'num_classes': num_classes,
        'class_names': processor['class_names'],
        'shapes': shapes,
        'use_privbox': USE_PRIVBOX,
        'final_test_acc': final_test_acc,
        'final_test_loss': final_test_loss
    }
    
    with open(os.path.join(SAVE_DIR, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)


if __name__ == '__main__':
    main()
