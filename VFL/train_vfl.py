#!/usr/bin/env python3
"""
垂直联邦学习训练脚本（使用PrivBox隐私保护）
使用多进程模拟多个参与方
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
from multiprocessing import Process, Queue, Manager
import time

# 切换到项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# 导入VFL模块
from federated_learning.vfl_server import VFLServer
from federated_learning.vfl_client import VFLActiveParty, VFLPassiveParty, create_vfl_parties
from federated_learning.vfl_utils import (
    split_features_for_cnn, 
    create_vfl_model_split,
    print_vfl_data_distribution,
    calculate_communication_cost
)
from federated_learning.privbox import PrivBoxProtocol

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("垂直联邦学习 (VFL) with PrivBox 隐私保护")
print("="*80)
print(f"使用设备: {device}")

# 超参数
NUM_PARTIES = 2  # 参与方数量（2-4）
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 30
USE_PRIVBOX = True  # 是否使用PrivBox隐私保护


def load_data():
    """加载处理后的数据"""
    print("\n【1/7】加载数据...")
    
    data_dir = 'data/processed_data'
    
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
    垂直分割数据
    不同参与方获得相同样本的不同特征
    """
    print(f"\n【2/7】垂直分割数据给 {num_parties} 个参与方...")
    
    # 分割训练数据
    X_train_parties, train_shapes = split_features_for_cnn(X_train, num_parties)
    
    # 分割测试数据
    X_test_parties, test_shapes = split_features_for_cnn(X_test, num_parties)
    
    print(f"\n训练数据分割:")
    for i, (X_party, shape) in enumerate(zip(X_train_parties, train_shapes)):
        print(f"  参与方 {i+1}: {X_party.shape} -> 特征区域: {shape}")
    
    return X_train_parties, X_test_parties, train_shapes


def party_worker(party_id, party_type, X_party, y_train, bottom_model_state,
                input_queue, output_queue, device_str):
    """
    参与方工作进程
    
    Args:
        party_id: 参与方ID
        party_type: 'active' 或 'passive'
        X_party: 该方的数据
        y_train: 标签（仅主动方需要）
        bottom_model_state: 底层模型状态字典
        input_queue: 输入队列（接收指令）
        output_queue: 输出队列（返回结果）
        device_str: 设备字符串
    """
    # 设置设备
    device = torch.device(device_str)
    
    # 创建参与方
    if party_type == 'active':
        party = VFLActiveParty(party_id, X_party, y_train, device, BATCH_SIZE)
    else:
        party = VFLPassiveParty(party_id, X_party, device, BATCH_SIZE)
    
    # 创建底层模型
    from torch import nn
    # 简化：直接加载模型
    # 实际应该根据input_shape动态创建
    shape = X_party.shape[2:]  # (height, width)
    
    bottom_model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2) if min(shape) > 4 else nn.Identity(),
        nn.Dropout(0.3),
        
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2) if min(shape) > 8 else nn.Identity(),
        nn.Dropout(0.3),
        
        nn.Flatten(),
    )
    
    # 计算展平后的维度
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, shape[0], shape[1])
        flat_dim = bottom_model(dummy_input).shape[1]
    
    # 添加全连接层
    bottom_model = nn.Sequential(
        bottom_model,
        nn.Linear(flat_dim, 64),
        nn.ReLU()
    )
    
    party.set_bottom_model(bottom_model)
    
    if bottom_model_state:
        party.set_model_weights(bottom_model_state)
    
    # 使用局部变量并进行非空检查，避免可选成员访问的类型问题
    bm = party.bottom_model
    if bm is None:
        raise RuntimeError("Bottom model is not set for the party; cannot create optimizer.")
    optimizer = optim.Adam(bm.parameters(), lr=LEARNING_RATE)
    
    # 监听指令
    while True:
        command = input_queue.get()
        
        if command['type'] == 'forward':
            # 前向传播
            batch_indices = command['indices']
            embeddings = party.forward(batch_indices)
            output_queue.put({
                'party_id': party_id,
                'embeddings': embeddings.detach().cpu()
            })
        
        elif command['type'] == 'backward':
            # 反向传播
            grad = command['gradient'].to(device)
            party.backward(grad)
            party.update_model(optimizer)
            output_queue.put({'party_id': party_id, 'done': True})
        
        elif command['type'] == 'get_weights':
            # 获取模型权重
            weights = party.get_model_weights()
            output_queue.put({'party_id': party_id, 'weights': weights})
        
        elif command['type'] == 'terminate':
            # 终止进程
            break


def train_vfl_single_process(X_train_parties, y_train, X_test_parties, y_test,
                            input_shapes, num_classes, processor):
    """
    单进程版本的VFL训练（用于调试和快速测试）
    """
    print(f"\n【3/7】创建VFL模型...")
    
    # 创建底层和顶层模型
    bottom_models, top_model = create_vfl_model_split(NUM_PARTIES, input_shapes, num_classes)
    
    # 创建服务器
    server = VFLServer(top_model, device, NUM_PARTIES, use_privbox=USE_PRIVBOX)
    
    # 创建参与方
    parties = create_vfl_parties(X_train_parties, y_train, device, BATCH_SIZE, active_party_id=0)
    
    # 为各方设置底层模型
    for i, (party, bottom_model) in enumerate(zip(parties, bottom_models)):
        party.set_bottom_model(bottom_model)
    
    print(f"  参与方数量: {NUM_PARTIES}")
    print(f"  顶层模型参数: {sum(p.numel() for p in top_model.parameters()):,}")
    for i, model in enumerate(bottom_models):
        params = sum(p.numel() for p in model.parameters())
        print(f"  参与方{i+1}底层模型参数: {params:,}")
    
    # 优化器
    optimizers = []
    for model in bottom_models:
        if model is None:
            continue
        optimizers.append(optim.Adam(model.parameters(), lr=LEARNING_RATE))
    top_optimizer = optim.Adam(top_model.parameters(), lr=LEARNING_RATE)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n【4/7】开始VFL训练...")
    print(f"  轮数: {EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  PrivBox隐私保护: {'启用' if USE_PRIVBOX else '禁用'}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'communication_cost': []
    }
    
    for epoch in range(EPOCHS):
        # 训练模式
        for model in bottom_models:
            model.train()
        top_model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_comm_cost = 0.0
        
        # 获取主动方的数据加载器
        active_party = parties[0]
        train_loader = active_party.get_all_batches()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_indices, labels in pbar:
            labels = labels.to(device)
            
            # 清除之前的梯度
            for optimizer in optimizers:
                optimizer.zero_grad()
            top_optimizer.zero_grad()
            
            # 1. 各方前向传播计算嵌入
            embeddings = []
            for party in parties:
                emb = party.forward(batch_indices)
                embeddings.append(emb)
                epoch_comm_cost += calculate_communication_cost([emb])
            
            # 2. 服务器聚合嵌入
            combined = server.aggregate_embeddings(embeddings)
            
            # 3. 顶层模型前向传播
            outputs = server.forward_top_model(combined)
            loss = criterion(outputs, labels)
            
            # 4. 完整反向传播
            loss.backward()
            
            # 5. 更新所有模型
            for optimizer in optimizers:
                optimizer.step()
            top_optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['communication_cost'].append(epoch_comm_cost)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
              f"Comm: {epoch_comm_cost:.2f} MB")
    
    return server, parties, bottom_models, history


def save_models(server, bottom_models, save_dir='model/saved_models'):
    """保存模型"""
    print(f"\n【5/7】保存模型...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存顶层模型
    server.save_model(os.path.join(save_dir, 'vfl_top_model.pth'))
    
    # 保存各方底层模型
    for i, model in enumerate(bottom_models):
        path = os.path.join(save_dir, f'vfl_bottom_model_party{i+1}.pth')
        torch.save(model.state_dict(), path)
        print(f"  参与方{i+1}底层模型已保存: {path}")


def plot_training_history(history, save_path='vfl_training_history.png'):
    """绘制训练历史"""
    print(f"\n【6/7】绘制训练曲线...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], marker='o', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], marker='o', label='Train Acc', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Communication Cost
    axes[2].plot(history['communication_cost'], marker='s', label='Comm Cost', color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Communication (MB)')
    axes[2].set_title('Communication Cost per Epoch')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  训练曲线已保存: {save_path}")


def main():
    # 加载数据
    X_train, y_train, X_test, y_test, processor = load_data()
    
    # 垂直分割数据
    X_train_parties, X_test_parties, input_shapes = split_data_vertical(
        X_train, X_test, NUM_PARTIES
    )
    
    # 打印数据分布
    print_vfl_data_distribution(X_train_parties, y_train, processor['class_names'])
    
    # 训练VFL模型（单进程版本）
    server, parties, bottom_models, history = train_vfl_single_process(
        X_train_parties, y_train, X_test_parties, y_test,
        input_shapes, processor['n_classes'], processor
    )
    
    # 保存模型
    save_models(server, bottom_models)
    
    # 绘制训练曲线
    plot_training_history(history)
    
    print("\n【7/7】训练完成!")
    print(f"  总通信成本: {sum(history['communication_cost']):.2f} MB")
    print(f"  最终训练准确率: {history['train_acc'][-1]:.4f}")


if __name__ == '__main__':
    main()
