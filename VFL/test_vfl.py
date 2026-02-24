#!/usr/bin/env python3
"""
测试垂直联邦学习系统
验证PrivBox协议和VFL训练流程
"""

import torch
import numpy as np
import sys
import os

# 切换到项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from federated_learning.privbox import SecretSharing, PrivBoxProtocol
from federated_learning.vfl_utils import split_features_for_cnn, create_vfl_model_split
from federated_learning.vfl_server import VFLServer
from federated_learning.vfl_client import VFLActiveParty, VFLPassiveParty

print("="*70)
print("测试垂直联邦学习系统")
print("="*70)

def test_secret_sharing():
    """测试秘密共享"""
    print("\n【测试1】秘密共享...")
    
    # 测试numpy数组
    secret = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    shares = SecretSharing.share(secret, num_parties=3)
    reconstructed = SecretSharing.reconstruct(shares)
    
    error = np.abs(secret - reconstructed).max()
    print(f"  原始秘密: {secret}")
    print(f"  重构秘密: {reconstructed}")
    print(f"  最大误差: {error}")
    assert error < 1e-6, "秘密共享重构失败"
    print("  ✓ 秘密共享测试通过")
    
    # 测试PyTorch张量
    secret_tensor = torch.randn(2, 3)
    shares_tensor = SecretSharing.share_tensor(secret_tensor, num_parties=3)
    reconstructed_tensor = SecretSharing.reconstruct_tensor(shares_tensor)
    
    error = (secret_tensor - reconstructed_tensor).abs().max().item()
    print(f"\n  张量秘密形状: {secret_tensor.shape}")
    print(f"  重构误差: {error}")
    assert error < 1e-6, "张量秘密共享重构失败"
    print("  ✓ 张量秘密共享测试通过")


def test_privbox_protocol():
    """测试PrivBox协议"""
    print("\n【测试2】PrivBox协议...")
    
    privbox = PrivBoxProtocol(num_parties=3, use_encryption=False)
    
    # 测试梯度保护
    gradient = torch.randn(4, 10)
    shares = privbox.protect_gradient(gradient)
    aggregated = privbox.secure_aggregate_gradients(shares)
    
    error = (gradient - aggregated).abs().max().item()
    print(f"  梯度形状: {gradient.shape}")
    print(f"  分享数: {len(shares)}")
    print(f"  聚合误差: {error}")
    assert error < 1e-5, "梯度保护和聚合失败"
    print("  ✓ PrivBox梯度保护测试通过")
    
    # 测试差分隐私噪声
    tensor = torch.ones(5, 5)
    noisy_tensor = privbox.add_dp_noise(tensor, epsilon=1.0)
    noise_level = (tensor - noisy_tensor).abs().mean().item()
    print(f"\n  添加DP噪声前: 全1张量")
    print(f"  噪声水平: {noise_level:.4f}")
    assert noise_level > 0, "差分隐私噪声未添加"
    print("  ✓ 差分隐私测试通过")


def test_feature_splitting():
    """测试特征分割"""
    print("\n【测试3】特征分割...")
    
    # 创建模拟数据
    X = np.random.randn(100, 1, 11, 11)  # 100个样本，11x11图像
    
    # 测试2方分割
    X_parties, shapes = split_features_for_cnn(X, num_parties=2)
    print(f"  原始数据: {X.shape}")
    print(f"  2方分割:")
    for i, (X_p, shape) in enumerate(zip(X_parties, shapes)):
        print(f"    参与方{i+1}: {X_p.shape}, 区域: {shape}")
    
    assert len(X_parties) == 2
    assert X_parties[0].shape[0] == X_parties[1].shape[0] == 100
    print("  ✓ 2方特征分割测试通过")
    
    # 测试3方分割
    X_parties, shapes = split_features_for_cnn(X, num_parties=3)
    print(f"\n  3方分割:")
    for i, (X_p, shape) in enumerate(zip(X_parties, shapes)):
        print(f"    参与方{i+1}: {X_p.shape}, 区域: {shape}")
    
    assert len(X_parties) == 3
    print("  ✓ 3方特征分割测试通过")


def test_model_creation():
    """测试VFL模型创建"""
    print("\n【测试4】VFL模型创建...")
    
    # 2方场景
    input_shapes = [(11, 6), (11, 5)]  # 左右分割11x11
    num_classes = 5
    
    bottom_models, top_model = create_vfl_model_split(2, input_shapes, num_classes)
    
    print(f"  底层模型数量: {len(bottom_models)}")
    print(f"  顶层模型输出: {num_classes}类")
    
    # 测试前向传播
    device = torch.device("cpu")
    batch_size = 4
    
    embeddings = []
    for i, (model, shape) in enumerate(zip(bottom_models, input_shapes)):
        model = model.to(device)
        x = torch.randn(batch_size, 1, shape[0], shape[1])
        emb = model(x)
        embeddings.append(emb)
        print(f"  参与方{i+1} 输入: {x.shape} -> 嵌入: {emb.shape}")
    
    # 测试顶层模型
    combined = torch.cat(embeddings, dim=-1)
    top_model = top_model.to(device)
    output = top_model(combined)
    
    print(f"  组合嵌入: {combined.shape}")
    print(f"  最终输出: {output.shape}")
    
    assert output.shape == (batch_size, num_classes)
    print("  ✓ VFL模型创建测试通过")


def test_vfl_training_step():
    """测试VFL训练一步"""
    print("\n【测试5】VFL训练步骤...")
    
    device = torch.device("cpu")
    batch_size = 8
    num_parties = 2
    num_classes = 5
    
    # 创建模拟数据
    X_train_1 = np.random.randn(100, 11, 6)
    X_train_2 = np.random.randn(100, 11, 5)
    y_train = np.random.randint(0, num_classes, 100)
    
    # 创建参与方
    active_party = VFLActiveParty(0, X_train_1, y_train, device, batch_size)
    passive_party = VFLPassiveParty(1, X_train_2, device, batch_size)
    
    # 创建模型
    input_shapes = [(11, 6), (11, 5)]
    bottom_models, top_model = create_vfl_model_split(num_parties, input_shapes, num_classes)
    
    active_party.set_bottom_model(bottom_models[0])
    passive_party.set_bottom_model(bottom_models[1])
    
    # 创建服务器
    server = VFLServer(top_model, device, num_parties, use_privbox=True)
    
    # 获取一个批次
    batches = list(active_party.get_all_batches())
    batch_indices, labels = batches[0]
    
    print(f"  批次索引: {batch_indices.shape}")
    print(f"  标签: {labels.shape}")
    
    # 前向传播
    emb1 = active_party.forward(batch_indices)
    emb2 = passive_party.forward(batch_indices)
    
    print(f"  主动方嵌入: {emb1.shape}")
    print(f"  被动方嵌入: {emb2.shape}")
    
    # 服务器聚合
    combined = server.aggregate_embeddings([emb1, emb2])
    combined.requires_grad_(True)
    combined.retain_grad()  # 保留中间张量的梯度
    
    print(f"  组合嵌入: {combined.shape}")
    
    # 顶层前向传播
    outputs = server.forward_top_model(combined)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    print(f"  输出: {outputs.shape}")
    print(f"  损失: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 分割梯度
    embedding_sizes = [emb1.shape[1], emb2.shape[1]]
    gradients = server.split_gradients(combined, embedding_sizes)
    
    print(f"  梯度1: {gradients[0].shape}")
    print(f"  梯度2: {gradients[1].shape}")
    
    print("  ✓ VFL训练步骤测试通过")


def main():
    """运行所有测试"""
    try:
        test_secret_sharing()
        test_privbox_protocol()
        test_feature_splitting()
        test_model_creation()
        test_vfl_training_step()
        
        print("\n" + "="*70)
        print("所有测试通过! ✓")
        print("="*70)
        print("\n可以运行 train_vfl.py 开始训练垂直联邦学习模型")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
