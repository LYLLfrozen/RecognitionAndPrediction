"""
垂直联邦学习工具函数
包括特征分割、模型分割等
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy


def split_features_vertical(X: np.ndarray, num_parties: int, 
                            overlap_ratio: float = 0.0) -> List[np.ndarray]:
    """
    按特征维度分割数据（垂直联邦学习）
    不同参与方获得相同样本的不同特征
    
    Args:
        X: 特征数据 (n_samples, height, width) 或 (n_samples, features)
        num_parties: 参与方数量
        overlap_ratio: 特征重叠比例（0-1之间）
        
    Returns:
        参与方特征列表
    """
    if len(X.shape) == 3:  # (n_samples, height, width)
        # 对于图像数据，按宽度分割
        n_samples, height, width = X.shape
        
        # 计算每个参与方的特征宽度
        overlap_width = int(width * overlap_ratio)
        base_width = (width + overlap_width * (num_parties - 1)) // num_parties
        
        party_features = []
        for i in range(num_parties):
            start_idx = i * base_width - (i * overlap_width if i > 0 else 0)
            end_idx = min(start_idx + base_width + overlap_width, width)
            
            # 提取该参与方的特征
            X_party = X[:, :, start_idx:end_idx]
            party_features.append(X_party)
            
            print(f"参与方 {i+1} 特征形状: {X_party.shape} (列 {start_idx}-{end_idx})")
    
    elif len(X.shape) == 2:  # (n_samples, features)
        n_samples, n_features = X.shape
        
        # 计算每个参与方的特征数
        features_per_party = n_features // num_parties
        
        party_features = []
        for i in range(num_parties):
            start_idx = i * features_per_party
            if i == num_parties - 1:
                end_idx = n_features  # 最后一个参与方获取剩余所有特征
            else:
                end_idx = (i + 1) * features_per_party
            
            X_party = X[:, start_idx:end_idx]
            party_features.append(X_party)
            
            print(f"参与方 {i+1} 特征数: {X_party.shape[1]} (特征 {start_idx}-{end_idx})")
    
    else:
        raise ValueError(f"不支持的数据维度: {X.shape}")
    
    return party_features


def split_features_for_cnn(X: np.ndarray, num_parties: int) -> Tuple[List[np.ndarray], List[Tuple]]:
    """
    为CNN模型分割特征
    将11x11的图像分割给不同参与方
    
    Args:
        X: 图像数据 (n_samples, 1, 11, 11)
        num_parties: 参与方数量
        
    Returns:
        (参与方特征列表, 各参与方的形状列表)
    """
    n_samples, channels, height, width = X.shape
    
    if num_parties == 2:
        # 两方：左右分割
        mid = width // 2
        X_party1 = X[:, :, :, :mid]  # 左半部分
        X_party2 = X[:, :, :, mid:]  # 右半部分
        
        party_features = [X_party1, X_party2]
        shapes = [X_party1.shape[2:], X_party2.shape[2:]]
        
    elif num_parties == 3:
        # 三方：上、中、下分割
        h1 = height // 3
        h2 = 2 * height // 3
        
        X_party1 = X[:, :, :h1, :]  # 上部
        X_party2 = X[:, :, h1:h2, :]  # 中部
        X_party3 = X[:, :, h2:, :]  # 下部
        
        party_features = [X_party1, X_party2, X_party3]
        shapes = [X_party1.shape[2:], X_party2.shape[2:], X_party3.shape[2:]]
        
    elif num_parties == 4:
        # 四方：四象限分割
        mid_h = height // 2
        mid_w = width // 2
        
        X_party1 = X[:, :, :mid_h, :mid_w]  # 左上
        X_party2 = X[:, :, :mid_h, mid_w:]  # 右上
        X_party3 = X[:, :, mid_h:, :mid_w]  # 左下
        X_party4 = X[:, :, mid_h:, mid_w:]  # 右下
        
        party_features = [X_party1, X_party2, X_party3, X_party4]
        shapes = [X_party1.shape[2:], X_party2.shape[2:], 
                 X_party3.shape[2:], X_party4.shape[2:]]
    
    else:
        raise ValueError(f"当前仅支持2-4个参与方，收到: {num_parties}")
    
    return party_features, shapes


def create_vfl_model_split(num_parties: int, input_shapes: List[Tuple],
                           num_classes: int = 5) -> Tuple[nn.ModuleList, nn.Module]:
    """
    创建垂直联邦学习的分割模型
    
    Args:
        num_parties: 参与方数量
        input_shapes: 各参与方的输入形状列表 [(h1,w1), (h2,w2), ...]
        num_classes: 分类数量
        
    Returns:
        (各方的底层模型列表, 顶层模型)
    """
    # 各方的底层特征提取模型
    bottom_models = nn.ModuleList()
    
    for i, shape in enumerate(input_shapes):
        # 为每个参与方创建CNN特征提取器
        h, w = shape
        
        # 创建灵活的CNN模型
        layers = []
        
        # 第一层卷积
        layers.extend([
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        ])
        
        # 第一次池化（如果空间足够大）
        if min(h, w) > 2:
            layers.append(nn.MaxPool2d(2))
            h_after_pool1 = h // 2
            w_after_pool1 = w // 2
        else:
            h_after_pool1 = h
            w_after_pool1 = w
        
        layers.append(nn.Dropout(0.3))
        
        # 第二层卷积
        layers.extend([
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ])
        
        # 第二次池化（如果空间足够大）
        if min(h_after_pool1, w_after_pool1) > 2:
            layers.append(nn.MaxPool2d(2))
            h_final = h_after_pool1 // 2
            w_final = w_after_pool1 // 2
        else:
            h_final = h_after_pool1
            w_final = w_after_pool1
        
        layers.append(nn.Dropout(0.3))
        
        # 展平和全连接层
        layers.append(nn.Flatten())
        
        # 计算展平后的大小
        flattened_size = 32 * h_final * w_final
        
        # 添加全连接层到固定的嵌入维度
        layers.extend([
            nn.Linear(flattened_size, 64),
            nn.ReLU()
        ])
        
        bottom_model = nn.Sequential(*layers)
        bottom_models.append(bottom_model)
    
    # 顶层聚合模型
    top_model = nn.Sequential(
        nn.Linear(64 * num_parties, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes)
    )
    
    return bottom_models, top_model


def print_vfl_data_distribution(party_features: List[np.ndarray], 
                                y: Optional[np.ndarray] = None,
                                class_names: Optional[List[str]] = None):
    """
    打印垂直联邦学习的数据分布情况
    
    Args:
        party_features: 各参与方的特征列表
        y: 标签（可选）
        class_names: 类别名称列表（可选）
    """
    print("\n" + "="*70)
    print("垂直联邦学习 - 数据分布:")
    print("="*70)
    
    total_features = 0
    for i, X_party in enumerate(party_features):
        print(f"\n参与方 {i+1}:")
        print(f"  数据形状: {X_party.shape}")
        
        if len(X_party.shape) == 4:  # (n_samples, channels, height, width)
            n_features = X_party.shape[2] * X_party.shape[3]
        elif len(X_party.shape) == 3:  # (n_samples, height, width)
            n_features = X_party.shape[1] * X_party.shape[2]
        else:
            n_features = X_party.shape[1]
        
        total_features += n_features
        print(f"  特征数: {n_features}")
    
    print(f"\n总特征数: {total_features}")
    
    if y is not None:
        print(f"样本数: {len(y)}")
        print(f"\n标签分布:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            if class_names:
                cls_name = class_names[cls]
            else:
                cls_name = f"类别{cls}"
            percentage = count / len(y) * 100
            print(f"  {cls_name}: {count} ({percentage:.1f}%)")
    
    print("="*70)


def calculate_communication_cost(tensors: List[torch.Tensor]) -> float:
    """
    计算通信成本（MB）
    
    Args:
        tensors: 需要传输的张量列表
        
    Returns:
        通信成本（MB）
    """
    total_bytes = 0
    for tensor in tensors:
        total_bytes += tensor.element_size() * tensor.numel()
    
    return total_bytes / (1024 * 1024)  # 转换为MB
