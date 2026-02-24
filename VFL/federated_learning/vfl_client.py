"""
垂直联邦学习客户端
包括主动方（Active Party）和被动方（Passive Party）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple, Optional, List
import copy
import numpy as np
from .privbox import PrivBoxProtocol


class VFLClient:
    """
    垂直联邦学习客户端基类
    """
    
    def __init__(self, party_id: int, X_train: np.ndarray, 
                 device: torch.device, batch_size: int = 64):
        """
        初始化VFL客户端
        
        Args:
            party_id: 参与方ID
            X_train: 训练数据（该方持有的特征）
            device: 计算设备
            batch_size: 批次大小
        """
        self.party_id = party_id
        self.device = device
        self.batch_size = batch_size
        
        # 将数据转换为张量
        if len(X_train.shape) == 3:  # (n_samples, height, width)
            # 添加通道维度
            X_train = np.expand_dims(X_train, axis=1)  # (n_samples, 1, height, width)
        
        self.X_train = torch.FloatTensor(X_train)
        self.num_samples = len(X_train)
        
        # 底层模型（将在训练时设置）
        self.bottom_model = None
        self.current_embeddings = None
        self.optimizer = None  # 优化器将在set_bottom_model时创建
    
    def get_num_samples(self) -> int:
        """获取数据量"""
        return self.num_samples
    
    def set_bottom_model(self, model: nn.Module, lr: float = 0.001):
        """
        设置底层模型
        
        Args:
            model: 底层特征提取模型
            lr: 学习率
        """
        self.bottom_model = model.to(self.device)
        self.optimizer = optim.Adam(self.bottom_model.parameters(), lr=lr)
    
    def forward(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算嵌入向量
        
        Args:
            batch_indices: 批次索引
            
        Returns:
            嵌入向量
        """
        if self.bottom_model is None:
            raise ValueError("底层模型未初始化")
        
        # 获取该批次的数据
        batch_data = self.X_train[batch_indices].to(self.device)
        
        # 计算嵌入
        self.bottom_model.train()
        embeddings = self.bottom_model(batch_data)
        
        # 保存当前嵌入（用于反向传播）
        self.current_embeddings = embeddings
        
        return embeddings
    
    def backward(self, grad_from_top: torch.Tensor):
        """
        反向传播
        
        Args:
            grad_from_top: 来自顶层的梯度
        """
        if self.current_embeddings is None:
            raise ValueError("没有前向传播的嵌入")
        
        # 直接使用梯度进行反向传播（不需要backward，因为已经在顶层backward了）
        # 将梯度赋值给嵌入向量
        if self.current_embeddings.grad is None:
            self.current_embeddings.grad = grad_from_top
        else:
            self.current_embeddings.grad += grad_from_top
    
    def update_model(self, optimizer: torch.optim.Optimizer):
        """
        更新底层模型
        
        Args:
            optimizer: 优化器
        """
        optimizer.step()
        optimizer.zero_grad()
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """获取模型权重"""
        if self.bottom_model is None:
            raise ValueError("底层模型未初始化")
        return copy.deepcopy(self.bottom_model.state_dict())
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """设置模型权重"""
        if self.bottom_model is None:
            raise ValueError("底层模型未初始化")
        self.bottom_model.load_state_dict(weights)


class VFLActiveParty(VFLClient):
    """
    主动方（Active Party）
    拥有标签，负责协调训练流程
    """
    
    def __init__(self, party_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 device: torch.device, batch_size: int = 64):
        """
        初始化主动方
        
        Args:
            party_id: 参与方ID
            X_train: 训练数据
            y_train: 训练标签
            device: 计算设备
            batch_size: 批次大小
        """
        super().__init__(party_id, X_train, device, batch_size)
        
        self.y_train = torch.LongTensor(y_train)
        
        # 创建数据加载器（仅包含索引）
        indices = torch.arange(len(y_train))
        dataset = TensorDataset(indices, self.y_train)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_batch(self, batch_idx: int):
        """获取批次数据"""
        # 返回批次索引和标签
        for i, (indices, labels) in enumerate(self.train_loader):
            if i == batch_idx:
                return indices, labels
        return None, None
    
    def get_all_batches(self):
        """获取所有批次"""
        return self.train_loader


class VFLPassiveParty(VFLClient):
    """
    被动方（Passive Party）
    仅拥有特征，不拥有标签
    """
    
    def __init__(self, party_id: int, X_train: np.ndarray, 
                 device: torch.device, batch_size: int = 64):
        """
        初始化被动方
        
        Args:
            party_id: 参与方ID
            X_train: 训练数据
            device: 计算设备
            batch_size: 批次大小
        """
        super().__init__(party_id, X_train, device, batch_size)
    
    def forward_by_indices(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        根据主动方提供的索引进行前向传播
        
        Args:
            batch_indices: 批次索引（由主动方提供）
            
        Returns:
            嵌入向量
        """
        return self.forward(batch_indices)


def create_vfl_parties(X_parties: List[np.ndarray], y_train: np.ndarray,
                      device: torch.device, batch_size: int = 64,
                      active_party_id: int = 0):
    """
    创建VFL参与方列表
    
    Args:
        X_parties: 各方的特征列表
        y_train: 训练标签
        device: 计算设备
        batch_size: 批次大小
        active_party_id: 主动方ID（拥有标签的一方）
        
    Returns:
        参与方列表
    """
    parties = []
    
    for i, X_party in enumerate(X_parties):
        if i == active_party_id:
            # 主动方
            party = VFLActiveParty(i, X_party, y_train, device, batch_size)
        else:
            # 被动方
            party = VFLPassiveParty(i, X_party, device, batch_size)
        
        parties.append(party)
    
    return parties
